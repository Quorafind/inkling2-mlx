"""Quantized full-attention KV caches.

1. Tiny reference model, CPU FP32: kv_bits=8 / 4 stay close to the exact logits
   (one-shot vs chunked prefill + decode both go through the quantized cache).
2. Real weights (GPU): teacher-forced next-token KL vs BF16 KV after a long prompt,
   plus decode speed and full-attention KV memory.

    python tests/check_kvquant.py TINY_DIR MODEL_DIR [context]
"""
import sys, time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.loader import load  # noqa: E402

tiny, model_dir = Path(sys.argv[1]), Path(sys.argv[2])
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 8192

with mx.stream(mx.cpu):
    model = load(tiny, verbose=False)
    lm = model.language_model
    ids = mx.array(np.load(tiny / "ref_ids.npy"))
    ref = np.load(tiny / "ref_logits.npy")
    for bits in (8, 4):
        cache = model.make_cache(kv_bits=bits)
        steps = [lm(ids[:, :9], cache=cache, last_only=False)]
        for t in range(9, ids.shape[1]):
            steps.append(lm(ids[:, t : t + 1], cache=cache, last_only=False))
        got = np.asarray(mx.concatenate(steps, axis=1))
        err = np.abs(got - ref).max() / np.abs(ref).max()
        print(f"tiny model, {bits}-bit full-attention KV, prefill + decode: rel.err {err:.1e}, "
              f"argmax agree {(got.argmax(-1) == ref.argmax(-1)).mean():.3f}")
del model

from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(model_dir, speculative=False)
lm = bot.model.language_model
text = bot.tokenizer.encode(open(Path(__file__).resolve().parents[1] / "README.md").read())
prompt = mx.array([(text * (ctx // len(text) + 2))[:ctx]])
cont = bot.tokenizer.encode(" The keeper walked to the shore and looked at the long list of numbers on the page.")


def run(bits):
    cache = bot.model.make_cache(kv_bits=bits)
    bot._prefill(lm.model.embed(prompt), cache, 512, None)
    out = []
    for t in cont:
        out.append(lm(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32))
    mx.eval(out)
    for _ in range(3):
        mx.eval(lm(mx.array([[cont[0]]]), cache=cache))
    t0 = time.perf_counter()
    for _ in range(10):
        mx.eval(lm(mx.array([[cont[0]]]), cache=cache))
    ms = (time.perf_counter() - t0) / 10 * 1e3
    full = [c[0] for c in cache if not hasattr(c[0], "max_size")]
    kv_bytes = sum(getattr(c, "nbytes", 0) if bits else c.keys[..., : c.offset, :].nbytes * 2 for c in full)
    logp = np.stack([np.asarray(o - mx.logsumexp(o)) for o in out])
    return logp, ms, kv_bytes


base, ms0, kv0 = run(None)
print(f"real model, {ctx}-token context: BF16 KV decode {ms0:.1f} ms/token, full-attention KV {kv0 / 1e9:.2f} GB")
for bits in (8, 4):
    q, ms, kvb = run(bits)
    kl = (np.exp(base) * (base - q)).sum(-1)
    agree = (base.argmax(-1) == q.argmax(-1)).mean()
    print(f"  {bits}-bit KV: decode {ms:.1f} ms/token, KV {kvb / 1e9:.2f} GB, vs BF16 KV: mean KL {kl.mean():.1e}, "
          f"max KL {kl.max():.1e}, top1 {agree:.3f} ({len(kl)} positions)")
