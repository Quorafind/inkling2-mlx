"""Real weights: do the decode glue fusions (decode_kernels) add error beyond BF16 noise?
Teacher-forces 200 tokens after a 3,000-token context, one token at a time, with the
fusions on and off, and compares both against the multi-token (prefill) path's logits
for the same positions.  python tests/decode_fusion_quality.py MODEL_DIR"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini import decode_kernels as DK  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

CTX, N = 3000, 200
bot = Inkling2Mini(sys.argv[1], speculative=False)
lm = bot.model.language_model
text = bot.tokenizer.encode(open(Path(__file__).resolve().parents[1] / "README.md").read())
seq = (text * 2)[: CTX + N]


def logprobs(a):
    a = a - a.max(-1, keepdims=True)
    return a - np.log(np.exp(a).sum(-1, keepdims=True))


def prefill_ref():
    cache = bot.model.make_cache(kv_bits=8)
    out = []
    ids = mx.array([seq])
    for s in range(0, CTX + N - 1, 1024):
        e = min(s + 1024, CTX + N - 1)
        lo = lm(ids[:, s:e], cache=cache, last_only=False)[0]
        keep = [i - s for i in range(s, e) if i >= CTX - 1]
        if keep:
            out.append(np.asarray(lo[keep[0]:].astype(mx.float32)))
        mx.eval(cache)
    return logprobs(np.concatenate(out))


def decode(enabled):
    DK.ENABLED = enabled
    try:
        cache = bot.model.make_cache(kv_bits=8)
        ids = mx.array([seq[:CTX - 1]])
        for s in range(0, CTX - 1, 1024):
            mx.eval(lm.model(ids[:, s:s + 1024], cache=cache))
        out = []
        for t in seq[CTX - 1: CTX + N - 1]:
            out.append(np.asarray(lm(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32)))
        return logprobs(np.stack(out))
    finally:
        DK.ENABLED = True


ref, on, off = prefill_ref(), decode(True), decode(False)


def cmp(a, b):
    kl = (np.exp(a) * (a - b)).sum(-1)
    return f"top-1 agree {(a.argmax(-1) == b.argmax(-1)).mean():.3f}, mean KL {kl.mean():.1e}, max KL {kl.max():.1e}"


print(f"{N} teacher-forced decode steps after {CTX} tokens (8-bit dense, 8-bit KV)")
print(f"  unfused decode vs prefill path: {cmp(ref, off)}")
print(f"  fused decode   vs prefill path: {cmp(ref, on)}")
print(f"  fused vs unfused decode:        {cmp(off, on)}")
