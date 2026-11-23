"""Is 8-bit KV within the model's own numerical noise at long context?

After a long prompt, teacher-force a 200-token continuation and compare next-token
distributions: (a) BF16 KV with a different prefill chunking (an exactly equivalent
computation: the noise floor), (b) 8-bit full-attention KV.

    python tests/kv8_quality.py MODEL_DIR [context]
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
bot = Inkling2Mini(sys.argv[1], speculative=False)
lm = bot.model.language_model
text = bot.tokenizer.encode(open(Path(__file__).resolve().parents[1] / "README.md").read())
prompt = (text * (ctx // len(text) + 2))[:ctx]
prompt += bot.tokenizer.encode("\n\n用中文总结上面这份文档的要点：")


def logprobs(kv_bits, chunk, tokens):
    cache = bot.model.make_cache(kv_bits=kv_bits)
    logits, _ = bot._prefill(lm.model.embed(mx.array([prompt])), cache, chunk, None)
    def lp(x):
        x = x.reshape(-1).astype(mx.float32)
        return np.asarray(x - mx.logsumexp(x))

    out = [lp(logits)]
    for t in tokens[:-1]:
        out.append(lp(lm(mx.array([[t]]), cache=cache)))  # evaluated step by step
    return np.stack(out)


# the continuation: greedy with BF16 KV
cache = bot.model.make_cache()
logits, _ = bot._prefill(lm.model.embed(mx.array([prompt])), cache, 1024, None)
cont = []
for _ in range(200):
    tok = int(mx.argmax(logits.reshape(-1)))
    cont.append(tok)
    logits = lm(mx.array([[tok]]), cache=cache)
    mx.eval(logits)
del cache
mx.clear_cache()

def block_logprobs(tokens):
    """BF16 KV; the continuation as one multi-token block instead of step by step:
    the same math through different kernels (GEMM vs GEMV, two attention kernels)."""
    cache = bot.model.make_cache()
    logits, _ = bot._prefill(lm.model.embed(mx.array([prompt])), cache, 1024, None)
    first = logits.reshape(-1).astype(mx.float32)
    rest = lm(mx.array([tokens[:-1]]), cache=cache, last_only=False)[0].astype(mx.float32)
    out = mx.concatenate([first[None], rest], axis=0)
    out = out - mx.logsumexp(out, axis=-1, keepdims=True)
    mx.eval(out)
    return np.asarray(out)


ref = logprobs(None, 1024, cont)
for label, fn in (("BF16 KV, block instead of steps (noise floor)", lambda: block_logprobs(cont)),
                  ("8-bit KV", lambda: logprobs(8, 1024, cont))):
    q = fn()
    kl = (np.exp(ref) * (ref - q)).sum(-1)
    print(f"{ctx}-token context, {label:42s}: mean KL {kl.mean():.1e}  median {np.median(kl):.1e}  "
          f"p99 {np.quantile(kl, 0.99):.1e}  top1 {(ref.argmax(-1) == q.argmax(-1)).mean():.3f}")
