"""Is the GPU BF16 path any less accurate than a CPU BF16 path?

Ground truth: CPU with every non-expert weight promoted to FP32 (experts stay in
their native NVFP4 / MXFP8, which is exact by construction). Then
KL(truth || GPU-BF16) vs KL(truth || CPU-BF16) over a prompt's positions.

    python tests/precision_vs_fp32.py MODEL_DIR
"""

import gc
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

model_dir = Path(sys.argv[1])
bot = Inkling2Mini(model_dir, speculative=False, dense_bits=16)
msgs = [{"role": "user", "content": "List three primary colors, then name one fruit for each, and explain why."}]
ids, _, _ = bot.encode(msgs, reasoning_effort="none")
ids = mx.concatenate([ids, mx.array([bot.tokenizer.encode("<|message_model|><|content_text|>Here are three primary colors")])], axis=1)


def logits_on(model, device):
    with mx.stream(device):
        out = model.language_model(ids, last_only=False)
        mx.eval(out)
    return np.asarray(out[0].astype(mx.float32))


gpu = logits_on(bot.model, mx.gpu)
cpu = logits_on(bot.model, mx.cpu)
del bot
gc.collect()
mx.clear_cache()
truth = logits_on(load(model_dir, dtype=mx.float32), mx.cpu)


def kl(p_logits, q_logits):
    p = np.exp(p_logits - p_logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    q = np.exp(q_logits - q_logits.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
    return (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)


for name, x in (("GPU bf16", gpu), ("CPU bf16", cpu)):
    k = kl(truth, x)
    agree = (truth.argmax(-1) == x.argmax(-1)).mean()
    print(f"{name} vs CPU fp32: mean KL {k.mean():.2e}  median KL {np.median(k):.2e}  max KL {k.max():.2e}  top1 {agree:.3f}")
