"""Quality cost of --dense-bits: KL of next-token distributions vs the native model
over a teacher-forced ~400-token chat transcript (GPU). For scale, the BF16-vs-FP32
noise measured by tests/precision_vs_fp32.py is a median KL of about 1e-3.

    python tests/dense_quality.py MODEL_DIR
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
msgs = [{"role": "user", "content": "Explain how a hash map works, then give a short Python example and a one-sentence summary in Chinese."}]
ids, _, _ = bot.encode(msgs, reasoning_effort="none")
toks = [t for t, _ in bot.generate_tokens(ids, max_tokens=400, temperature=0, speculative=False)]
seq = mx.concatenate([ids, mx.array([toks])], axis=1)
start = ids.shape[1] - 1


def logp(model):
    out = model.language_model(seq, last_only=False)[0, start:-1].astype(mx.float32)
    out = out - mx.logsumexp(out, axis=-1, keepdims=True)
    return np.asarray(out)


ref = logp(bot.model)
del bot
for bits in (8, 6, 4):
    gc.collect()
    mx.clear_cache()
    model = load(model_dir, verbose=False, dense_bits=bits)
    q = logp(model)
    del model
    kl = (np.exp(ref) * (ref - q)).sum(-1)
    agree = (ref.argmax(-1) == q.argmax(-1)).mean()
    print(f"dense {bits}-bit vs native: mean KL {kl.mean():.2e}  median {np.median(kl):.2e}  "
          f"p99 {np.quantile(kl, 0.99):.2e}  top1 {agree:.3f}  ({len(kl)} positions)")
