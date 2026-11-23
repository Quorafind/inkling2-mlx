"""Real weights: ring-buffer decode (physical layout, rotated mask) vs the
chronological path, past the 1024-token window so the ring has wrapped.

    python tests/check_ring.py MODEL_DIR
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini import fused  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1], speculative=False)
lm = bot.model.language_model
text = bot.tokenizer.encode(open(Path(__file__).resolve().parents[1] / "README.md").read())
ids = mx.array([(text * 3)[:1500]])
forced = (text * 3)[1500:1530]


def run(chronological):
    fused.RING_DECODE = not chronological  # chronological: the multi-token path
    try:
        cache = bot.model.make_cache()
        for s in range(0, 1500, 500):
            mx.eval(lm.model(ids[:, s:s + 500], cache=cache))
        out = []
        for t in forced:
            out.append(np.asarray(lm(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32)))
        return np.stack(out)
    finally:
        fused.RING_DECODE = True


ring, chrono = run(False), run(True)
p = np.exp(chrono - chrono.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
q = np.exp(ring - ring.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
kl = (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)
print(f"30 decode steps after a 1500-token prefill: top1 agree {(ring.argmax(-1) == chrono.argmax(-1)).mean():.3f}  "
      f"mean KL {kl.mean():.1e}  max KL {kl.max():.1e}  max|dlogit| {np.abs(ring - chrono).max():.3f}")
