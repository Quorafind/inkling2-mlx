"""Real weights: prefill through the band-bias kernel vs the materialized-mask path.

    python tests/check_band_model.py MODEL_DIR [context]
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini import fused  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
bot = Inkling2Mini(sys.argv[1], speculative=False)
lm = bot.model.language_model
text = bot.tokenizer.encode(open(Path(__file__).resolve().parents[1] / "README.md").read())
ids = mx.array([(text * (n // len(text) + 2))[:n]])
cont = bot.tokenizer.encode(" In summary, the runtime keeps the checkpoint's native expert precision")


def run(use_band):
    fused.BAND_MIN_QUERIES = 16 if use_band else 10**9
    cache = bot.model.make_cache()
    bot._prefill(lm.model.embed(ids), cache, 1024, None)
    out = [lm(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32) for t in cont]
    mx.eval(out)
    return np.stack([np.asarray(o - mx.logsumexp(o)) for o in out])


a, b = run(False), run(True)
kl = (np.exp(a) * (a - b)).sum(-1)
print(f"{n}-token prefill, band kernel vs materialized mask: next-token mean KL {kl.mean():.1e}, "
      f"max KL {kl.max():.1e}, top1 agree {(a.argmax(-1) == b.argmax(-1)).mean():.3f} ({len(kl)} positions)")
