"""Cost of one target forward over L new tokens with a warm cache (the verify step
of speculative decoding), plus how many distinct experts L tokens touch.

    python tests/bench_verify.py MODEL_DIR
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402
from mlx_vlm.models.inkling2 import language as L  # noqa: E402

bot = Inkling2Mini(sys.argv[1])
lm = bot.model.language_model
messages = [{"role": "user", "content": "Write a short story about a lighthouse keeper who finds a message in a bottle. " * 8}]
ids, _, _ = bot.encode(messages, reasoning_effort="none")
cache = bot.model.make_cache()
mx.eval(lm(ids, cache=cache))
text = mx.array(bot.tokenizer.encode(" The keeper walked down to the shore every morning and watched the waves roll in."))

# Count distinct experts per layer application.
routes = []
orig_route = L._route


def spy(*a):
    idx, w = orig_route(*a)
    routes.append(idx)
    return idx, w


for n in (1, 2, 3, 4, 6, 8, 12, 16):
    toks = text[None, :n]
    times = []
    for rep in range(4):
        from mlx_vlm.speculative.cache_state import start_speculative_cache

        tx = start_speculative_cache(cache, n)
        L._route = spy if rep == 0 else orig_route
        routes.clear()
        t = time.perf_counter()
        out = lm(toks, cache=cache, last_only=False)
        mx.eval(out)
        times.append(time.perf_counter() - t)
        L._route = orig_route
        if rep == 0:
            uniq = np.mean([len(np.unique(np.asarray(r))) for r in routes])
        tx.abort()
    t = min(times[1:])
    print(f"L={n:2d}  forward {t * 1e3:6.1f} ms  ({t * 1e3 / n:5.1f} ms/token)  "
          f"distinct experts/application {uniq:5.1f} of {8 * n}")
