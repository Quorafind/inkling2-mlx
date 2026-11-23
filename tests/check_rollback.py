"""Strict speculative-rollback test on the tiny reference model (FP32, CPU).

Repeatedly verifies a block of L tokens through a cache transaction, commits only
the first k (as DSpark does after a rejection), and continues. Every committed
position's logits must equal plain one-token-at-a-time decoding. Covers the ring
cache's snapshot / replay rollback, the full-attention cache's trim (and its sink
slot), and the short-convolution state rollback; the 8-token window wraps.

    python tests/check_rollback.py TINY_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.speculative.cache_state import start_speculative_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.loader import load  # noqa: E402

mx.set_default_device(mx.cpu)
model = load(Path(sys.argv[1]), verbose=False)
lm = model.language_model
rng = np.random.default_rng(1)
prompt = rng.integers(0, 500, 6).tolist()
stream = rng.integers(0, 500, 40).tolist()

# Plain decoding: logits after each stream token.
cache = model.make_cache()
lm(mx.array([prompt]), cache=cache)
plain = [np.asarray(lm(mx.array([[t]]), cache=cache)[0, -1]) for t in stream]

# Speculative: blocks of L, keep k (k+1 positions retained), with varying sizes.
cache = model.make_cache()
lm(mx.array([prompt]), cache=cache)
got, pos, r = [], 0, 0
plan = [(4, 0), (1, 0), (1, 0), (5, 2), (3, 2), (1, 0), (8, 7), (6, 1), (2, 0), (1, 0), (7, 3), (4, 3), (8, 0), (1, 0), (5, 4)]
while pos < len(stream):
    L, k = plan[r % len(plan)]
    r += 1
    block = stream[pos : pos + L]
    if not block:
        break
    k = min(k, len(block) - 1)
    if L == 1:  # a plain decode step (in-place ring path), as in plain segments
        out = lm(mx.array([block]), cache=cache, last_only=False)[0]
    else:
        tx = start_speculative_cache(cache, len(block))
        out = lm(mx.array([block]), cache=cache, last_only=False)[0]
        tx.commit([k + 1])
    got.extend(np.asarray(out[: k + 1]))
    pos += k + 1

got = np.stack(got)
ref = np.stack(plain[: len(got)])
err = np.abs(got - ref).max() / np.abs(ref).max()
print(f"{len(got)} committed positions over {r} rounds: rel.err {err:.1e}")
print("ROLLBACK OK" if err < 1e-4 else "ROLLBACK MISMATCH")
