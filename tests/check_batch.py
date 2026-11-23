"""Batched decoding == per-request decoding.

Tiny reference model, FP32 on the CPU (strict): prompts of different lengths are
prefilled one by one, their 48 per-visit caches merged into left-padded batch
caches, then teacher-forced tokens are decoded for all rows together (with one
row filtered out half way). Every row's logits must match running that row alone.
The longest prompt exceeds the 8-token sliding window, so a rotated cache is merged.

    python tests/check_batch.py TINY_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.cache import CacheList

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.loader import load  # noqa: E402

mx.set_default_device(mx.cpu)
model = load(Path(sys.argv[1]), verbose=False)
lm = model.language_model
rng = np.random.default_rng(0)
prompts = [rng.integers(0, 500, n).tolist() for n in (5, 13, 9)]
forced = rng.integers(0, 500, (3, 10)).tolist()
STEPS, DROP_AT, DROP_ROW = 10, 4, 1


def single(prompt, steps):
    cache = model.make_cache()
    out = [lm(mx.array([prompt]), cache=cache)[0, -1]]
    return cache, out


# Reference: each row alone.
ref = []
for r, p in enumerate(prompts):
    cache = model.make_cache()
    logits = [lm(mx.array([p]), cache=cache)[0, -1]]
    for t in range(STEPS):
        logits.append(lm(mx.array([[forced[r][t]]]), cache=cache)[0, -1])
    ref.append(np.stack([np.asarray(x) for x in logits]))

# Batched.
caches, got = [], [[] for _ in prompts]
for r, p in enumerate(prompts):
    c = model.make_cache()
    got[r].append(np.asarray(lm(mx.array([p]), cache=c)[0, -1]))
    caches.append(c)
batch = [CacheList.merge([c[i] for c in caches]) for i in range(len(caches[0]))]
rows = [0, 1, 2]
for t in range(STEPS):
    if t == DROP_AT:
        keep = [i for i, r in enumerate(rows) if r != DROP_ROW]
        for c in batch:
            c.filter(mx.array(keep))
        rows = [rows[i] for i in keep]
    y = mx.array([[forced[r][t]] for r in rows])
    out = lm(y, cache=batch)
    for i, r in enumerate(rows):
        got[r].append(np.asarray(out[i, -1]))

ok = True
for r in range(3):
    g = np.stack(got[r])
    n = len(g)
    err = np.abs(g - ref[r][:n]).max() / np.abs(ref[r][:n]).max()
    good = err < 1e-4
    ok &= good
    print(f"row {r} (prompt {len(prompts[r]):2d} tok, {n - 1} decode steps): rel.err {err:.1e} {'ok' if good else 'MISMATCH'}")
print("BATCH OK" if ok else "BATCH MISMATCH")
