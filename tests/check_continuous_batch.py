"""Continuous batching == decoding each request alone (inkling2_mini/batching.py).

Tiny reference model, FP32 on the CPU (strict). Three prompts of different lengths (one
longer than the 8-token sliding window) are prefilled one by one into single-request
caches, then teacher-forced through a schedule where rows join and leave a running
batch at different times:

  A decodes alone (single-request caches), then joins a new batch;
  B joins at once, C joins three steps later (extend into a running batch);
  B leaves (extracted back into single-request caches) and finishes alone;
  A leaves last and finishes alone.

Every logit of every row must match running that row alone. A second part runs the
same schedule with 8-bit full-attention KV outside the batch (dequantized on joining,
quantized again on leaving). Its reference is the row alone with the same conversions at
the same steps, so it is strict too (the tiny random model is too chaotic to compare
against a differently rounded run).

    python tests/check_continuous_batch.py TINY_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.batching import extract_row, to_batch  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

mx.set_default_device(mx.cpu)
model = load(Path(sys.argv[1]), verbose=False)
lm = model.language_model
rng = np.random.default_rng(1)
prompts = {"A": rng.integers(0, 500, 6).tolist(), "B": rng.integers(0, 500, 13).tolist(),
           "C": rng.integers(0, 500, 3).tolist()}
STEPS = 14
forced = {k: rng.integers(0, 500, STEPS).tolist() for k in prompts}
JOIN = {"A": 3, "B": 3, "C": 6}   # global step at which a row enters the batch
LEAVE = {"A": 12, "B": 9, "C": 99}  # global step at which a row leaves (99: stays)
START = {"A": 0, "B": 3, "C": 6}   # global step of a row's first decode step


def reference(name, kv_bits=None):
    cache = model.make_cache(kv_bits=kv_bits)
    out = [lm(mx.array([prompts[name]]), cache=cache)[0, -1]]
    for t in range(STEPS):
        out.append(lm(mx.array([[forced[name][t]]]), cache=cache)[0, -1])
    return np.stack([np.asarray(x) for x in out])


def reference_converted(name, kv_bits):
    """The row alone, converted to batch-format numerics while the schedule has it in
    the batch (BF16-style KV for new tokens) and back to kv_bits when it leaves."""
    cache = model.make_cache(kv_bits=kv_bits)
    out = [lm(mx.array([prompts[name]]), cache=cache)[0, -1]]
    for t in range(STEPS):
        g = START[name] + t
        if g == JOIN[name]:
            cache = extract_row(to_batch(cache), 0, None)
        if g == LEAVE[name]:
            cache = extract_row(to_batch(cache), 0, kv_bits)
        out.append(lm(mx.array([[forced[name][t]]]), cache=cache)[0, -1])
    return np.stack([np.asarray(x) for x in out])


def scheduled(kv_bits=None):
    got = {k: [] for k in prompts}
    single, step_of = {}, {k: 0 for k in prompts}
    for k, p in prompts.items():
        single[k] = model.make_cache(kv_bits=kv_bits)
        got[k].append(np.asarray(lm(mx.array([p]), cache=single[k])[0, -1]))
    batch, rows = None, []
    for g in range(START["C"] + STEPS + 2):
        for k in prompts:  # joins
            if g == JOIN[k] and k not in rows:
                part = to_batch(single.pop(k))
                if batch is None:
                    batch = part
                else:
                    for c, p in zip(batch, part):
                        c.extend(p)
                rows.append(k)
        for k in list(rows):  # leaves
            if g == LEAVE[k]:
                i = rows.index(k)
                single[k] = extract_row(batch, i, kv_bits)
                keep = [j for j in range(len(rows)) if j != i]
                if keep:
                    for c in batch:
                        c.filter(mx.array(keep))
                else:
                    batch = None
                rows.remove(k)
        active = [k for k in rows if step_of[k] < STEPS]
        if not active:
            batch, rows = None, []
        elif len(active) < len(rows):  # finished rows leave without being kept
            keep = [rows.index(k) for k in active]
            for c in batch:
                c.filter(mx.array(keep))
            rows = active
        if rows:
            y = mx.array([[forced[k][step_of[k]]] for k in rows])
            out = lm(y, cache=batch)
            for i, k in enumerate(rows):
                got[k].append(np.asarray(out[i, -1]))
                step_of[k] += 1
        for k in list(single):  # rows decoding alone
            if g >= START[k] and step_of[k] < STEPS:
                got[k].append(np.asarray(lm(mx.array([[forced[k][step_of[k]]]]), cache=single[k])[0, -1]))
                step_of[k] += 1
    return {k: np.stack(v) for k, v in got.items()}


ok = True
print("BF16-style caches (FP32 on the CPU): strict")
got = scheduled()
for k in prompts:
    ref = reference(k)
    n = min(len(ref), len(got[k]))
    err = np.abs(got[k][:n] - ref[:n]).max() / np.abs(ref).max()
    good = err < 1e-4 and n == STEPS + 1
    ok &= good
    print(f"  row {k} (prompt {len(prompts[k]):2d}, {n - 1} steps, joins {JOIN[k]}, leaves "
          f"{LEAVE[k] if LEAVE[k] < 99 else '-'}): rel.err {err:.1e} {'ok' if good else 'MISMATCH'}")

print("8-bit full-attention KV outside the batch: strict, against the same conversions alone")
got8 = scheduled(kv_bits=8)
for k in prompts:
    ref = reference_converted(k, 8)
    err = np.abs(got8[k] - ref).max() / np.abs(ref).max()
    good = err < 1e-4
    ok &= good
    print(f"  row {k}: rel.err {err:.1e} {'ok' if good else 'MISMATCH'}")
print("CONTINUOUS BATCH OK" if ok else "CONTINUOUS BATCH MISMATCH")
sys.exit(0 if ok else 1)
