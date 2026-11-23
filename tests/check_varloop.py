"""Pass counts other than 3 (variable-loop checkpoints, TML-d24-varloop-research).

Tiny reference model, FP32 on the CPU (strict), against the transformers reference run
with loop_num_passes = 1, 2, 4, 5 (make_tiny_reference.py records those logits):

  1. one-shot forward with the config's pass count set
  2. chunked prefill + decode through caches made for that pass count
     (make_cache(passes=N); the forward pass reads N from the number of caches)
  3. states of different pass counts interleaved on one loaded model
  4. the DSpark capture visits map to the same (pass, layer) positions

    python tests/check_varloop.py TINY_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.loader import load  # noqa: E402

tiny = Path(sys.argv[1])
mx.set_default_device(mx.cpu)
model = load(tiny, verbose=False)
lm = model.language_model
cfg = lm.config
ids = mx.array(np.load(tiny / "ref_ids.npy"))
refs = {3: np.load(tiny / "ref_logits.npy")}
for n in (1, 2, 4, 5):
    refs[n] = np.load(tiny / f"ref_logits_p{n}.npy")
ok = True


def compare(name, got, want, tol=1e-4):
    global ok
    got = np.asarray(got.astype(mx.float32))
    rel = np.abs(got - want).max() / np.abs(want).max()
    good = rel < tol
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} {name:44s} rel={rel:.1e}")


def cached(passes):
    cache = model.make_cache(passes=passes)
    assert len(cache) == cfg.visits(passes) and cfg.passes_for(len(cache)) == passes
    steps = [lm(ids[:, :5], cache=cache, last_only=False), lm(ids[:, 5:13], cache=cache, last_only=False)]
    for t in range(13, ids.shape[1]):
        steps.append(lm(ids[:, t : t + 1], cache=cache, last_only=False))
    return mx.concatenate(steps, axis=1)


for n in (1, 2, 3, 4, 5):
    print(f"{n} pass{'es' if n > 1 else ''} ({cfg.visits(n)} visits)")
    cfg.loop_config["num_passes"] = n
    compare("one-shot forward", lm(ids, last_only=False), refs[n])
    cfg.loop_config["num_passes"] = 3  # the caches alone must carry the pass count
    compare("chunked prefill + decode, caches for N passes", cached(n), refs[n])

print("interleaved states (1, 3 and 5 passes on one model)")
caches = {n: model.make_cache(passes=n) for n in (1, 3, 5)}
outs = {n: [lm(ids[:, :8], cache=caches[n], last_only=False)] for n in caches}
for t in range(8, ids.shape[1]):
    for n in caches:
        outs[n].append(lm(ids[:, t : t + 1], cache=caches[n], last_only=False))
for n in caches:
    compare(f"{n} passes, interleaved", mx.concatenate(outs[n], axis=1), refs[n])

print("DSpark capture visits ([5, 23, 41, 48] at 3 passes; tiny: loop 2-6, 8 layers)")
s_, n_ = cfg.loop_start, cfg.num_loop_layers


def where(v, passes):
    if v < s_:
        return ("pre", v)
    if v >= cfg.visits(passes):
        return ("final",)
    if v >= s_ + passes * n_:
        return ("post", v - passes * n_ + n_)
    p, j = divmod(v - s_, n_)
    return ("loop", p, s_ + j)


trained = [1, s_ + n_ + 1, s_ + 3 * n_ - 1, cfg.visits(3)]  # pre, pass 2, last pass, final
for n in (1, 2, 4, 5):
    mapped = [cfg.map_visit(v, n, 3) for v in trained]
    got = [where(v, n) for v in mapped]
    want = [("pre", 1), ("loop", min(n // 2, n - 1), s_ + 1), ("loop", n - 1, s_ + n_ - 1), ("final",)]
    good = got == want and mapped == sorted(set(mapped))
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} {n} passes: {trained} -> {mapped} {got}")
print("VARLOOP OK" if ok else "VARLOOP MISMATCH")
sys.exit(0 if ok else 1)
