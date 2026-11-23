"""Grouped NVFP4 expert matvec vs mx.gather_qmm on real expert weights, plus timing.
    python tests/test_grouped.py MODEL_DIR
"""
import sys, time
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.loader import load  # noqa: E402
from inkling2_mini.kernels import grouped_nvfp4_qmv, grouped_nvfp4_qmv_run  # noqa: E402

model = load(Path(sys.argv[1]), dense_bits=8)
sw = model.language_model.model.layers[3].mlp.switch_mlp
rng = np.random.default_rng(0)
for R, uniq in [(16, 11), (64, 29), (128, 39), (128, 60), (256, 70)]:
    experts = np.sort(rng.choice(128, uniq, replace=False))
    ids = np.sort(np.concatenate([experts, rng.choice(experts, R - uniq)])).astype(np.int32)
    idx = mx.array(ids)
    for name, lin, k in (("gate_up", sw.gate_up, 2304), ("down", sw.down_proj, 1152)):
        x = mx.random.normal((R, k)).astype(mx.bfloat16)
        ref = lin(x.reshape(R, 1, 1, k), idx.reshape(R, 1), sorted_indices=True).reshape(R, -1)
        times = {}
        for label, f in [("gather_qmm", lambda: lin(x.reshape(R, 1, 1, k), idx.reshape(R, 1), sorted_indices=True)),
                         ("grouped V=4", lambda: grouped_nvfp4_qmv(x, lin, idx, 4)),
                         ("run-8", lambda: grouped_nvfp4_qmv_run(x, lin, idx, 8)),
                         ("run-16", lambda: grouped_nvfp4_qmv_run(x, lin, idx, 16))]:
            mx.eval(f()); t = time.perf_counter()
            for _ in range(50): mx.eval(f())
            times[label] = (time.perf_counter() - t) / 50 * 1e3
        got = grouped_nvfp4_qmv_run(x, lin, idx, 16)
        err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max() / mx.abs(ref.astype(mx.float32)).max())
        print(f"R={R:3d} uniq={uniq:3d} {name:7s} rel.err {err:.1e}  " + "  ".join(f"{k} {v:.3f} ms" for k, v in times.items()))
