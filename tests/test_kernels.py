"""Fused attention-residual kernel (GPU) vs an exact FP32 reference on the CPU.

    python tests/test_kernels.py
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.kernels import attention_residual, attention_residual_reference  # noqa: E402

H = 2304
ok = True
for S, Q, L, norm, wrows in [(1, 1, 1, True, 1), (2, 1, 1, True, 1), (3, 3, 1, True, 3), (5, 1, 7, True, 1),
                             (4, 3, 33, False, 1), (2, 1, 1, False, 1), (5, 1, 1, True, 1)]:
    rng = np.random.default_rng(S * 100 + Q + L)
    states = [mx.array(rng.normal(0, 3, (2, L, H)).astype(np.float32)).astype(mx.bfloat16) for _ in range(S - 1)]
    states.append(mx.array(rng.normal(0, 3, (2, L, H)).astype(np.float32)))  # FP32 partial block
    query = mx.array(rng.normal(0, 1, (Q, H)).astype(np.float32))
    w = mx.array(rng.normal(1, 0.2, (wrows, H)).astype(np.float32)).astype(mx.bfloat16) if norm else None
    with mx.stream(mx.cpu):
        rm, rn = attention_residual_reference(states, query, w)
        mx.eval(rm, rn) if norm else mx.eval(rm)
    m, n = attention_residual(states, query, w, write_mixed=True)
    em = float(mx.abs(m - rm).max() / mx.abs(rm).max())
    line = f"S={S} Q={Q} L={L:2d} norm={norm}: mixed rel.err {em:.1e}"
    good = em < 1e-5
    if norm:
        en = float(mx.abs(n.astype(mx.float32) - rn.astype(mx.float32)).max() / mx.abs(rn.astype(mx.float32)).max())
        line += f"  normed rel.err {en:.1e}"
        good &= en < 1e-2  # BF16 output rounding
    ok &= good
    print(("ok   " if good else "FAIL ") + line)
print("KERNELS OK" if ok else "KERNEL MISMATCH")
