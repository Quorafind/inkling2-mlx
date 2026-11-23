"""Prefill glue kernels vs the MLX op chains they replace (the mlx-vlm modules
themselves on the CPU), on real layer weights.

    python tests/test_prefill_kernels.py MODEL_DIR
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_vlm.models.inkling2.language import _combine_routes  # noqa: E402
from mlx_vlm.models.switch_layers import _scatter_unsort  # noqa: E402
from inkling2_mini import prefill_kernels as P  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

model = load(Path(sys.argv[1]), verbose=False, dense_bits=8)
layers = model.language_model.model.layers
ok = True


def check(name, got, ref, tol=0):
    global ok
    g, r = np.asarray(got.astype(mx.float32)), np.asarray(ref.astype(mx.float32))
    err = np.abs(g - r).max()
    exact = (g == r).mean()
    good = err <= tol * max(np.abs(r).max(), 1e-6)
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} {name:44s} max|diff| {err:.2e}  bit-identical {exact:.4f}")


rng = np.random.default_rng(0)
for L in (37, 1024):
    print(f"L={L}")
    sconv = layers[1].attn_sconv
    x = mx.array(rng.normal(0, 2, (1, L, 2304)).astype(np.float32)).astype(mx.bfloat16)
    state = mx.array(rng.normal(0, 2, (1, 3, 2304)).astype(np.float32))
    res = mx.array(rng.normal(0, 2, (1, L, 2304)).astype(np.float32))

    class C:  # minimal conv cache holding the state
        def __init__(self, s): self.c = [None, None, s, None]
        def __getitem__(self, i): return self.c[i]
        def __setitem__(self, i, v): self.c[i] = v
    with mx.stream(mx.cpu):
        ref = sconv(x, C(state), None)
        mx.eval(ref)
    check("short conv (attn, 2304 ch)", P.short_conv(x, state, sconv.conv.weight), ref, 1e-2)
    check("short conv + FP32 residual", P.short_conv(x, state, sconv.conv.weight, res, mx.float32),
          ref.astype(mx.float32) + res, 1e-2)
    a = layers[5].self_attn
    kconv = a.k_sconv
    kx = mx.array(rng.normal(0, 2, (1, L, 1024)).astype(np.float32)).astype(mx.bfloat16)
    din = mx.array(rng.normal(0, 1, (1, L, 2304)).astype(np.float32)).astype(mx.bfloat16)
    kst = mx.array(rng.normal(0, 2, (1, 3, 1024)).astype(np.float32))
    dyn = kconv.weight_proj(din).astype(mx.float32)
    mx.eval(dyn)
    orig_proj = kconv.weight_proj
    kconv.weight_proj = lambda _: dyn  # same dynamic taps on both sides
    with mx.stream(mx.cpu):
        ref = kconv(kx, din, [kst, None, None, None], None)
        mx.eval(ref)
    kconv.weight_proj = orig_proj
    check("dynamic conv (full-layer K)", P.dynamic_conv(kx, kst, kconv.static_weight, dyn, kconv.num_heads), ref, 1e-2)

    # MoE combine
    K, H = 8, 2304
    idx = mx.array(rng.integers(0, 128, (1, L, K)).astype(np.uint32))
    flat = idx.reshape(-1)
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    y = mx.array(rng.normal(0, 1, (L * K, 1, H)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.random((1, L, K)).astype(np.float32))
    ref = _combine_routes(_scatter_unsort(y, inv, idx.shape).squeeze(-2), w)
    check("un-sort + weighted combine", P.combine_routes(y.reshape(L * K, H), inv, w.reshape(L, K)).reshape(1, L, H), ref, 1e-2)
print("PREFILL KERNELS OK" if ok else "PREFILL KERNEL MISMATCH")
