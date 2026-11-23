"""Decode glue kernels vs the MLX op chains they replace, on real weights.
    python tests/test_decode_kernels.py MODEL_DIR"""
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_vlm.models.inkling2.language import _route  # noqa: E402
from inkling2_mini import decode_kernels as DK  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

model = load(Path(sys.argv[1]), verbose=False, dense_bits=8)
layers = model.language_model.model.layers
ok = True


def bench(f, n=200):
    mx.eval(f())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - t) / n * 1e3


rng = np.random.default_rng(0)
print("router")
for li in (0, 11, 20):
    moe = layers[li].mlp
    for T in (1, 8):
        x = mx.array(rng.normal(0, 1, (1, T, 2304)).astype(np.float32)).astype(mx.bfloat16)
        ri, rw = _route(x, moe.gate_weight, moe.e_score_correction_bias, moe.top_k, moe.route_scale)
        gi, gw = DK.route(x, moe.gate_weight, moe.e_score_correction_bias, moe.top_k, moe.route_scale)
        rs = [sorted(zip(a, b)) for a, b in zip(np.asarray(ri).reshape(T, -1).tolist(), np.asarray(rw).reshape(T, -1).tolist())]
        gs = [sorted(zip(a, b)) for a, b in zip(np.asarray(gi).reshape(T, -1).tolist(), np.asarray(gw).reshape(T, -1).tolist())]
        same_sets = all([i for i, _ in a] == [i for i, _ in b] for a, b in zip(rs, gs))
        werr = max(abs(wa - wb) for a, b in zip(rs, gs) for (_, wa), (_, wb) in zip(a, b))
        good = same_sets and werr < 1e-5
        ok &= good
        line = f"  layer {li:2d} T={T}: same experts {same_sets}, max weight diff {werr:.1e} {'ok' if good else 'MISMATCH'}"
        if T == 1:
            t0 = bench(lambda: _route(x, moe.gate_weight, moe.e_score_correction_bias, moe.top_k, moe.route_scale))
            wt = moe.gate_weight.astype(mx.float32).T
            mx.eval(wt)
            t1 = bench(lambda: DK.route(x, moe.gate_weight, moe.e_score_correction_bias, moe.top_k, moe.route_scale, wt))
            line += f"   (synced: mlx-vlm {t0:.3f} ms, fused {t1:.3f} ms)"
        print(line)
print("DECODE KERNELS OK" if ok else "DECODE KERNELS MISMATCH")

# attention prep vs the reference op chain (same GPU inputs)
from mlx_vlm.models.inkling2.language import l2_clip  # noqa: E402

print("attention prep")
cfg = model.language_model.config
for li, L, offset in ((1, 1, 500), (5, 1, 9000), (5, 7, 20000), (11, 300, 0)):
    a = layers[li].self_attn
    y = mx.array(rng.normal(0, 2, (1, L, sum(a._sizes))).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.normal(0, 2, (1, L, 1024)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.normal(0, 2, (1, L, 1024)).astype(np.float32)).astype(mx.bfloat16)
    nq, ng, nr = a._sizes[:3]
    o2 = nq + ng
    ls = (cfg.log_scaling_alpha, cfg.log_scaling_n_floor) if not a.is_sliding else None
    q1, k1, v1, r1 = DK.attn_prep(y, k, v, a.q_scale, a.rel_scale, o2, 24, 8, 16, cfg.qk_l2_clip_threshold,
                                  cfg.rel_l2_clip_threshold, cfg.rms_norm_eps, offset, ls)
    q0 = l2_clip(y[..., :nq].reshape(1, L, 24, 128), cfg.qk_l2_clip_threshold, a.q_scale)
    k0 = l2_clip(k.reshape(1, L, 8, 128), cfg.qk_l2_clip_threshold)
    v0 = mx.fast.rms_norm(v.reshape(1, L, 8, 128).astype(mx.float32), None, cfg.rms_norm_eps).astype(mx.bfloat16)
    r0 = l2_clip(y[..., o2:o2 + nr].reshape(1, L, 24, 16), cfg.rel_l2_clip_threshold, a.rel_scale)
    if ls:
        pos = mx.arange(L) + offset + 1
        tau = 1 + cfg.log_scaling_alpha * mx.log(mx.maximum(pos / cfg.log_scaling_n_floor, 1))
        q0 = (q0.astype(mx.float32) * tau[..., None, None]).astype(mx.bfloat16)
        r0 = (r0.astype(mx.float32) * tau[..., None, None]).astype(mx.bfloat16)
    for name, g_, r_ in (("q", q1, q0.transpose(0, 2, 1, 3)), ("k", k1, k0.transpose(0, 2, 1, 3)),
                         ("v", v1, v0.transpose(0, 2, 1, 3)), ("r", r1, r0)):
        g_, r_ = np.asarray(g_.astype(mx.float32)), np.asarray(r_.astype(mx.float32))
        eq = (g_ == r_).mean()
        err = np.abs(g_ - r_).max() / max(np.abs(r_).max(), 1e-9)
        good = err < 1e-2 and eq > 0.99
        ok &= good
        print(f"  layer {li:2d} L={L:3d} offset={offset:5d} {name}: bit-identical {eq:.4f}, rel.err {err:.1e} {'ok' if good else 'MISMATCH'}")
print("DECODE KERNELS OK" if ok else "DECODE KERNELS MISMATCH")

# ring decode attention vs mask + SDPA over the chronological window
from inkling2_mini.caches import SinkRingKVCache  # noqa: E402
from inkling2_mini.decode_attention import ring_decode_attention  # noqa: E402
from mlx_vlm.models.inkling.language import banded_additive_mask  # noqa: E402

print("ring decode attention")
W, REL = 1024, 1024
for total in (5, 700, 1024, 1025, 3000):
    ring = SinkRingKVCache(W)
    allk = mx.array(rng.normal(0, 1.5, (1, 8, total, 128)).astype(np.float32)).astype(mx.bfloat16)
    allv = mx.array(rng.normal(0, 1, (1, 8, total, 128)).astype(np.float32)).astype(mx.bfloat16)
    if total > 1:
        ring.update_and_fetch(allk[:, :, : total - 1], allv[:, :, : total - 1])
    kb, vb, oldest, n = ring.update_and_fetch_with_sink(allk[:, :, -1:], allv[:, :, -1:])
    q = mx.array(rng.normal(0, 1.5, (1, 24, 1, 128)).astype(np.float32)).astype(mx.bfloat16)
    r = mx.array(rng.normal(0, 0.3, (1, 1, 24, 16)).astype(np.float32)).astype(mx.bfloat16)
    proj = mx.array(rng.normal(0, 1, (16, REL)).astype(np.float32)).astype(mx.bfloat16)
    sinks = mx.array(rng.normal(0, 2, (1, 24)).astype(np.float32)).astype(mx.bfloat16)
    got = ring_decode_attention(q, kb, vb, n, oldest, W, (r @ proj).reshape(1, 24, REL), sinks, REL)
    ck, cv = allk[:, :, -n:], allv[:, :, -n:]  # chronological window
    m = banded_additive_mask(r, proj, n - 1, n, W, REL, shape_ref=ck)
    m = mx.concatenate([m, sinks.reshape(1, 24, 1, 1)], -1)
    z = mx.zeros((1, 8, 1, 128), mx.bfloat16)
    ref = mx.fast.scaled_dot_product_attention(q, mx.concatenate([ck, z], 2), mx.concatenate([cv, z], 2), scale=128**-0.5, mask=m)
    err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max() / mx.abs(ref.astype(mx.float32)).max())
    good = err < 1e-2
    ok &= good
    print(f"  {total:5d} tokens through the ring (n={n}, oldest={oldest}): rel.err {err:.1e} {'ok' if good else 'MISMATCH'}")
print("DECODE KERNELS OK" if ok else "DECODE KERNELS MISMATCH")

# single-token NVFP4 experts vs gather_qmm + SwiGLU + combine (the model's own path)
print("single-token experts")
from inkling2_mini import model as M  # noqa: E402


def chain(f, x0, n=40):
    def run():
        x = x0
        for i in range(n):
            x = f(x, i)
        return x
    mx.eval(run())
    t = time.perf_counter()
    for _ in range(5):
        mx.eval(run())
    return (time.perf_counter() - t) / 5 / n * 1e3


cfgt = model.language_model.config
for li in (0, 7, 11, 19):
    moe = layers[li].mlp
    sw = moe.switch_mlp
    x = mx.array(rng.normal(0, 1, (1, 1, 2304)).astype(np.float32)).astype(mx.bfloat16)
    idx, w = DK.route(x, moe.gate_weight, moe.e_score_correction_bias, moe.top_k, moe.route_scale)
    DK.ENABLED = False
    ref = moe(x)
    DK.ENABLED = True
    got = DK.moe_decode(x, sw, idx, w, moe.inter, cfgt.mlp_gate_softcap, cfgt.mlp_up_softcap)
    r_, g_ = np.asarray(ref.astype(mx.float32)), np.asarray(got.astype(mx.float32))
    err = np.abs(r_ - g_).max() / np.abs(r_).max()
    eq = (r_ == g_).mean()
    good = err < 1e-2
    ok &= good
    print(f"  layer {li:2d}: bit-identical {eq:.4f}, rel.err {err:.1e} {'ok' if good else 'MISMATCH'}")
pool = [layers[j].mlp for j in range(20)]
xs = [mx.array(rng.normal(0, 1, (1, 1, 2304)).astype(np.float32)).astype(mx.bfloat16) for _ in range(8)]
routes = [DK.route(xs[j], pool[j].gate_weight, pool[j].e_score_correction_bias, 8, pool[j].route_scale) for j in range(8)]


def via_mlx(x, i):
    m = pool[i % 20]
    xe = mx.expand_dims(x, (-2, -3))
    ix, wt = routes[i % 8]
    gu = m.switch_mlp.gate_up(xe, ix)
    o = m.switch_mlp.down_proj(m.switch_mlp.activation(gu[..., m.inter:], gu[..., :m.inter]), ix)
    from inkling2_mini import prefill_kernels as PK
    return PK.combine_routes(o.reshape(-1, 2304), mx.arange(8, dtype=mx.uint32), wt.reshape(1, -1)).reshape(1, 1, 2304)


def via_kernel(x, i):
    m = pool[i % 20]
    ix, wt = routes[i % 8]
    return DK.moe_decode(x, m.switch_mlp, ix, wt, m.inter, cfgt.mlp_gate_softcap, cfgt.mlp_up_softcap)


nb = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(pool[0].switch_mlp.parameters())) / 128 * 8
for name, f in (("gather_qmm + SwiGLU + combine", via_mlx), ("fused decode kernels", via_kernel)):
    ms = chain(f, xs[0])
    print(f"  {name:30s} {ms * 1e3:6.1f} us per visit ({nb / ms / 1e6:.0f} GB/s over {nb / 1e6:.1f} MB)")
print("DECODE KERNELS OK" if ok else "DECODE KERNELS MISMATCH")
