"""8-bit KV fused decode attention vs an exact FP32 reference over the same
dequantized cache (CPU), plus timing against BF16-KV decode (mask + fused SDPA).

    python tests/test_decode_attention.py
"""
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_vlm.models.inkling.language import banded_additive_mask  # noqa: E402
from inkling2_mini.decode_attention import q8_decode_attention  # noqa: E402

B, H, KV, D, REL = 1, 24, 8, 128, 1024


def bench(f, n=20):
    mx.eval(f())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - t) / n * 1e3


ok = True
for N, L in [(5, 1), (700, 1), (1024, 1), (1500, 1), (3, 3), (40, 8), (1500, 5), (2048, 8), (4096, 1), (32768, 1),
             (65536, 1), (65536, 8), (131072, 1)]:
    rng = np.random.default_rng(N)
    q = mx.array(rng.normal(0, 1.5, (B, H, L, D)).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.normal(0, 1.5, (B, KV, N, D)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.normal(0, 1, (B, KV, N, D)).astype(np.float32)).astype(mx.bfloat16)
    cap = N + 200  # strided views of a larger cache buffer, as in QuantSinkKVCache
    pad = lambda a: mx.concatenate([a, mx.zeros((*a.shape[:2], cap - N, a.shape[3]), a.dtype)], axis=2)  # noqa: E731
    qk0, qv0 = mx.quantize(k, 64, 8), mx.quantize(v, 64, 8)
    mx.eval(qk0, qv0)
    qk = [pad(a)[..., :N, :] for a in qk0]
    qv = [pad(a)[..., :N, :] for a in qv0]
    rel = mx.array(rng.normal(0, 1, (B, H, L, REL)).astype(np.float32)).astype(mx.bfloat16)
    sinks = mx.array(rng.normal(0, 2, (B, H, L)).astype(np.float32)).astype(mx.bfloat16)
    got = q8_decode_attention(q, qk, qv, rel, sinks, REL)
    with mx.stream(mx.cpu):
        # the very same quantized cache, dequantized exactly in FP32
        kd = (qk0[1].astype(mx.float32)[..., None] * 0 + 0)  # placeholder shape helper (unused)
        def deq(qq):
            w, sc, bi = qq
            vals = mx.stack([(w >> (8 * i)) & 0xFF for i in range(4)], axis=-1).reshape(*w.shape[:-1], -1).astype(mx.float32)
            vals = vals.reshape(*vals.shape[:-1], -1, 64)
            return (vals * sc.astype(mx.float32)[..., None] + bi.astype(mx.float32)[..., None]).reshape(*w.shape[:-1], -1)
        kd, vd = deq(qk0), deq(qv0)
        qf = q.astype(mx.float32).reshape(B, KV, 3, L, D)
        logits = (qf @ kd[:, :, None].swapaxes(-1, -2)) * D**-0.5  # (B, KV, 3, L, N)
        qpos = mx.arange(N - L, N)[:, None]
        dist = qpos - mx.arange(N)[None, :]  # (L, N)
        relf = rel.astype(mx.float32).reshape(B, KV, 3, L, REL)
        bias = mx.take_along_axis(relf, mx.broadcast_to(mx.clip(dist, 0, REL - 1), (B, KV, 3, L, N)), axis=-1)
        bias = mx.where((dist >= 0) & (dist < REL), bias, 0)
        logits = mx.where(dist >= 0, logits + bias, -mx.inf)
        allm = mx.concatenate([logits, sinks.astype(mx.float32).reshape(B, KV, 3, L, 1)], axis=-1)
        p = mx.softmax(allm, axis=-1)[..., :N]
        ref = (p @ vd[:, :, None]).reshape(B, H, L, D)
        mx.eval(ref)
    err = float(mx.abs(got.astype(mx.float32) - ref).max() / mx.abs(ref).max())
    good = err < 1e-2
    ok &= good
    line = f"N={N:6d} L={L}: rel.err vs FP32 reference {err:.1e} {'ok' if good else 'MISMATCH'}"
    if N >= 4096 and L == 1:
        kz = mx.concatenate([k, mx.zeros((B, KV, 1, D), k.dtype)], 2)
        vz = mx.concatenate([v, mx.zeros((B, KV, 1, D), v.dtype)], 2)
        r = (mx.random.normal((B, 1, H, 16)) * 0.3).astype(mx.bfloat16)
        proj = mx.random.normal((16, REL)).astype(mx.bfloat16)
        sk = sinks.reshape(B, H, 1, 1)

        def bf16_path():
            m = banded_additive_mask(r, proj, N - 1, N, 0, REL, shape_ref=k)
            return mx.fast.scaled_dot_product_attention(q, kz, vz, scale=D**-0.5, mask=mx.concatenate([m, sk], -1))

        t16 = bench(bf16_path)
        t8 = bench(lambda: q8_decode_attention(q, qk, qv, rel, sinks, REL))
        gb16, gb8 = 2 * KV * N * D * 2 / 1e9, 2 * KV * N * (D + 8) / 1e9
        line += (f"   BF16 KV (mask+SDPA) {t16:6.3f} ms ({gb16 / t16 * 1e3:4.0f} GB/s)   "
                 f"8-bit fused {t8:6.3f} ms ({gb8 / t8 * 1e3:4.0f} GB/s)   {t16 / t8:.2f}x")
    print(line)
print("Q8 DECODE ATTENTION OK" if ok else "Q8 DECODE ATTENTION MISMATCH")
