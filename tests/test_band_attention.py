"""Band-bias flash attention kernel vs the materialized-mask path (mlx-vlm's banded
mask + concatenated zero sink + fused SDPA), with Inkling-like statistics, plus timing.

    python tests/test_band_attention.py
"""
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_vlm.models.inkling.language import banded_additive_mask  # noqa: E402
from inkling2_mini.band_attention import band_attention  # noqa: E402

B, H, KV, D, DREL, REL = 1, 24, 8, 128, 16, 1024


def l2clip(x, t):
    n = mx.sqrt((x.astype(mx.float32) ** 2).sum(-1, keepdims=True))
    return (x * mx.minimum(t / (n + 1e-6), 1)).astype(mx.bfloat16)


def reference(q, k, v, r, proj, sinks, kL, qL, sliding=0):
    mask = banded_additive_mask(r, proj, kL - qL, kL, sliding, REL, shape_ref=k)
    mask = mx.concatenate([mask, sinks[..., None]], axis=-1)
    z = mx.zeros((B, KV, 1, D), k.dtype)
    return mx.fast.scaled_dot_product_attention(q, mx.concatenate([k, z], 2), mx.concatenate([v, z], 2),
                                                scale=D**-0.5, mask=mask)


def bench(f, n=5):
    mx.eval(f())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - t) / n * 1e3


ok = True
for qL, kL, sl in [(64, 64, 0), (100, 100, 0), (64, 1500, 0), (77, 3000, 0), (512, 512, 0), (512, 4096, 0),
                   (1024, 16384, 0), (1024, 65536, 0),
                   (300, 300, 1024), (1024, 1024, 1024), (1024, 2047, 1024), (77, 1100, 1024), (2048, 3071, 1024)]:
    mx.random.seed(qL + kL)
    q = l2clip(mx.random.normal((B, H, qL, D)) * 3, 22.6)
    k = l2clip(mx.random.normal((B, KV, kL, D)) * 3, 22.6)
    v = mx.random.normal((B, KV, kL, D)).astype(mx.bfloat16)
    r = (mx.random.normal((B, qL, H, DREL)) * 0.8).astype(mx.bfloat16)
    proj = mx.random.normal((DREL, REL)).astype(mx.bfloat16)
    sinks = (mx.random.normal((B, H, qL)) * 2).astype(mx.bfloat16)
    rel_logits = (r @ proj).transpose(0, 2, 1, 3)  # (B, H, qL, REL)
    ref = reference(q, k, v, r, proj, sinks, kL, qL, sl)
    got = band_attention(q, k, v, rel_logits, sinks, REL, sl)
    err = float(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max() / mx.abs(ref.astype(mx.float32)).max())
    good = err < 2e-2
    ok &= good
    line = f"qL={qL:5d} kL={kL:6d} window={sl or '-':>4}: rel.err {err:.1e} {'ok' if good else 'MISMATCH'}"
    if kL >= 4096 or (sl and qL >= 1024):
        t_ref = bench(lambda: reference(q, k, v, r, proj, sinks, kL, qL, sl))
        t_new = bench(lambda: band_attention(q, k, v, (r @ proj).transpose(0, 2, 1, 3), sinks, REL, sl))
        line += f"   mask+sdpa {t_ref:7.1f} ms   band kernel {t_new:7.1f} ms   ({t_ref / t_new:.1f}x)"
    print(line)
print("BAND ATTENTION OK" if ok else "BAND ATTENTION MISMATCH")
