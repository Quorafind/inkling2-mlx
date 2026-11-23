"""Stage-by-stage timing of one decoder-layer visit during prefill (1024-token chunk).

    python tests/profile_prefill_layer.py MODEL_DIR [layer ...]
"""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_vlm.models.inkling.language import banded_additive_mask  # noqa: E402
from mlx_vlm.models.inkling2.language import _combine_routes, _route, l2_clip  # noqa: E402
from mlx_vlm.models.switch_layers import _gather_sort, _scatter_unsort  # noqa: E402

from inkling2_mini import band_attention as band  # noqa: E402
from inkling2_mini.fused import _gated_output  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402
from inkling2_mini.model import _attnres  # noqa: E402

bot = Inkling2Mini(sys.argv[1], speculative=False)
lm = bot.model.language_model
cfg = lm.config
L = 1024
text = bot.tokenizer.encode(open(Path(__file__).resolve().parents[1] / "README.md").read())
ids = mx.array([text[:L]])
emb = lm.model.embed(ids)
mx.eval(emb)


def timed(fn, n=5):
    out = fn()
    mx.eval(out)
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
        mx.eval(out)
    return out, (time.perf_counter() - t) / n * 1e3


for li in [int(a) for a in sys.argv[2:]] or [1, 5]:
    layer = lm.model.layers[li]
    attn = layer.self_attn
    x = emb
    blocks = [emb]
    rows = []

    def rec(name, fn):
        out, ms = timed(fn)
        rows.append((name, ms))
        return out

    qn = rec("attn residual + norm (kernel)", lambda: _attnres([*blocks, x], layer.attention_input.query,
                                                              layer._qkv_weight(), write_mixed=False)[1])
    h = qn[..., 0, :]
    y = rec("fused q/g/r/s/k/v projection", lambda: attn.qkv_all(h))
    nq, ng, nr, ns, nk, nv = attn._sizes
    o4 = nq + ng + nr + ns
    k_raw, v_raw = y[..., o4:o4 + nk], y[..., o4 + nk:o4 + nk + nv]
    if attn.is_sliding:
        k = rec("k short conv", lambda: attn.k_sconv(k_raw, None, None))
        v = rec("v short conv", lambda: attn.v_sconv(v_raw, None, None))
    else:
        k = rec("k dynamic conv", lambda: attn.k_sconv(k_raw, h, None, None))
        v = rec("v dynamic conv", lambda: attn.v_sconv(v_raw, h, None, None))

    def clips():
        q = l2_clip(y[..., :nq].reshape(1, L, attn.n_heads, 128), cfg.qk_l2_clip_threshold, attn.q_scale)
        kk = l2_clip(k.reshape(1, L, attn.n_kv, 128), cfg.qk_l2_clip_threshold)
        vv = mx.fast.rms_norm(v.reshape(1, L, attn.n_kv, 128).astype(mx.float32), None, 1e-6).astype(mx.bfloat16)
        r = l2_clip(y[..., nq + ng:nq + ng + nr].reshape(1, L, attn.n_heads, 16), cfg.rel_l2_clip_threshold, attn.rel_scale)
        return [a.transpose(0, 2, 1, 3) if a.shape[2] != 16 or True else a for a in (q, kk, vv)] + [r]

    q, kk, vv, r = rec("l2 clips, value norm, transposes", clips)
    sinks = y[..., nq + ng + nr:o4].transpose(0, 2, 1)[..., None]
    if attn.is_sliding:
        def mask_fn():
            m = banded_additive_mask(r, attn.rel_proj.astype(mx.bfloat16), 0, L, attn.sliding, attn.rel_extent, shape_ref=kk)
            return mx.concatenate([m, sinks.astype(m.dtype)], axis=-1)
        mask = rec("sliding mask + sink column", mask_fn)
        z = mx.zeros((1, attn.n_kv, 1, 128), mx.bfloat16)
        kz, vz = mx.concatenate([kk, z], 2), mx.concatenate([vv, z], 2)
        out = rec("SDPA with mask", lambda: mx.fast.scaled_dot_product_attention(q, kz, vz, scale=128**-0.5, mask=mask))
    else:
        rel_logits = rec("band table (r @ proj)", lambda: (r @ attn.rel_proj.astype(mx.bfloat16)).transpose(0, 2, 1, 3))
        out = rec("band attention kernel", lambda: band.band_attention(q, kk, vv, rel_logits, sinks[..., 0], attn.rel_extent))
    gate = y[..., nq:nq + ng]
    ao = rec("output gate + o_proj", lambda: attn.o_proj(_gated_output(out.transpose(0, 2, 1, 3).reshape(1, L, -1), gate)))
    res = rec("attn short conv (2304 ch)", lambda: layer.attn_sconv(ao, None, None).astype(mx.float32) + x.astype(mx.float32))
    hid = rec("mlp residual + norm (kernel)", lambda: _attnres([*blocks, res], layer.mlp_input.query,
                                                               layer.post_attention_layernorm.weight, write_mixed=False)[1][..., 0, :])
    moe = layer.mlp
    sw = moe.switch_mlp
    idx_w = rec("router (sigmoid top-8)", lambda: _route(hid, moe.gate_weight, moe.e_score_correction_bias, moe.top_k, moe.route_scale))
    indices, weights = idx_w
    xs_idx = rec("route sort + gather rows", lambda: _gather_sort(mx.expand_dims(hid, (-2, -3)), indices))
    xs, idx, inv = xs_idx
    gu = rec("experts gate_up (gather_qmm)", lambda: sw.gate_up(xs, idx, sorted_indices=True))
    act = rec("softcapped SwiGLU", lambda: sw.activation(gu[..., moe.inter:], gu[..., :moe.inter]))
    dn = rec("experts down (gather_qmm)", lambda: sw.down_proj(act, idx, sorted_indices=True))
    comb = rec("unsort + weighted combine", lambda: _combine_routes(_scatter_unsort(dn, inv, indices.shape).squeeze(-2), weights))
    rec("mlp short conv + residual", lambda: (res + layer.mlp_sconv(comb, None, None).astype(mx.float32)).astype(mx.bfloat16))

    total = sum(ms for _, ms in rows)
    gemm = sum(ms for n, ms in rows if "projection" in n or "gather_qmm" in n or "o_proj" in n)
    print(f"\nlayer {li} ({'sliding' if attn.is_sliding else 'full'}), {L} tokens: {total:.1f} ms per visit "
          f"(GEMMs {gemm:.1f} ms = {gemm / total:.0%}); x48 visits ~ {total * 48 / 1e3:.2f} s per chunk")
    for name, ms in sorted(rows, key=lambda r: -r[1]):
        print(f"  {ms:6.2f} ms  {ms / total:5.1%}  {name}")
