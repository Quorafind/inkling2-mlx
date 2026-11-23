"""Fused Metal kernels for Inkling 2 Mini's decode glue.

A decode step is a serial chain of about 20 dependent kernels per layer visit, 48 visits
per token, and each dependent dispatch costs about 3.5 us on an M6. The weights-reading
kernels already run near the memory-bandwidth floor, so what is left to win is the
number of kernels on that chain. Each kernel here replaces a chain of MLX ops:

* route: sigmoid, + correction bias, top-k selection and weight normalization after the
  FP32 router GEMV (mlx-vlm's compiled _route).
* attn_prep: q / k / relative-state L2 clips (with learned scales), value RMSNorm and the
  full layers' log position scaling, writing q / k / v already in attention layout. Also
  used by prefill chunks (B = 1).
* moe_decode: one token through its routed NVFP4 experts in two kernels (gate_up +
  softcapped SwiGLU, then down + the weighted sum of the routes), replacing gather_qmm,
  the activation, gather_qmm and the combine. Bit-identical.

The sliding-window ring attention kernel lives in decode_attention.py, and the
single-token short convolutions (with the next conv state) in prefill_kernels.py. All of
them switch off with ENABLED = False (A/B tests).
"""

import mlx.core as mx

_k = {}


def use_kernels():
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# ------------------------------------------------------------------ router
_ROUTE = r"""
    // One threadgroup (E threads) per token: sigmoid, + bias, top-K, normalize.
    uint t = threadgroup_position_in_grid.x;
    uint e = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    constexpr int NSG = (E + 31) / 32;  // the last simdgroup may be partial (E < 32: tiny models)
    float score = 1.0f / (1.0f + metal::precise::exp(-logits[int64_t(t) * E + e]));
    float key = score + float(bias[e]);
    threadgroup float red_v[NSG];
    threadgroup int red_i[NSG];
    threadgroup float sel_score[K];
    threadgroup int sel_idx[K];
    bool taken = false;
    for (int k = 0; k < K; ++k) {
        float v = taken ? -INFINITY : key;
        float m = simd_max(v);
        int cand = (v == m) ? int(e) : E;
        int mi = simd_min(cand);
        if (lane == 0) { red_v[sg] = m; red_i[sg] = mi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bm = red_v[0]; int bi = red_i[0];
        for (int s = 1; s < NSG; ++s) {
            if (red_v[s] > bm || (red_v[s] == bm && red_i[s] < bi)) { bm = red_v[s]; bi = red_i[s]; }
        }
        if (int(e) == bi) { taken = true; sel_idx[k] = bi; sel_score[k] = score; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (e < K) {
        float tot = 0;
        for (int k = 0; k < K; ++k) tot += sel_score[k];
        indices[t * K + e] = uint32_t(sel_idx[e]);
        weights[t * K + e] = sel_score[e] / (tot + 1e-20f) * SCALE;
    }
"""


def route(x, gate_weight, bias, top_k, route_scale, gate_weight_t32=None):
    """x: (B, L, H); gate_weight (E, H). Logits are an FP32 GEMV of the BF16 inputs (as
    the reference); `gate_weight_t32` is an optional cached FP32 transposed copy.
    Returns (indices (B, L, K) uint32, weights (B, L, K) FP32)."""
    *lead, H = x.shape
    T = x.size // H
    E = gate_weight.shape[0]
    wt = gate_weight_t32 if gate_weight_t32 is not None else gate_weight.astype(mx.float32).T
    logits = x.reshape(T, H).astype(mx.float32) @ wt
    key = ("route", float(route_scale))
    if key not in _k:
        _k[key] = mx.fast.metal_kernel(
            name="inkling_decode_route", input_names=["logits", "bias"],
            output_names=["indices", "weights"],
            source=_ROUTE.replace("SCALE", f"{float(route_scale):.9e}f"),
        )
    idx, wts = _k[key](
        inputs=[logits, bias.astype(mx.float32)],
        template=[("E", E), ("K", top_k)],
        grid=(T * E, 1, 1), threadgroup=(E, 1, 1),
        output_shapes=[(T, top_k), (T, top_k)], output_dtypes=[mx.uint32, mx.float32],
    )
    return idx.reshape(*lead, top_k), wts.reshape(*lead, top_k)


ENABLED = True  # master switch for the decode fusions (A/B tests)


def active():
    return ENABLED and use_kernels()


# ------------------------------------------------------------ attention prep
_PREP = r"""
    // One simdgroup per (token, row); rows: [q heads | k heads | v heads | r heads].
    // q, r: L2 clip after the learned scale, then (full layers) the log position scale;
    // k: L2 clip; v: RMSNorm without weight. Writes q / k / v as (B, heads, L, D).
    const int t = threadgroup_position_in_grid.x;   // token (B = 1)
    const int row = threadgroup_position_in_grid.y;
    const int lane = thread_index_in_simdgroup;
    const int L = meta[0];
    const int offset = meta[1];
    float tau = 1.0f;
    if (LOGSCALE) {
        float pos = float(offset + t + 1);
        tau = 1.0f + ALPHA * metal::precise::log(metal::max(pos / NFLOOR, 1.0f));
    }
    if (row < H + 2 * KV) {
        const bool is_q = row < H, is_k = !is_q && row < H + KV;
        const device T* src = is_q ? y + int64_t(t) * y_shape[2] + row * 128
                            : is_k ? k + (int64_t(t) * KV + (row - H)) * 128
                                   : v + (int64_t(t) * KV + (row - H - KV)) * 128;
        float xv[4], ss = 0;
        for (int i = 0; i < 4; ++i) {
            float a = float(src[lane * 4 + i]);
            if (is_q) a *= float(q_scale[row * 128 + lane * 4 + i]);
            xv[i] = a; ss += a * a;
        }
        ss = simd_sum(ss);
        device T* dst;
        if (is_q) dst = q_out + (int64_t(row) * L + t) * 128;
        else if (is_k) dst = k_out + (int64_t(row - H) * L + t) * 128;
        else dst = v_out + (int64_t(row - H - KV) * L + t) * 128;
        if (!is_q && !is_k) {  // value: RMSNorm
            float r = metal::rsqrt(ss / 128.0f + EPS);
            for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = static_cast<T>(xv[i] * r);
        } else {
            float f = metal::min(QK_CLIP / (metal::sqrt(ss) + 1e-6f), 1.0f);
            for (int i = 0; i < 4; ++i) {
                float c = float(static_cast<T>(xv[i] * f));     // l2_clip rounds to T
                dst[lane * 4 + i] = static_cast<T>(is_q && LOGSCALE ? c * tau : c);
            }
        }
    } else {  // relative state: D_REL values, lanes < D_REL
        const int h = row - H - 2 * KV;
        float a = 0;
        if (lane < D_REL) a = float(y[int64_t(t) * y_shape[2] + R_OFF + h * D_REL + lane]) * float(rel_scale[h * D_REL + lane]);
        float ss = simd_sum(a * a);
        float f = metal::min(REL_CLIP / (metal::sqrt(ss) + 1e-6f), 1.0f);
        if (lane < D_REL) {
            float c = float(static_cast<T>(a * f));
            r_out[(int64_t(t) * H + h) * D_REL + lane] = static_cast<T>(LOGSCALE ? c * tau : c);
        }
    }
"""


def attn_prep(y, k, v, q_scale, rel_scale, r_off, n_heads, n_kv, d_rel, qk_clip, rel_clip, eps,
              offset=0, log_scaling=None):
    """y: (1, L, N) fused projection output (q at columns [0, H*128), relative states at
    [r_off, r_off + H*d_rel)); k, v: (1, L, KV*128) after their convolutions.
    log_scaling: None or (alpha, n_floor). Returns q (1, H, L, 128), k, v (1, KV, L, 128),
    r (1, L, H, d_rel)."""
    _, L, _ = y.shape
    key = ("prep", n_heads, n_kv, d_rel, float(qk_clip), float(rel_clip), float(eps), log_scaling, r_off)
    if key not in _k:
        alpha, nfloor = log_scaling or (0.0, 1.0)
        src = (_PREP.replace("QK_CLIP", f"{qk_clip:.9e}f").replace("REL_CLIP", f"{rel_clip:.9e}f")
               .replace("EPS", f"{eps:.9e}f").replace("ALPHA", f"{alpha:.9e}f")
               .replace("NFLOOR", f"{float(nfloor):.9e}f").replace("R_OFF", str(r_off)))
        _k[key] = mx.fast.metal_kernel(
            name="inkling_attn_prep", input_names=["y", "k", "v", "q_scale", "rel_scale", "meta"],
            output_names=["q_out", "k_out", "v_out", "r_out"], source=src,
        )
    meta = mx.array([L, int(offset)], mx.int32)
    rows = n_heads + 2 * n_kv + n_heads
    return _k[key](
        inputs=[y, k, v, q_scale, rel_scale, meta],
        template=[("T", y.dtype), ("H", n_heads), ("KV", n_kv), ("D_REL", d_rel),
                  ("LOGSCALE", log_scaling is not None)],
        grid=(L * 32, rows, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, n_heads, L, 128), (1, n_kv, L, 128), (1, n_kv, L, 128), (1, L, n_heads, d_rel)],
        output_dtypes=[y.dtype] * 4,
    )


# ------------------------------------------------- single-token NVFP4 experts
_MOE_UP = r"""
    // One token, R routes. Simdgroup: 4 gate rows + the matching 4 up rows of one route's
    // expert, then the softcapped SwiGLU; writes act (R, INTER).
    uint route = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    int row0 = threadgroup_position_in_grid.y * 8 + sg * 4;
    int expert = int(ids[route]);
    constexpr int N = 2 * INTER;
    constexpr int P = 8;
    constexpr int BLOCK = P * 32;
    constexpr int FULL = (K / BLOCK) * BLOCK;
    constexpr int TAIL = K - FULL;
    const float gscale = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
    const device uint8_t* wg = (const device uint8_t*)w + size_t(expert) * N * (K / 2) + row0 * (K / 2) + lane * (P / 2);
    const device uint8_t* wu = wg + size_t(INTER) * (K / 2);
    const device uint8_t* sgp = s + size_t(expert) * N * (K / 16) + row0 * (K / 16) + lane / (16 / P);
    const device uint8_t* sup = sgp + size_t(INTER) * (K / 16);
    thread float xt[P];
    thread float rg[4] = {0}, ru[4] = {0};
    for (int k = 0; k < FULL; k += BLOCK) {
        load_vector<T, float, P>(x + lane * P + k, xt);
        for (int r = 0; r < 4; ++r) {
            rg[r] += qdot<float, P, 4>(wg + r * (K / 2), xt, dequantize_scale<float, 16>(sgp[r * (K / 16)]));
            ru[r] += qdot<float, P, 4>(wu + r * (K / 2), xt, dequantize_scale<float, 16>(sup[r * (K / 16)]));
        }
        wg += BLOCK / 2; wu += BLOCK / 2; sgp += BLOCK / 16; sup += BLOCK / 16;
    }
    if (TAIL > 0 && int(lane) < TAIL / P) {
        load_vector<T, float, P>(x + lane * P + FULL, xt);
        for (int r = 0; r < 4; ++r) {
            rg[r] += qdot<float, P, 4>(wg + r * (K / 2), xt, dequantize_scale<float, 16>(sgp[r * (K / 16)]));
            ru[r] += qdot<float, P, 4>(wu + r * (K / 2), xt, dequantize_scale<float, 16>(sup[r * (K / 16)]));
        }
    }
    for (int r = 0; r < 4; ++r) {
        float gs = simd_sum(rg[r]), us = simd_sum(ru[r]);
        if (lane == 0) {
            float gate = float(T(gs * gscale)), up = float(T(us * gscale));
            float sig = 1.0f / (1.0f + metal::exp(-metal::abs(gate)));
            sig = gate < 0 ? 1.0f - sig : sig;
            float a = metal::precise::tanh(gate / GCAP) * GCAP * sig * metal::precise::tanh(up / UCAP) * UCAP;
            act[route * INTER + row0 + r] = T(a);
        }
    }
"""

_MOE_DOWN = r"""
    // One token: simdgroup r computes 4 output rows of route r's expert; the threadgroup
    // then sums the R routes in route order with their weights (as combine_routes).
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    int row0 = threadgroup_position_in_grid.y * 4;
    int expert = int(ids[sg]);
    constexpr int P = 8;
    constexpr int BLOCK = P * 32;
    constexpr int FULL = (K / BLOCK) * BLOCK;
    constexpr int TAIL = K - FULL;
    const float gscale = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
    const device uint8_t* ws = (const device uint8_t*)w + size_t(expert) * N * (K / 2) + row0 * (K / 2) + lane * (P / 2);
    const device uint8_t* scales = s + size_t(expert) * N * (K / 16) + row0 * (K / 16) + lane / (16 / P);
    const device T* xr = act + sg * K;
    thread float xt[P];
    thread float res[4] = {0};
    for (int k = 0; k < FULL; k += BLOCK) {
        load_vector<T, float, P>(xr + lane * P + k, xt);
        for (int r = 0; r < 4; ++r)
            res[r] += qdot<float, P, 4>(ws + r * (K / 2), xt, dequantize_scale<float, 16>(scales[r * (K / 16)]));
        ws += BLOCK / 2; scales += BLOCK / 16;
    }
    if (TAIL > 0 && int(lane) < TAIL / P) {
        load_vector<T, float, P>(xr + lane * P + FULL, xt);
        for (int r = 0; r < 4; ++r)
            res[r] += qdot<float, P, 4>(ws + r * (K / 2), xt, dequantize_scale<float, 16>(scales[r * (K / 16)]));
    }
    threadgroup float part[R][4];
    for (int r = 0; r < 4; ++r) {
        float acc = simd_sum(res[r]);
        if (lane == 0) part[sg][r] = float(T(acc * gscale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < 4) {
        float acc = 0.0f;
        for (int k = 0; k < R; ++k) acc += float(wts[k]) * part[k][lane];
        out[row0 + lane] = T(acc);
    }
"""


def supports_moe_decode(sw):
    from . import kernels

    return kernels.supports_grouped_nvfp4(sw.gate_up) and kernels.supports_grouped_nvfp4(sw.down_proj)


def moe_decode(x, sw, indices, weights, inter, gate_cap, up_cap):
    """One token through its R routed NVFP4 experts: x (..., H) with one token;
    indices (R,) uint32 in route order; weights (R,) FP32. Returns (..., H) in x.dtype."""
    from mlx_vlm.models.masked_quantized import _HEADER, _global_scale

    H = x.shape[-1]
    R = indices.size
    if "moe_up" not in _k:
        src = _MOE_UP.replace("GCAP", f"{float(gate_cap):.9e}f").replace("UCAP", f"{float(up_cap):.9e}f")
        _k["moe_up"] = mx.fast.metal_kernel(
            name="inkling_moe_decode_up", input_names=["x", "w", "s", "g", "ids"],
            output_names=["act"], header=_HEADER, source=src, compile_options={"math_mode": "safe"},
        )
        _k["moe_down"] = mx.fast.metal_kernel(
            name="inkling_moe_decode_down", input_names=["act", "w", "s", "g", "ids", "wts"],
            output_names=["out"], header=_HEADER, source=_MOE_DOWN, compile_options={"math_mode": "safe"},
        )
    ids = indices.reshape(R)
    gu, dn = sw.gate_up, sw.down_proj
    act = _k["moe_up"](
        inputs=[x.reshape(H), gu.weight, gu.scales, _global_scale(gu), ids],
        template=[("T", x.dtype), ("K", H), ("INTER", inter)],
        grid=(32, inter // 8 * 2, R), threadgroup=(32, 2, 1),
        output_shapes=[(R, inter)], output_dtypes=[x.dtype],
    )[0]
    out = _k["moe_down"](
        inputs=[act, dn.weight, dn.scales, _global_scale(dn), ids, weights.reshape(R).astype(mx.float32)],
        template=[("T", x.dtype), ("K", inter), ("N", H), ("R", R)],
        grid=(32, H // 4 * R, 1), threadgroup=(32, R, 1),
        output_shapes=[(H,)], output_dtypes=[x.dtype],
    )[0]
    return out.reshape(x.shape)
