"""One decode token's MoE in three kernels: router -> experts -> sum, short conv, residual.

moe_decode_conv replaces six kernels per layer visit (router cast + FP32 GEMV, top-k, the
two decode_kernels.moe_decode kernels, the separate short conv + residual; about ten on the
MXFP8 layers, which fell back to gather_qmm):

1. router_logits: one simdgroup per expert, BF16 input and BF16 router weight, FP32 sums.
2. gate/up + softcapped SwiGLU. Every simdgroup runs the top-k itself (_TOPK: sigmoid, bias,
   ties to the lower expert, normalized weights), so routing needs no kernel of its own.
3. down + the weighted route sum (in route order) + the layer's 4-tap short conv + the FP32
   residual, writing the next conv state.

route() runs the same router GEMV and top-k code for multi-token passes, so speculative
verification picks exactly the experts single-token decode would.

* The same kernels read NVFP4 (group 16, E4M3 scales, per-expert global scale) and MXFP8
  (group 32, E8M0 scales).
* The dot products follow MLX's qmv order (8 values per lane step, 256 per simdgroup step,
  for NVFP4 and MXFP8 alike) and the route sum is the FMA chain of
  prefill_kernels.combine_routes, so the output matches route + gather_qmm + combine, which
  verification uses, up to one or two BF16 steps on a handful of values per layer (as the
  previous single-token kernel did). Wider loads (16 or 32 values per lane) read about 3 %
  faster, but decode and verification then drift further apart.

Each expert output is rounded to the model dtype before the SwiGLU and before the weighted
route sum (in route order), as in the reference.
"""

import math

import mlx.core as mx

_k = {}

# Router logits: one simdgroup per expert row, BF16 x and BF16 router weight, FP32 sums
# (the reference computes an FP32 GEMV of the same BF16 values; only the summation order
# differs).
_ROUTER = r"""
    uint lane = thread_index_in_simdgroup;
    uint e = threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup;
    uint t = threadgroup_position_in_grid.y;
    const device T* wr = wg + size_t(e) * H;
    const device T* xt = x + size_t(t) * H;
    float acc = 0.0f;
    for (int i = lane * 8; i < H; i += 256) {
        for (int j = 0; j < 8; ++j) acc = fma(float(xt[i + j]), float(wr[i + j]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) logits[size_t(t) * E + e] = acc;
"""

# Routes of T tokens, one simdgroup per token, with the same TOPK code as the up kernel, so
# multi-token passes (speculative verification) route exactly like single-token decode.
_ROUTE = r"""
    uint lane = thread_index_in_simdgroup;
    uint t = threadgroup_position_in_grid.x;
    const device float* logits = all_logits + size_t(t) * E;
    TOPK_SRC
    float tot = 0.0f;
    for (int k = 0; k < KE; ++k) tot += sel_sc[k];
    if (lane < KE) {
        ids[t * KE + lane] = sel_idx[lane];
        wts[t * KE + lane] = sel_sc[lane] / (tot + 1e-20f) * SCALE;
    }
"""

# Top-K of sigmoid(logits) + bias inside a simdgroup (E / 32 experts per lane), as
# decode_kernels.route: ties go to the lower expert, weights are the selected scores
# normalized in selection order, times the route scale.
_TOPK = r"""
    constexpr int EPL = E / 32;
    float sc[EPL], key[EPL];
    for (int i = 0; i < EPL; ++i) {
        int e = int(lane) + 32 * i;
        sc[i] = 1.0f / (1.0f + metal::precise::exp(-logits[e]));
        key[i] = sc[i] + float(bias[e]);
    }
    uint sel_idx[KE];
    float sel_sc[KE];
    for (int k = 0; k < KE; ++k) {
        float bv = -INFINITY; int bi = E;
        for (int i = 0; i < EPL; ++i) {
            int e = int(lane) + 32 * i;
            if (key[i] > bv || (key[i] == bv && e < bi)) { bv = key[i]; bi = e; }
        }
        float m = simd_max(bv);
        int win = simd_min(bv == m ? bi : E);
        float s = 0.0f;
        for (int i = 0; i < EPL; ++i) {
            if (int(lane) + 32 * i == win) { s = sc[i]; key[i] = -INFINITY; }
        }
        sel_sc[k] = simd_sum(s);
        sel_idx[k] = uint(win);
    }
"""

# Gate/up of one route's expert (RPS gate rows + the matching up rows per simdgroup), then
# the softcapped SwiGLU; writes act (KE, INTER). Every simdgroup routes the token itself
# (TOPK), and the first one writes the routes for the down kernel.
_UP = r"""
    uint route = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    TOPK_SRC
    if (threadgroup_position_in_grid.x == 0 && route == 0 && sg == 0) {
        float tot = 0.0f;
        for (int k = 0; k < KE; ++k) tot += sel_sc[k];
        if (lane < KE) {
            ids_out[lane] = sel_idx[lane];
            wts_out[lane] = sel_sc[lane] / (tot + 1e-20f) * SCALE;
        }
    }
    int row0 = (threadgroup_position_in_grid.x * NSG + sg) * RPS;
    uint expert = sel_idx[route];
    constexpr int KB = K * BITS / 8;     // bytes per row
    constexpr int KS = K / GS;           // scales per row
    constexpr int N = 2 * INTER;
    const device uint8_t* wg_ = (const device uint8_t*)w + size_t(expert) * N * KB + size_t(row0) * KB;
    const device uint8_t* wu_ = wg_ + size_t(INTER) * KB;
    const device uint8_t* sg0 = s + size_t(expert) * N * KS + size_t(row0) * KS;
    const device uint8_t* su0 = sg0 + size_t(INTER) * KS;
    float rg[RPS], ru[RPS];
    for (int r = 0; r < RPS; ++r) { rg[r] = 0.0f; ru[r] = 0.0f; }
    // MLX's fp_qmv_impl order (gather_qmm): 32 / BITS values per lane step
    constexpr int VPT = 8, BLOCK = VPT * 32, FULL = (K / BLOCK) * BLOCK, TAIL = K - FULL;  // MLX qmv order
    thread float xt[VPT];
    for (int k = 0; k < FULL + (TAIL > 0 ? BLOCK : 0); k += BLOCK) {
        if (k == FULL && int(lane) >= TAIL / VPT) break;
        load_vector<T, float, VPT>(x + lane * VPT + k, xt);
        const int u = (k + lane * VPT) / VPT;  // this lane's chunk of VPT values (4 bytes)
        for (int r = 0; r < RPS; ++r) {
            rg[r] += qdot<float, VPT, BITS>(wg_ + r * KB + u * VPT * BITS / 8, xt, dequantize_scale<float, GS>(sg0[r * KS + u * VPT / GS]));
            ru[r] += qdot<float, VPT, BITS>(wu_ + r * KB + u * VPT * BITS / 8, xt, dequantize_scale<float, GS>(su0[r * KS + u * VPT / GS]));
        }
    }
    const float gscale = HAS_GS ? g[expert] / (448.0f * 6.0f) : 1.0f;
    for (int r = 0; r < RPS; ++r) {
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

# Down: simdgroup j computes ROWS output rows of route j's expert; the threadgroup sums the
# routes in route order with their weights, rounds to T (the MoE output), then applies the
# layer's 4-tap short convolution with its state, adds the FP32 residual and writes the next
# conv state (prefill_kernels.short_conv for one token).
_DOWN = r"""
    uint lane = thread_index_in_simdgroup, j = simdgroup_index_in_threadgroup;
    int row0 = threadgroup_position_in_grid.x * ROWS;
    uint expert = ids[j];
    constexpr int KB = K * BITS / 8;
    constexpr int KS = K / GS;
    const device uint8_t* w0 = (const device uint8_t*)w + size_t(expert) * N * KB + size_t(row0) * KB;
    const device uint8_t* s0 = s + size_t(expert) * N * KS + size_t(row0) * KS;
    const device T* xr = act + j * K;
    float res[ROWS];
    for (int r = 0; r < ROWS; ++r) res[r] = 0.0f;
    constexpr int VPT = 8, BLOCK = VPT * 32, FULL = (K / BLOCK) * BLOCK, TAIL = K - FULL;  // MLX qmv order
    thread float xt[VPT];
    for (int k = 0; k < FULL + (TAIL > 0 ? BLOCK : 0); k += BLOCK) {
        if (k == FULL && int(lane) >= TAIL / VPT) break;
        load_vector<T, float, VPT>(xr + lane * VPT + k, xt);
        const int u = (k + lane * VPT) / VPT;
        for (int r = 0; r < ROWS; ++r)
            res[r] += qdot<float, VPT, BITS>(w0 + r * KB + u * VPT * BITS / 8, xt, dequantize_scale<float, GS>(s0[r * KS + u * VPT / GS]));
    }
    const float gscale = HAS_GS ? g[expert] / (448.0f * 6.0f) : 1.0f;
    threadgroup float part[KE][ROWS];
    for (int r = 0; r < ROWS; ++r) {
        float acc = simd_sum(res[r]);
        if (lane == 0) part[j][r] = float(T(acc * gscale));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0 && lane < ROWS) {
        const int d = row0 + lane;
        float acc = 0.0f;
        for (int k = 0; k < KE; ++k) acc = fma(float(wts[k]), part[k][lane], acc);  // as combine_routes (verify)
        const float x0 = float(T(acc));
        const float xm3 = state[d], xm2 = state[N + d], xm1 = state[2 * N + d];
        const float c = float(cw[d * 4 + 0]) * xm3 + float(cw[d * 4 + 1]) * xm2
                      + float(cw[d * 4 + 2]) * xm1 + float(cw[d * 4 + 3]) * x0;
        const float r = float(T((c + x0) / CONV_SCALE));
        out[d] = O(r + float(resid[d]));
        nstate[d] = xm2; nstate[N + d] = xm1; nstate[2 * N + d] = x0;
    }
"""


def expert_format(linear):
    """(bits, has_global_scale) for a quantized SwitchLinear the kernels read, else None."""
    mode = getattr(linear, "mode", None)
    bits = getattr(linear, "bits", None)
    gs = getattr(linear, "group_size", None)
    if mode == "nvfp4" and bits == 4 and gs == 16:
        return 4, True
    if mode == "mxfp8" and bits == 8 and gs == 32:
        return 8, False
    return None


def supports(sw):
    a, b = expert_format(sw.gate_up), expert_format(sw.down_proj)
    return a is not None and a == b


def _global_scale(linear):
    """mlx-vlm's per-expert NVFP4 global scale, computed once per layer and kept (mlx-vlm
    recomputes it, one small kernel, on every call)."""
    gs = getattr(linear, "_mk_global_scale", None)
    if gs is None:
        from mlx_vlm.models.masked_quantized import _global_scale as gsc

        gs = gsc(linear)
        mx.eval(gs)
        object.__setattr__(linear, "_mk_global_scale", gs)
    return gs


# tuned on an M6 (tests/bench_moe_decode.py); differences between the good configs are ~2 %
CONFIG = dict(RPS=1, NSG=2, ROWS=2)
ROUTER_NSG = 4


def _mlx_header():
    from mlx_vlm.models.masked_quantized import _HEADER as h

    return h


def _kernels(gate_cap, up_cap, route_scale, top_k, n_experts):
    key = ("moe", gate_cap, up_cap, route_scale, top_k, n_experts)
    if key not in _k:
        up = (_UP.replace("TOPK_SRC", _TOPK).replace("GCAP", f"{float(gate_cap):.9e}f")
              .replace("UCAP", f"{float(up_cap):.9e}f").replace("SCALE", f"{float(route_scale):.9e}f"))
        _k[key] = (
            mx.fast.metal_kernel(name="inkling_moe_up_routed", input_names=["x", "logits", "bias", "w", "s", "g"],
                                 output_names=["act", "ids_out", "wts_out"], header=_mlx_header(), source=up,
                                 compile_options={"math_mode": "safe"}),
            mx.fast.metal_kernel(name="inkling_moe_down_conv",
                                 input_names=["act", "w", "s", "g", "ids", "wts", "state", "cw", "resid"],
                                 output_names=["out", "nstate"], header=_mlx_header(),
                                 source=_DOWN.replace("CONV_SCALE", f"{math.sqrt(5):.9e}f"),
                                 compile_options={"math_mode": "safe"}),
        )
    return _k[key]


def moe_decode_conv(hidden, moe, conv_state, conv_weight, residual, out_dtype, cfg=None):
    """One decode token through route -> experts -> weighted sum -> 4-tap short conv ->
    + residual, in three kernels.

    hidden (1, 1, H) BF16: the MoE input; moe: GroupedSparseMoE; conv_state (1, 3, H) FP32
    (zeros at sequence start); conv_weight (H, 4, 1); residual (1, 1, H) FP32.
    Returns (out (1, 1, H) out_dtype, next conv state (1, 3, H) FP32)."""
    H = hidden.shape[-1]
    sw = moe.switch_mlp
    gu, dn = sw.gate_up, sw.down_proj
    bits, has_gs = expert_format(gu)
    c = dict(CONFIG, **(cfg or {}))
    E, KE, inter = moe.gate_weight.shape[0], moe.top_k, moe.inter
    up, down = _kernels(sw.activation.gate_cap, sw.activation.up_cap, moe.route_scale, KE, E)
    x = hidden.reshape(H)
    logits = router_logits(x.reshape(1, H), moe.gate_weight)
    one = mx.ones((1,), mx.float32)
    g_up = _global_scale(gu) if has_gs else one
    g_dn = _global_scale(dn) if has_gs else one
    rows_per_tg = c["RPS"] * c["NSG"]
    act, ids, wts = up(
        inputs=[x, logits, moe.e_score_correction_bias, gu.weight, gu.scales, g_up],
        template=[("T", x.dtype), ("K", H), ("INTER", inter), ("BITS", bits), ("HAS_GS", has_gs), ("GS", 16 if bits == 4 else 32),
                  ("RPS", c["RPS"]), ("NSG", c["NSG"]), ("E", E), ("KE", KE)],
        grid=(inter // rows_per_tg * 32 * c["NSG"], KE, 1), threadgroup=(32 * c["NSG"], 1, 1),
        output_shapes=[(KE, inter), (KE,), (KE,)], output_dtypes=[x.dtype, mx.uint32, mx.float32],
    )
    out, nstate = down(
        inputs=[act, dn.weight, dn.scales, g_dn, ids, wts, conv_state.astype(mx.float32).reshape(3 * H),
                conv_weight.reshape(H * 4), residual.reshape(H)],
        template=[("T", x.dtype), ("O", out_dtype), ("K", inter), ("N", H), ("KE", KE), ("BITS", bits),
                  ("HAS_GS", has_gs), ("ROWS", c["ROWS"]), ("GS", 16 if bits == 4 else 32)],
        grid=(H // c["ROWS"] * 32, KE, 1), threadgroup=(32, KE, 1),
        output_shapes=[(H,), (3 * H,)], output_dtypes=[out_dtype, mx.float32],
    )
    return out.reshape(1, 1, H), nstate.reshape(1, 3, H)


ENABLED = True  # A/B switch: False restores decode_kernels.moe_decode + the separate conv


def router_logits(x, gate_weight):
    """x (T, H) and gate_weight (E, H), both BF16: FP32 logits (T, E), each a BF16 x BF16
    dot product summed in FP32 in a fixed order (the same for every T)."""
    T_, H = x.shape
    E = gate_weight.shape[0]
    if "router" not in _k:
        _k["router"] = mx.fast.metal_kernel(name="inkling_moe_router", input_names=["x", "wg"],
                                            output_names=["logits"], source=_ROUTER,
                                            compile_options={"math_mode": "safe"})
    return _k["router"](inputs=[x, gate_weight], template=[("T", x.dtype), ("H", H), ("E", E), ("NSG", ROUTER_NSG)],
                        grid=(32 * E, T_, 1), threadgroup=(32 * ROUTER_NSG, 1, 1),
                        output_shapes=[(T_, E)], output_dtypes=[mx.float32])[0]


def route(x, gate_weight, bias, top_k, route_scale):
    """x (..., H) BF16. Router logits, then sigmoid / + bias / top-k / normalization exactly
    as the fused decode kernels do it. Returns (indices (..., K) uint32, weights (..., K) FP32)."""
    *lead, H = x.shape
    T_ = x.size // H
    E = gate_weight.shape[0]
    logits = router_logits(x.reshape(T_, H), gate_weight)
    key = ("route", float(route_scale), top_k, E)
    if key not in _k:
        _k[key] = mx.fast.metal_kernel(
            name="inkling_moe_route", input_names=["all_logits", "bias"], output_names=["ids", "wts"],
            source=_ROUTE.replace("TOPK_SRC", _TOPK).replace("SCALE", f"{float(route_scale):.9e}f"),
            compile_options={"math_mode": "safe"},
        )
    ids, wts = _k[key](inputs=[logits, bias], template=[("E", E), ("KE", top_k)],
                       grid=(32 * T_, 1, 1), threadgroup=(32, 1, 1),
                       output_shapes=[(T_, top_k), (T_, top_k)], output_dtypes=[mx.uint32, mx.float32])
    return ids.reshape(*lead, top_k), wts.reshape(*lead, top_k)


def route_ok(moe, x):
    return (ENABLED and x.dtype == mx.bfloat16 and moe.gate_weight.dtype == x.dtype
            and moe.gate_weight.shape[0] % 32 == 0 and x.shape[-1] % 256 == 0)


def fused_ok(moe, hidden):
    """Single decode token, kernels on, a format the kernels read, E a multiple of 32."""
    from . import decode_kernels

    return (
        ENABLED
        and hidden.size == hidden.shape[-1]
        and hidden.dtype == mx.bfloat16
        and decode_kernels.active()
        and supports(moe.switch_mlp)
        and moe.gate_weight.shape[0] % 32 == 0
        and moe.gate_weight.dtype == hidden.dtype
        and hidden.shape[-1] % 256 == 0
        and moe.inter % 64 == 0
    )
