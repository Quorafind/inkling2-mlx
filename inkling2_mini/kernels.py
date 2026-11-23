"""Fused Metal kernels for the per-visit glue of Inkling 2 Mini.

Single-token decode runs ~2,000 small dependent kernels per token (48 layer
visits), so kernel latency, not bandwidth, is a large share of decode time. The
block attention residual is the biggest single source: stack -> cast -> RMSNorm ->
scores -> softmax -> weighted sum, then usually an RMSNorm on the result. Here it
is one kernel per call, reading the block states in place (no stack copy).
"""

import math

import mlx.core as mx

_TG = 256  # threads per token
_kernels = {}


def _source(S, Q, H, norm, write_mixed, w_rows, eps, norm_eps):
    per = math.ceil(H / _TG)
    load = []
    for s in range(S):
        load.append(
            f"""
    {{
        float acc = 0.0f;
        for (int i = 0; i < K_PER; ++i) {{
            int h = tid + i * K_TG;
            float v = h < K_H ? float(s{s}[tok * K_H + h]) : 0.0f;
            xs[{s}][i] = v;
            acc += v * v;
        }}
        part[{s}] = acc;
    }}"""
        )
    load = "\n".join(load)
    norm_code = ""
    if norm:
        norm_code = f"""
    // RMSNorm of each output row with its weight row, written as T_OUT.
    for (int q = 0; q < K_Q; ++q) {{
        float acc = 0.0f;
        for (int i = 0; i < K_PER; ++i) acc += outv[q][i] * outv[q][i];
        part[q] = acc;
    }}
    attnres_reduce(part, K_Q, red, tot, tid, lane, sg);
    for (int q = 0; q < K_Q; ++q) {{
        float r = metal::rsqrt(tot[q] / float(K_H) + {norm_eps:.9e}f);
        int wrow = {"q" if w_rows > 1 else "0"};
        for (int i = 0; i < K_PER; ++i) {{
            int h = tid + i * K_TG;
            if (h < K_H) normed[(tok * K_Q + q) * K_H + h] = static_cast<T_OUT>(outv[q][i] * r * float(w[wrow * K_H + h]));
        }}
    }}"""
    mixed_code = ""
    if write_mixed:
        mixed_code = """
    for (int q = 0; q < K_Q; ++q)
        for (int i = 0; i < K_PER; ++i) {
            int h = tid + i * K_TG;
            if (h < K_H) mixed[(tok * K_Q + q) * K_H + h] = outv[q][i];
        }"""
    header = f"""
constant constexpr int K_TG = {_TG};
constant constexpr int K_H = {H};
constant constexpr int K_S = {S};
constant constexpr int K_Q = {Q};
constant constexpr int K_PER = {per};
constant constexpr int K_NSG = K_TG / 32;
constant constexpr int K_MAXR = {max(S * Q, S, Q)};

inline void attnres_reduce(thread float* part, int n, threadgroup float (*red)[K_MAXR],
                   threadgroup float* tot, uint tid, uint lane, uint sg) {{
    for (int j = 0; j < n; ++j) {{
        float v = metal::simd_sum(part[j]);
        if (lane == 0) red[sg][j] = v;
    }}
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (tid < uint(n)) {{
        float acc = 0.0f;
        for (int g = 0; g < K_NSG; ++g) acc += red[g][tid];
        tot[tid] = acc;
    }}
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}}
"""
    body = f"""
    uint tok = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float red[K_NSG][K_MAXR];
    threadgroup float tot[K_MAXR];
    float xs[K_S][K_PER];
    float part[K_MAXR];

    // 1. RMS of every state (keys = values = normalized states).
{load}
    attnres_reduce(part, K_S, red, tot, tid, lane, sg);
    float rms[K_S];
    for (int s = 0; s < K_S; ++s) rms[s] = metal::rsqrt(tot[s] / float(K_H) + {eps:.9e}f);
    for (int s = 0; s < K_S; ++s)
        for (int i = 0; i < K_PER; ++i) xs[s][i] *= rms[s];

    // 2. Scores of every (query, state) pair.
    for (int q = 0; q < K_Q; ++q)
        for (int s = 0; s < K_S; ++s) {{
            float acc = 0.0f;
            for (int i = 0; i < K_PER; ++i) {{
                int h = tid + i * K_TG;
                acc += h < K_H ? xs[s][i] * query[q * K_H + h] : 0.0f;
            }}
            part[q * K_S + s] = acc;
        }}
    attnres_reduce(part, K_Q * K_S, red, tot, tid, lane, sg);

    // 3. Softmax over states and the weighted sum.
    float outv[K_Q][K_PER];
    const float scale = metal::rsqrt(float(K_H));
    for (int q = 0; q < K_Q; ++q) {{
        float m = -INFINITY;
        for (int s = 0; s < K_S; ++s) m = metal::max(m, tot[q * K_S + s] * scale);
        float wsum = 0.0f, wts[K_S];
        for (int s = 0; s < K_S; ++s) {{ wts[s] = metal::exp(tot[q * K_S + s] * scale - m); wsum += wts[s]; }}
        for (int i = 0; i < K_PER; ++i) {{
            float acc = 0.0f;
            for (int s = 0; s < K_S; ++s) acc += wts[s] * xs[s][i];
            outv[q][i] = acc / wsum;
        }}
    }}
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
{mixed_code}
{norm_code}
"""
    return header, body


def _kernel(S, Q, H, norm, write_mixed, w_rows, eps, norm_eps):
    key = (S, Q, H, norm, write_mixed, w_rows, eps, norm_eps)
    if key not in _kernels:
        header, body = _source(*key)
        inputs = [f"s{i}" for i in range(S)] + ["query"] + (["w"] if norm else [])
        outputs = (["mixed"] if write_mixed else []) + (["normed"] if norm else [])
        _kernels[key] = mx.fast.metal_kernel(
            name=f"inkling_attnres_s{S}_q{Q}_n{int(norm)}_m{int(write_mixed)}_w{w_rows}",
            input_names=inputs,
            output_names=outputs,
            header=header,
            source=body,
        )
    return _kernels[key]


def use_kernels():
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def attention_residual(states, query, norm_weight=None, write_mixed=True, eps=1e-6,
                       norm_eps=1e-6, out_dtype=mx.bfloat16):
    """Block attention residual over `states` (list of (B, L, H) arrays, the last
    one being the partial block) with `query` (H,) or (Q, H), values = keys =
    RMS-normalized states. Returns (mixed FP32 (B, L, Q, H) or None,
    normed out_dtype (B, L, Q, H) or None); `normed` = RMSNorm(mixed) * weight,
    with weight (H,) or one row per query (Q, H)."""
    B, L, H = states[-1].shape
    query = query.reshape(-1, H).astype(mx.float32)
    Q = query.shape[0]
    norm = norm_weight is not None
    w_rows = 1
    if norm:
        norm_weight = norm_weight.reshape(-1, H)
        w_rows = norm_weight.shape[0]
    kernel = _kernel(len(states), Q, H, norm, write_mixed, w_rows, eps, norm_eps)
    inputs = [s if s.shape == (B, L, H) else mx.broadcast_to(s, (B, L, H)) for s in states]
    inputs = inputs + [query] + ([norm_weight] if norm else [])
    shapes, dtypes = [], []
    if write_mixed:
        shapes.append((B, L, Q, H))
        dtypes.append(mx.float32)
    if norm:
        shapes.append((B, L, Q, H))
        dtypes.append(out_dtype)
    outs = kernel(
        inputs=inputs,
        template=[("T_OUT", out_dtype)],
        grid=(B * L * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    mixed = outs[0] if write_mixed else None
    normed = outs[-1] if norm else None
    return mixed, normed


def attention_residual_reference(states, query, norm_weight=None, eps=1e-6, norm_eps=1e-6,
                                 out_dtype=mx.bfloat16):
    """Unfused equivalent (the mlx-vlm formulation); used on the CPU and in tests."""
    H = states[-1].shape[-1]
    stacked = mx.stack(list(mx.broadcast_arrays(*states)), axis=-2).astype(mx.float32)
    keys = mx.fast.rms_norm(stacked, None, eps)
    scores = (keys @ query.reshape(-1, H).astype(mx.float32).T) * H**-0.5
    weights = mx.softmax(scores, axis=-2).swapaxes(-1, -2)
    mixed = weights @ keys
    normed = None
    if norm_weight is not None:
        w = norm_weight.reshape(-1, H)
        normed = (mx.fast.rms_norm(mixed, None, norm_eps) * w.astype(mx.float32)).astype(out_dtype)
    return mixed, normed


# ---------------------------------------------------------------------------
# Grouped NVFP4 expert matvec: each run of sorted routes that share an expert
# reads that expert's packed weights once per V routes. MLX's gather_qmm only
# groups routes when there are >= 4 rows per expert on average (>= 512 routes for
# 128 experts); batched decode and speculative verification sit far below that, so
# without this every route re-reads its expert. Adapted from mlx-vlm's
# grouped_masked_native_nvfp4_qmv, extended with a tail block so input widths that
# are not a multiple of 256 (the 1152-wide down projection) work.

_grouped = {}


def _grouped_nvfp4_kernel():
    if "k" not in _grouped:
        from mlx_vlm.models.masked_quantized import _HEADER

        _grouped["k"] = mx.fast.metal_kernel(
            name="inkling_grouped_nvfp4_qmv_tail",
            input_names=["x", "w", "s", "g", "ids"],
            output_names=["out"],
            header=_HEADER,
            compile_options={"math_mode": "safe"},
            source=r"""
uint first = threadgroup_position_in_grid.z;
uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
int row0 = threadgroup_position_in_grid.y * 8 + sg * 4;
int expert = ids[first];
if (first > 0 && ids[first - 1] == expert) return;
int end = first + 1;
while (end < int(ids_shape[0]) && ids[end] == expert) ++end;
constexpr int P = 8;
constexpr int BLOCK = P * 32;
constexpr int FULL = (K / BLOCK) * BLOCK;
constexpr int TAIL = K - FULL;  // multiple of 16; handled by the first TAIL/P lanes
const float gscale = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
for (int begin = first; begin < end; begin += V) {
    thread float xt[V][P];
    thread float result[V][4] = {{0}};
    const device uint8_t* ws = (const device uint8_t*)w + size_t(expert) * N * (K / 2) + row0 * (K / 2) + lane * (P / 2);
    const device uint8_t* scales = s + size_t(expert) * N * (K / 16) + row0 * (K / 16) + lane / (16 / P);
    for (int k = 0; k < FULL; k += BLOCK) {
        #pragma unroll
        for (int v = 0; v < V; ++v) load_vector<T, float, P>(x + min(begin + v, end - 1) * K + lane * P + k, xt[v]);
        for (int r = 0; r < 4; ++r) {
            auto wl = ws + r * (K / 2);
            float scale = dequantize_scale<float, 16>(scales[r * (K / 16)]);
            #pragma unroll
            for (int v = 0; v < V; ++v) result[v][r] += qdot<float, P, 4>(wl, xt[v], scale);
        }
        ws += BLOCK / 2;
        scales += BLOCK / 16;
    }
    if (TAIL > 0 && int(lane) < TAIL / P) {
        #pragma unroll
        for (int v = 0; v < V; ++v) load_vector<T, float, P>(x + min(begin + v, end - 1) * K + lane * P + FULL, xt[v]);
        for (int r = 0; r < 4; ++r) {
            auto wl = ws + r * (K / 2);
            float scale = dequantize_scale<float, 16>(scales[r * (K / 16)]);
            #pragma unroll
            for (int v = 0; v < V; ++v) result[v][r] += qdot<float, P, 4>(wl, xt[v], scale);
        }
    }
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            float acc = simd_sum(result[v][r]);
            if (lane == 0 && begin + v < end) out[(begin + v) * N + row0 + r] = T(acc * gscale);
        }
    }
}
""",
        )
    return _grouped["k"]


def _grouped_nvfp4_kernel_v2():
    """Run-wide variant: per k-block, every route of a run (up to RUN) accumulates
    against the same 4 weight rows, so DRAM weight reads happen once per run chunk
    and repeated route reads hit L1. Registers: result[RUN][4] plus one x chunk."""
    if "k2" not in _grouped:
        from mlx_vlm.models.masked_quantized import _HEADER

        _grouped["k2"] = mx.fast.metal_kernel(
            name="inkling_grouped_nvfp4_qmv_run",
            input_names=["x", "w", "s", "g", "ids"],
            output_names=["out"],
            header=_HEADER,
            compile_options={"math_mode": "safe"},
            source=r"""
uint first = threadgroup_position_in_grid.z;
uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
int row0 = threadgroup_position_in_grid.y * 8 + sg * 4;
int expert = ids[first];
if (first > 0 && ids[first - 1] == expert) return;
int end = first + 1;
while (end < int(ids_shape[0]) && ids[end] == expert) ++end;
constexpr int P = 8;
constexpr int BLOCK = P * 32;
constexpr int FULL = (K / BLOCK) * BLOCK;
constexpr int TAIL = K - FULL;
const float gscale = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
const device uint8_t* wbase = (const device uint8_t*)w + size_t(expert) * N * (K / 2) + row0 * (K / 2) + lane * (P / 2);
const device uint8_t* sbase = s + size_t(expert) * N * (K / 16) + row0 * (K / 16) + lane / (16 / P);
for (int begin = first; begin < end; begin += RUN) {
    int count = min(RUN, end - begin);
    thread float result[RUN][4] = {{0}};
    thread float xt[P];
    const device uint8_t* ws = wbase;
    const device uint8_t* scales = sbase;
    for (int k = 0; k < FULL; k += BLOCK) {
        float sc[4];
        for (int r = 0; r < 4; ++r) sc[r] = dequantize_scale<float, 16>(scales[r * (K / 16)]);
        for (int v = 0; v < RUN; ++v) {
            if (v >= count) break;
            load_vector<T, float, P>(x + (begin + v) * K + lane * P + k, xt);
            for (int r = 0; r < 4; ++r) result[v][r] += qdot<float, P, 4>(ws + r * (K / 2), xt, sc[r]);
        }
        ws += BLOCK / 2;
        scales += BLOCK / 16;
    }
    if (TAIL > 0 && int(lane) < TAIL / P) {
        float sc[4];
        for (int r = 0; r < 4; ++r) sc[r] = dequantize_scale<float, 16>(scales[r * (K / 16)]);
        for (int v = 0; v < RUN; ++v) {
            if (v >= count) break;
            load_vector<T, float, P>(x + (begin + v) * K + lane * P + FULL, xt);
            for (int r = 0; r < 4; ++r) result[v][r] += qdot<float, P, 4>(ws + r * (K / 2), xt, sc[r]);
        }
    }
    for (int v = 0; v < RUN; ++v) {
        if (v >= count) break;
        for (int r = 0; r < 4; ++r) {
            float acc = simd_sum(result[v][r]);
            if (lane == 0) out[(begin + v) * N + row0 + r] = T(acc * gscale);
        }
    }
}
""",
        )
    return _grouped["k2"]


def grouped_nvfp4_qmv_run(x, linear, ids, run=16):
    from mlx_vlm.models.masked_quantized import _global_scale

    k, n = linear.input_dims, linear.output_dims
    assert n % 8 == 0 and k % 16 == 0 and x.shape == (ids.shape[0], k)
    return _grouped_nvfp4_kernel_v2()(
        inputs=[x, linear.weight, linear.scales, _global_scale(linear), ids],
        template=[("T", x.dtype), ("K", k), ("N", n), ("RUN", run)],
        grid=(32, n // 8 * 2, ids.shape[0]),
        threadgroup=(32, 2, 1),
        output_shapes=[(ids.shape[0], n)],
        output_dtypes=[x.dtype],
    )[0]


def supports_grouped_nvfp4(linear):
    from mlx_vlm.models.masked_quantized import supports_masked_qmv

    return supports_masked_qmv(linear, input_alignment=16) and linear.input_dims % 16 == 0


def grouped_nvfp4_qmv(x, linear, ids, v=4):
    """x: (R, K) one input row per route, in route order; ids: (R,) int32 expert per
    route, sorted. Returns (R, N) in x.dtype."""
    from mlx_vlm.models.masked_quantized import _global_scale

    k, n = linear.input_dims, linear.output_dims
    assert n % 8 == 0 and k % 16 == 0 and x.shape == (ids.shape[0], k)
    return _grouped_nvfp4_kernel()(
        inputs=[x, linear.weight, linear.scales, _global_scale(linear), ids],
        template=[("T", x.dtype), ("K", k), ("N", n), ("V", v)],
        grid=(32, n // 8 * 2, ids.shape[0]),
        threadgroup=(32, 2, 1),
        output_shapes=[(ids.shape[0], n)],
        output_dtypes=[x.dtype],
    )[0]
