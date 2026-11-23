"""Fused Metal kernels for Inkling 2 Mini's prefill glue.

Profiling a 1024-token prefill chunk showed about half the time outside the GEMMs. These
kernels replace the biggest pieces with single passes over memory, computing the same
arithmetic as the MLX op chains they replace:

* short_conv: the depthwise causal convolution (taps 4) with its residual and
  1/sqrt(K+1) scaling. It replaces fp32 casts, a state concatenation, a generic conv1d,
  the residual add and rounding (mlx-vlm only has a kernel for <= 8 tokens). An optional
  residual input fuses the decoder layer's following add.
* dynamic_conv: the same for the full-attention layers' K/V convolutions, whose
  taps are static per channel plus dynamic per (token, head).
* combine_routes: MoE un-sort + weighted sum of the top-8 expert outputs, reading
  the sorted expert outputs through the inverse permutation. It replaces a 37 MB
  gather, an FP32 cast and a reduction.
"""

import math

import mlx.core as mx

from . import decode_kernels

_k = {}


def use_kernels():
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# ---------------------------------------------------------------- short conv
_CONV = r"""
    // grid: (C, ceil(L / TB), B); each thread: TB consecutive tokens of one channel.
    uint c = thread_position_in_grid.x, tb = thread_position_in_grid.y, b = thread_position_in_grid.z;
    const int L = x_shape[1], C = x_shape[2];
    if (int(c) >= C) return;
    int t0 = int(tb) * TB;
    if (t0 >= L) return;
    const float w0 = float(w[c * 4 + 0]), w1 = float(w[c * 4 + 1]);
    const float w2 = float(w[c * 4 + 2]), w3 = float(w[c * 4 + 3]);
    // window of the three inputs before t0: from the state when t0 - k < 0
    float xm3, xm2, xm1;
    {
        float h[3];
        for (int j = 0; j < 3; ++j) {
            int t = t0 - 3 + j;
            h[j] = t >= 0 ? float(x[(int64_t(b) * L + t) * C + c]) : float(state[(int64_t(b) * 3 + (3 + t)) * C + c]);
        }
        xm3 = h[0]; xm2 = h[1]; xm1 = h[2];
    }
    int t1 = min(L, t0 + TB);
    for (int t = t0; t < t1; ++t) {
        int64_t i = (int64_t(b) * L + t) * C + c;
        float x0 = float(x[i]);
        float acc = w0 * xm3 + w1 * xm2 + w2 * xm1 + w3 * x0;
        float r = float(T((acc + x0) / CONV_SCALE));  // the convolution rounds to x.dtype
        STORE
        xm3 = xm2; xm2 = xm1; xm1 = x0;
    }
    if (t1 == L) {  // this block holds the last token: the new state is the last three inputs
        int64_t sb = int64_t(b) * 3 * C + c;
        nstate[sb] = xm3; nstate[sb + C] = xm2; nstate[sb + 2 * C] = xm1;
    }
"""

_DYNCONV = r"""
    // out[t, c] = sum_lag x[t - lag, c] * (static[lag, c] + dyn[t, head(c), lag]); then (out + x) / 3
    uint c = thread_position_in_grid.x, tb = thread_position_in_grid.y, b = thread_position_in_grid.z;
    const int L = x_shape[1], C = x_shape[2];
    if (int(c) >= C) return;
    int t0 = int(tb) * TB;
    if (t0 >= L) return;
    const int head = int(c) / (C / HEADS);
    const float s0 = float(sw[0 * C + c]), s1 = float(sw[1 * C + c]);
    const float s2 = float(sw[2 * C + c]), s3 = float(sw[3 * C + c]);
    float xm3, xm2, xm1;
    {
        float h[3];
        for (int j = 0; j < 3; ++j) {
            int t = t0 - 3 + j;
            h[j] = t >= 0 ? float(x[(int64_t(b) * L + t) * C + c]) : float(state[(int64_t(b) * 3 + (3 + t)) * C + c]);
        }
        xm3 = h[0]; xm2 = h[1]; xm1 = h[2];
    }
    int t1 = min(L, t0 + TB);
    for (int t = t0; t < t1; ++t) {
        int64_t i = (int64_t(b) * L + t) * C + c;
        const device float* d = dyn + ((int64_t(b) * L + t) * HEADS + head) * 4;
        float x0 = float(x[i]);
        float acc = x0 * (s0 + d[0]) + xm1 * (s1 + d[1]) + xm2 * (s2 + d[2]) + xm3 * (s3 + d[3]);
        out[i] = static_cast<T>((acc + x0) / CONV_SCALE);
        xm3 = xm2; xm2 = xm1; xm1 = x0;
    }
    if (t1 == L) {
        int64_t sb = int64_t(b) * 3 * C + c;
        nstate[sb] = xm3; nstate[sb + C] = xm2; nstate[sb + 2 * C] = xm1;
    }
"""

TB = 16


def _conv_kernel(has_res):
    key = ("conv", has_res)
    if key not in _k:
        store = "out[i] = static_cast<O>(r + float(res[i]));" if has_res else "out[i] = static_cast<O>(r);"
        _k[key] = mx.fast.metal_kernel(
            name=f"inkling_short_conv_res{int(has_res)}",
            input_names=["x", "state", "w"] + (["res"] if has_res else []),
            output_names=["out", "nstate"],
            source=_CONV.replace("STORE", store),
        )
    return _k[key]


def short_conv(x, state, weight, residual=None, out_dtype=None, with_state=False):
    """x: (B, L, C); state: (B, 3, C) FP32 inputs before x (zeros at sequence start);
    weight: (C, 4, 1) (MLX conv layout) or (C, 4). Returns T((conv + x) / sqrt(5)) in
    x.dtype, or out_dtype(that + residual) when a residual is given; with_state also
    returns the next state (the last three inputs, FP32)."""
    B, L, C = x.shape
    w = weight.reshape(C, 4)
    has_res = residual is not None
    out_dtype = out_dtype or (residual.dtype if has_res else x.dtype)
    inputs = [x, state.astype(mx.float32), w] + ([residual] if has_res else [])
    out = _conv_kernel(has_res)(
        inputs=inputs,
        template=[("T", x.dtype), ("O", out_dtype), ("TB", TB)],
        grid=(C, (L + TB - 1) // TB, B),
        threadgroup=(min(256, C), 1, 1),
        output_shapes=[(B, L, C), (B, 3, C)],
        output_dtypes=[out_dtype, mx.float32],
    )
    return (out[0], out[1]) if with_state else out[0]


def dynamic_conv(x, state, static_weight, dyn, heads, with_state=False):
    """x: (B, L, C); state: (B, 3, C); static_weight: (4, C); dyn: (B, L, heads * 4)
    FP32 per-token taps. Returns T((conv + x) / 3)."""
    B, L, C = x.shape
    key = ("dyn",)
    if key not in _k:
        _k[key] = mx.fast.metal_kernel(
            name="inkling_dynamic_conv", input_names=["x", "state", "sw", "dyn"],
            output_names=["out", "nstate"], source=_DYNCONV,
        )
    out = _k[key](
        inputs=[x, state.astype(mx.float32), static_weight, dyn.astype(mx.float32)],
        template=[("T", x.dtype), ("TB", TB), ("HEADS", heads)],
        grid=(C, (L + TB - 1) // TB, B),
        threadgroup=(min(256, C), 1, 1),
        output_shapes=[(B, L, C), (B, 3, C)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return (out[0], out[1]) if with_state else out[0]


# ------------------------------------------------------------- MoE combine
_COMBINE = r"""
    // grid: (H / 4, T, 1). out[t, h] = sum_k w[t, k] * y[inv[t * K + k], h]
    uint h4 = thread_position_in_grid.x, t = thread_position_in_grid.y;
    const int H = y_shape[1];
    int h = int(h4) * 4;
    if (h >= H) return;
    float4 acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int r = int(inv[t * K + k]);
        float wk = float(w[t * K + k]);
        const device T* row = y + int64_t(r) * H + h;
        acc += wk * float4(float(row[0]), float(row[1]), float(row[2]), float(row[3]));
    }
    device T* o = out + int64_t(t) * H + h;
    o[0] = T(acc.x); o[1] = T(acc.y); o[2] = T(acc.z); o[3] = T(acc.w);
"""


def combine_routes(y, inverse, weights):
    """y: (R, H) sorted expert outputs; inverse: (R,) sorted position of route
    t * K + k; weights: (T, K). Returns (T, H) in y.dtype."""
    R, H = y.shape
    T_, K = weights.shape
    if "comb" not in _k:
        _k["comb"] = mx.fast.metal_kernel(
            name="inkling_combine_routes", input_names=["y", "inv", "w"],
            output_names=["out"], source=_COMBINE,
        )
    return _k["comb"](
        inputs=[y, inverse.astype(mx.uint32), weights.astype(mx.float32)],
        template=[("T", y.dtype), ("K", K)],
        grid=(H // 4, T_, 1),
        threadgroup=(min(256, H // 4), 1, 1),
        output_shapes=[(T_, H)],
        output_dtypes=[y.dtype],
    )[0]


def _patch_scale(src, k):
    # Same FP32 divisor as the reference (math.sqrt(k) rounded to float).
    return src.replace("CONV_SCALE", f"{math.sqrt(k):.9e}f")


_CONV = _patch_scale(_CONV, 5)
_DYNCONV = _patch_scale(_DYNCONV, 9)


# ------------------------------------------------------------ module glue
def _next_state(state, x):
    """The last three inputs (FP32), as mlx-vlm's _next_conv_state without a mask."""
    if x.shape[1] >= 3:
        return x[:, -3:].astype(mx.float32)
    return mx.concatenate([state, x.astype(mx.float32)], axis=1)[:, -3:]


def conv_fast_ok(x, cache, mask):
    # Prefill chunks, and single tokens / short blocks when the decode fusions are on
    # (the kernel also emits the next conv state), unless a speculative transaction is
    # recording conv states.
    return (
        (x.shape[1] > 8 or decode_kernels.active())
        and mask is None
        and x.dtype == mx.bfloat16
        and not getattr(cache, "is_speculating", False)
        and use_kernels()
    )


def make_fast_convs():
    from mlx_vlm.models.inkling2.language import DynamicShortConvolution, ShortConvolution

    class FastShortConvolution(ShortConvolution):
        def __call__(self, x, cache=None, mask=None, residual=None, out_dtype=None):
            if self.kernel_size == 4 and conv_fast_ok(x, cache, mask):
                state = cache[self.conv_idx] if cache is not None else None
                if state is None:
                    state = mx.zeros((x.shape[0], 3, x.shape[-1]), mx.float32)
                out, nstate = short_conv(x, state, self.conv.weight, residual, out_dtype, with_state=True)
                if cache is not None:
                    cache[self.conv_idx] = nstate
                return out
            out = super().__call__(x, cache, mask)
            if residual is None:
                return out if out_dtype is None else out.astype(out_dtype)
            return (out.astype(mx.float32) + residual.astype(mx.float32)).astype(out_dtype or residual.dtype)

    class FastDynamicShortConvolution(DynamicShortConvolution):
        def __call__(self, x, dynamic_input, cache=None, mask=None):
            if self.kernel_size == 4 and conv_fast_ok(x, cache, mask):
                state = cache[self.conv_idx] if cache is not None else None
                if state is None:
                    state = mx.zeros((x.shape[0], 3, x.shape[-1]), mx.float32)
                dyn = self.weight_proj(dynamic_input).astype(mx.float32)
                out, nstate = dynamic_conv(x, state, self.static_weight, dyn, self.num_heads, with_state=True)
                if cache is not None:
                    cache[self.conv_idx] = nstate
                return out
            return super().__call__(x, dynamic_input, cache, mask)

    return FastShortConvolution, FastDynamicShortConvolution


FastShortConvolution, FastDynamicShortConvolution = make_fast_convs()


def speed_up_convs(*modules):
    """Re-class mlx-vlm conv modules in place (same parameters)."""
    from mlx_vlm.models.inkling2.language import DynamicShortConvolution, ShortConvolution

    for m in modules:
        if type(m) is ShortConvolution:
            m.__class__ = FastShortConvolution
        elif type(m) is DynamicShortConvolution:
            m.__class__ = FastDynamicShortConvolution


# ------------------------------------------------------------ message gates
# A loop layer on passes 2-3 mixes the previous pass's messages m into its
# inputs h:  out = (1 - tanh(F z)) * h + tanh(G z) * m,  z = D [rms(h), rms(m)],
# then RMS-normalizes out with the layer's input norm. The two small GEMMs stay
# MLX matmuls; the element-wise chains before and after them are one kernel each.

_GATE_PREP = r"""
    // zin[g, t] = bf16([h * rsqrt(mean(h^2) + eps), m * rsqrt(mean(m^2) + eps)])
    uint t = threadgroup_position_in_grid.x, g = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const int N = m_shape[0];
    const device float* hp = h + (int64_t(t) * QH + (QH > 1 ? g : 0)) * H;
    const device float* mp = m + (int64_t(t) * MQ + MOFF + g) * H;
    float hv[PER], mv[PER], sh = 0, sm2 = 0;
    for (int i = 0; i < PER; ++i) {
        int j = tid + i * TG;
        hv[i] = j < H ? hp[j] : 0; mv[i] = j < H ? mp[j] : 0;
        sh += hv[i] * hv[i]; sm2 += mv[i] * mv[i];
    }
    threadgroup float red[2][TG / 32];
    sh = simd_sum(sh); sm2 = simd_sum(sm2);
    if (lane == 0) { red[0][sg] = sh; red[1][sg] = sm2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sh = 0; sm2 = 0;
    for (int i = 0; i < TG / 32; ++i) { sh += red[0][i]; sm2 += red[1][i]; }
    float rh = metal::rsqrt(sh / float(H) + EPS), rm = metal::rsqrt(sm2 / float(H) + EPS);
    device bfloat16_t* zp = zin + (int64_t(g) * N + t) * (2 * H);
    for (int i = 0; i < PER; ++i) {
        int j = tid + i * TG;
        if (j < H) { zp[j] = bfloat16_t(hv[i] * rh); zp[H + j] = bfloat16_t(mv[i] * rm); }
    }
"""

_GATE_APPLY = r"""
    // out[g, t] = bf16(rmsnorm((1 - tanh(f)) * h + tanh(gg) * m) * w)
    uint t = threadgroup_position_in_grid.x, g = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const int N = m_shape[0];
    const device float* hp = h + (int64_t(t) * QH + (QH > 1 ? g : 0)) * H;
    const device float* mp = m + (int64_t(t) * MQ + MOFF + g) * H;
    const device float* fp = fg + (int64_t(g) * N + t) * (2 * H);
    float ov[PER], ss = 0;
    for (int i = 0; i < PER; ++i) {
        int j = tid + i * TG;
        float o = 0;
        if (j < H) o = (1.0f - metal::precise::tanh(fp[j])) * hp[j] + metal::precise::tanh(fp[H + j]) * mp[j];
        ov[i] = o; ss += o * o;
    }
    threadgroup float red[TG / 32];
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0;
    for (int i = 0; i < TG / 32; ++i) ss += red[i];
    float r = metal::rsqrt(ss / float(H) + EPS);
    const device bfloat16_t* wp = w + (QW > 1 ? g : 0) * H;
    device bfloat16_t* op = out + (int64_t(g) * N + t) * H;
    for (int i = 0; i < PER; ++i) {
        int j = tid + i * TG;
        if (j < H) op[j] = bfloat16_t(ov[i] * r * float(wp[j]));
    }
"""

_GTG = 256


def _gate_kernel(kind):
    if kind not in _k:
        src = _GATE_PREP if kind == "prep" else _GATE_APPLY
        names = ["h", "m"] if kind == "prep" else ["h", "m", "fg", "w"]
        _k[kind] = mx.fast.metal_kernel(
            name=f"inkling_gate_{kind}", input_names=names,
            output_names=["zin"] if kind == "prep" else ["out"],
            source=src.replace("EPS", "1e-6f"),
        )
    return _k[kind]


def message_gates(h, messages, moff, gates, down, fg, norm_weight):
    """Gated + RMS-normalized inputs for `gates` consecutive messages starting at
    message slot `moff`. h: (N, QH, H) FP32 (QH 1 = shared by all gates);
    messages: (N, 4, H) FP32; down: (G, r, 2H) BF16; fg: (G, 2H, r) FP32 [F; G];
    norm_weight: (H,) or (G, H). Returns (G, N, H) BF16."""
    N, QH, H = h.shape
    per = (H + _GTG - 1) // _GTG
    w = norm_weight.reshape(-1, H).astype(mx.bfloat16)
    tmpl = [("H", H), ("PER", per), ("TG", _GTG), ("QH", QH), ("MQ", messages.shape[1]), ("MOFF", moff)]
    zin = _gate_kernel("prep")(
        inputs=[h, messages], template=tmpl, grid=(N * _GTG, gates, 1), threadgroup=(_GTG, 1, 1),
        output_shapes=[(gates, N, 2 * H)], output_dtypes=[mx.bfloat16],
    )[0]
    z = (zin @ down.swapaxes(-1, -2)).astype(mx.float32)  # (G, N, r): BF16 GEMM, rounded as the reference
    fgo = z @ fg.swapaxes(-1, -2)  # (G, N, 2H) FP32
    return _gate_kernel("apply")(
        inputs=[h, messages, fgo, w], template=tmpl + [("QW", w.shape[0])],
        grid=(N * _GTG, gates, 1), threadgroup=(_GTG, 1, 1),
        output_shapes=[(gates, N, H)], output_dtypes=[mx.bfloat16],
    )[0]
