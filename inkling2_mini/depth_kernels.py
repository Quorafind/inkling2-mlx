"""Single-token depth mix: the block attention residual over pre-normalized blocks.

kernels.attention_residual recomputes the RMS of every block on every call (three dependent
reductions in one threadgroup: block RMS, scores, output RMS). In one decode token every
block is produced once and then read by every later layer visit, so here each state's
normalized copy is kept the first time a call sees it (the call's own partial state is
normalized anyway and written out as `xnorm`), and a call needs two reductions: the
partial's RMS together with all scores, then the output RMS.

The MLP-input call also takes the attention output and runs the post-attention 4-tap short
conv + residual add in the same kernel (prefill_kernels.short_conv for one token), writing
the FP32 residual and the next conv state.

Same arithmetic as kernels.attention_residual and short_conv; only FP32 summation orders
differ. Up to MAX_TOKENS tokens (decode and speculative verification, which must agree);
the fused conv only for one token. Longer inputs keep kernels.attention_residual.
"""

import math

import mlx.core as mx

from . import kernels

_k = {}
TG = 512
EPS = 1e-6
ENABLED = True  # A/B switch


class NormCache:
    """Normalized (FP32, (T, H)) copies of the states of one single-token forward pass, by
    object identity (the state is kept alive with its entry, so ids are not reused)."""

    def __init__(self):
        self.d = {}

    def put(self, x, xn):
        self.d[id(x)] = (x, xn)

    def get(self, x):
        e = self.d.get(id(x))
        if e is not None and e[0] is x:
            return e[1]
        xn = mx.fast.rms_norm(x.reshape(-1, x.shape[-1]).astype(mx.float32), None, EPS)
        self.put(x, xn)
        return xn


ACTIVE = None  # the NormCache of the running single-token forward pass, if any


MAX_TOKENS = 8  # speculative verification blocks use the same kernels as decode


def begin(tokens):
    global ACTIVE
    ACTIVE = NormCache() if (ENABLED and tokens <= MAX_TOKENS and kernels.use_kernels()) else None
    return ACTIVE


def end():
    global ACTIVE
    ACTIVE = None


def _source(NB, Q, H, write_mixed, norm, w_rows, conv, base):
    S = NB + 1
    blocks = "\n".join(
        f"            {{ const float v = b{i}[tok * H + h]; xb[{i}][k] = v; for (int q = 0; q < Q; ++q) acc[q * S + {i}] += v * qv[q]; }}"
        for i in range(NB))
    if conv:
        last = f"""
            const float a0 = float(a[h]);
            const float s0 = state[h], s1 = state[H + h], s2 = state[2 * H + h];
            const float c = float(cw[h * 4 + 0]) * s0 + float(cw[h * 4 + 1]) * s1
                          + float(cw[h * 4 + 2]) * s2 + float(cw[h * 4 + 3]) * a0;
            const float rc = float(T((c + a0) / {math.sqrt(5):.9e}f));  // the conv rounds to T
            const float r = {"rc + float(base[h])" if base else "rc"};
            res_out[h] = r;
            nstate[h] = s1; nstate[H + h] = s2; nstate[2 * H + h] = a0;"""
    else:
        last = "\n            const float r = float(x[tok * H + h]);"
    mixed = """
            if (WRITE_MIXED) mixed[(tok * Q + q) * H + h] = o;""" if write_mixed else ""
    out_norm = ""
    if norm:
        out_norm = f"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = 0; q < Q; ++q) {{ const float v = simd_sum(sq[q]); if (lane == 0) red[q][sg] = v; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(Q)) {{
        float t = 0.0f;
        for (int g = 0; g < NR; ++g) t += red[tid][g];
        tot[tid] = metal::rsqrt(t / float(H) + {EPS:.9e}f);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = 0; q < Q; ++q) {{
        const float rq = tot[q];
        for (int k = 0; k < PER; ++k) {{
            const int h = tid + k * TG;
            if (h < H) normed[(tok * Q + q) * H + h] = static_cast<T_OUT>(outv[q][k] * rq * float(w[{"q" if w_rows > 1 else "0"} * H + h]));
        }}
    }}"""
    return f"""
    constexpr int H = {H}, Q = {Q}, NB = {NB}, S = {S}, PER = (H + TG - 1) / TG, NR = TG / 32, NS = Q * S + 1;
    constexpr bool WRITE_MIXED = {str(write_mixed).lower()};
    const uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint tok = threadgroup_position_in_grid.x;   // conv: one token only
    float acc[NS];
    for (int i = 0; i < NS; ++i) acc[i] = 0.0f;
    float xb[NB > 0 ? NB : 1][PER], xl[PER];
    for (int k = 0; k < PER; ++k) {{
        const int h = tid + k * TG;
        xl[k] = 0.0f;
        for (int i = 0; i < NB; ++i) xb[i][k] = 0.0f;
        if (h < H) {{
            float qv[Q];
            for (int q = 0; q < Q; ++q) qv[q] = query[q * H + h];
{blocks}{last}
            xl[k] = r;
            acc[Q * S] += r * r;
            for (int q = 0; q < Q; ++q) acc[q * S + NB] += r * qv[q];
        }}
    }}
    threadgroup float red[NS][NR];
    threadgroup float tot[NS];
    for (int i = 0; i < NS; ++i) {{ const float v = simd_sum(acc[i]); if (lane == 0) red[i][sg] = v; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(NS)) {{ float t = 0.0f; for (int g = 0; g < NR; ++g) t += red[tid][g]; tot[tid] = t; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rl = metal::rsqrt(tot[Q * S] / float(H) + {EPS:.9e}f);
    const float scale = metal::rsqrt(float(H));
    for (int k = 0; k < PER; ++k) xl[k] *= rl;   // the partial's normalized values
    float outv[Q][PER], sq[Q];
    for (int q = 0; q < Q; ++q) {{
        float wts[S], m = -INFINITY;
        for (int s = 0; s < S; ++s) {{ wts[s] = tot[q * S + s] * (s == NB ? rl : 1.0f) * scale; m = metal::max(m, wts[s]); }}
        float z = 0.0f;
        for (int s = 0; s < S; ++s) {{ wts[s] = metal::exp(wts[s] - m); z += wts[s]; }}
        sq[q] = 0.0f;
        for (int k = 0; k < PER; ++k) {{
            float o = 0.0f;
            for (int s = 0; s < NB; ++s) o += wts[s] * xb[s][k];
            o += wts[NB] * xl[k];
            o /= z;
            outv[q][k] = o;
            sq[q] += o * o;
            const int h = tid + k * TG;
            if (h < H) {{{mixed}
            }}
        }}
    }}
    for (int k = 0; k < PER; ++k) {{ const int h = tid + k * TG; if (h < H) xnorm[tok * H + h] = xl[k]; }}{out_norm}
"""


def depth_mix(blocks, query, last=None, norm_weight=None, write_mixed=False, out_dtype=mx.bfloat16,
              conv=None):
    """One to MAX_TOKENS tokens (rows of a batch or positions of a verify block; the mix is
    per token). blocks: earlier states (each (B, L, H)); query (H,) or (Q, H).
    The partial state is `last` ((.., H)), or, with conv = (attn, state, conv_weight, base),
    base + T(conv(attn)) computed here (base None: no residual add).
    Returns dict with 'mixed' (B, L, Q, H) FP32, 'normed' (B, L, Q, H) out_dtype, 'residual'
    (1, 1, H) FP32 and 'state' (1, 3, H) FP32 (conv only, one token)."""
    cache = ACTIVE
    query = query.reshape(-1, query.shape[-1]).astype(mx.float32)
    Q, H = query.shape
    NB = len(blocks)
    norm = norm_weight is not None
    w_rows = norm_weight.reshape(-1, H).shape[0] if norm else 1
    has_conv = conv is not None
    has_base = has_conv and conv[3] is not None
    key = (NB, Q, H, write_mixed, norm, w_rows, has_conv, has_base)
    if key not in _k:
        names = [f"b{i}" for i in range(NB)]
        names += (["a", "state", "cw"] + (["base"] if has_base else [])) if has_conv else ["x"]
        names += ["query"] + (["w"] if norm else [])
        outs = (["res_out", "nstate"] if has_conv else []) + (["mixed"] if write_mixed else []) \
            + (["normed"] if norm else []) + ["xnorm"]
        _k[key] = (mx.fast.metal_kernel(
            name=f"inkling_depth_mix_b{NB}_q{Q}_m{int(write_mixed)}_n{int(norm)}{w_rows}_c{int(has_conv)}{int(has_base)}",
            input_names=names, output_names=outs, source=_source(*key)), outs)
    kern, outs = _k[key]
    T_ = 1 if has_conv else last.size // H
    inputs = [cache.get(b) for b in blocks]
    inputs = [b if b.shape[0] == T_ else mx.broadcast_to(b, (T_, H)) for b in inputs]  # e.g. a shared state
    template = [("TG", TG)]
    if has_conv:
        attn, state, cw, base = conv
        assert attn.size == H, "the fused conv takes one token"
        T_ = 1
        inputs += [attn.reshape(H), state.astype(mx.float32).reshape(3 * H), cw.reshape(H * 4)]
        inputs += [base.reshape(H)] if has_base else []
        template.append(("T", attn.dtype))
    else:
        T_ = last.size // H
        inputs.append(last.reshape(T_, H))
    inputs.append(query)
    if norm:
        inputs.append(norm_weight.reshape(-1, H))
        template.append(("T_OUT", out_dtype))
    shapes = {"res_out": (H,), "nstate": (3 * H,), "mixed": (T_, Q, H), "normed": (T_, Q, H), "xnorm": (T_, H)}
    dtypes = {"res_out": mx.float32, "nstate": mx.float32, "mixed": mx.float32, "normed": out_dtype,
              "xnorm": mx.float32}
    res = kern(inputs=inputs, template=template, grid=(TG * T_, 1, 1), threadgroup=(TG, 1, 1),
               output_shapes=[shapes[o] for o in outs], output_dtypes=[dtypes[o] for o in outs])
    r = dict(zip(outs, res))
    out = {}
    if has_conv:
        out["residual"] = r["res_out"].reshape(1, 1, H)
        out["state"] = r["nstate"].reshape(1, 3, H)
        cache.put(out["residual"], r["xnorm"])
    else:
        cache.put(last, r["xnorm"])
    lead = (1, 1) if has_conv else tuple(last.shape[:-1])
    if write_mixed:
        out["mixed"] = r["mixed"].reshape(*lead, Q, H)
    if norm:
        out["normed"] = r["normed"].reshape(*lead, Q, H)
    return out
