"""Fused single-token decode attention over an 8-bit KV cache (full-attention layers).

At long contexts decode time grows with the K/V each full-attention visit reads
(8 visits x 32 KB per cached token in BF16: 2.1 GB per decoded token at 64k). This
reads an affine 8-bit cache instead (half the bytes), dequantizing in registers.

Adapted from MLX's sdpa_vector_2pass_1_gqa / sdpa_vector_2pass_2 (split-K flash
decoding): one threadgroup per (KV head, key block), whose 3 simdgroups each scan a
third of the block for all 3 query heads of that KV head, so each key is read once.
Changes:
  * K / V rows are 8-bit affine (4 values per uint32 per lane, a scale / bias per 64);
  * the relative-position bias is added for keys within rel_extent of the query
    (a heads x rel_extent table; zero beyond, as in the model);
  * the per-query sink logit (zero value) is folded into the second pass;
  * partial outputs stay FP32 (MLX stores them in the activation dtype).
"""

import mlx.core as mx

_k = {}

_PASS1 = r"""
    // grid: (KV heads, B, blocks) threadgroups of (32, G * L) threads. Simdgroup g scans
    // sub-chunk g % G of this key block for query position l = g / G and all G query
    // heads of the KV head; the L simdgroups sharing a sub-chunk read the same rows (DRAM
    // once, then cache). Each writes its own partial; pass 2 reduces them.
    const int N = meta[0], L = meta[2];
    const uint3 tid = threadgroup_position_in_grid;
    const uint3 tpg = threadgroups_per_grid;
    const int simd_lid = thread_index_in_simdgroup;
    const int g = simdgroup_index_in_threadgroup;
    const int c = g % G, l = g / G;
    const int kv_head = tid.x, batch = tid.y, block = tid.z, blocks = tpg.z;
    const int num_kv = tpg.x, num_q = num_kv * G;
    const int base_head = batch * num_q + kv_head * G;
    const int qpos = N - L + l;                    // absolute position of this query
    const int chunk = (N + blocks - 1) / blocks;
    const int kstart = block * chunk, kend = min(qpos + 1, min(N, kstart + chunk));  // causal
    const int sub = (chunk + G - 1) / G;
    const int s0 = kstart + c * sub, s1 = min(kend, s0 + sub);

    const int64_t kw_s = kw_strides[2], ks_s = ks_strides[2], vw_s = vw_strides[2], vs_s = vs_strides[2];
    const device uint32_t* kwp = kw + int64_t(batch) * kw_strides[0] + kv_head * kw_strides[1] + int64_t(s0) * kw_s + simd_lid;
    const device T* ksp = ks + int64_t(batch) * ks_strides[0] + kv_head * ks_strides[1] + int64_t(s0) * ks_s + simd_lid / 16;
    const device T* kbp = kb + int64_t(batch) * ks_strides[0] + kv_head * ks_strides[1] + int64_t(s0) * ks_s + simd_lid / 16;
    const device uint32_t* vwp = vw + int64_t(batch) * vw_strides[0] + kv_head * vw_strides[1] + int64_t(s0) * vw_s + simd_lid;
    const device T* vsp = vs + int64_t(batch) * vs_strides[0] + kv_head * vs_strides[1] + int64_t(s0) * vs_s + simd_lid / 16;
    const device T* vbp = vb + int64_t(batch) * vs_strides[0] + kv_head * vs_strides[1] + int64_t(s0) * vs_s + simd_lid / 16;

    // queries: (B * H, L, 128); bias: (B * H, L, REL)
    float q[G][4];
    for (int j = 0; j < G; j++)
        for (int i = 0; i < 4; i++)
            q[j][i] = SCALE * float(queries[((base_head + j) * L + l) * 128 + simd_lid * 4 + i]);

    float mx_[G], se[G], o[G][4];
    for (int j = 0; j < G; j++) {
        mx_[j] = -INFINITY; se[j] = 0;
        for (int i = 0; i < 4; i++) o[j][i] = 0;
    }

    for (int t = s0; t < s1; t++) {
        uint32_t kq = kwp[0], vq = vwp[0];
        float ksc = float(ksp[0]), kbi = float(kbp[0]);
        float vsc = float(vsp[0]), vbi = float(vbp[0]);
        float kr[4], vr[4];
        for (int i = 0; i < 4; i++) {
            kr[i] = ksc * float((kq >> (8 * i)) & 0xff) + kbi;
            vr[i] = vsc * float((vq >> (8 * i)) & 0xff) + vbi;
        }
        kwp += kw_s; ksp += ks_s; kbp += ks_s;
        vwp += vw_s; vsp += vs_s; vbp += vs_s;
        const int dist = qpos - t;
        for (int j = 0; j < G; j++) {
            float score = 0;
            for (int i = 0; i < 4; i++) score += q[j][i] * kr[i];
            score = simd_sum(score);
            if (dist < REL) score += float(bias[(int64_t(base_head + j) * L + l) * REL + dist]);
            float new_max = max(mx_[j], score);
            float factor = fast::exp(mx_[j] - new_max);
            float e = fast::exp(score - new_max);
            mx_[j] = new_max;
            se[j] = se[j] * factor + e;
            for (int i = 0; i < 4; i++) o[j][i] = o[j][i] * factor + e * vr[i];
        }
    }

    const int parts = blocks * G, part = block * G + c;
    for (int j = 0; j < G; j++) {
        const int row = (base_head + j) * L + l;   // (head, query position)
        device float* op = partials + (int64_t(row) * parts + part) * 128 + simd_lid * 4;
        for (int i = 0; i < 4; i++) op[i] = o[j][i];
        if (simd_lid == 0) {
            sums[row * parts + part] = se[j];
            maxs[row * parts + part] = mx_[j];
        }
    }
"""

_PASS2 = r"""
    // grid: (B * H * L) threadgroups of 128 threads; thread d owns output element d.
    const int row = threadgroup_position_in_grid.x;
    const int d = thread_position_in_threadgroup.x;
    const int parts = meta[1];
    const device float* m = maxs + row * parts;
    const device float* s = sums + row * parts;
    float sink = float(sinks[row]);
    float M = sink;
    for (int b = 0; b < parts; b++) M = max(M, m[b]);
    float denom = fast::exp(sink - M), acc = 0;
    for (int b = 0; b < parts; b++) {
        float w = (m[b] == -INFINITY) ? 0.0f : fast::exp(m[b] - M);
        denom += w * s[b];
        acc += w * partials[(int64_t(row) * parts + b) * 128 + d];
    }
    out[row * 128 + d] = static_cast<T>(acc / denom);
"""


def _kernels():
    if "p1" not in _k:
        _k["p1"] = mx.fast.metal_kernel(
            name="inkling_q8_decode_pass1",
            input_names=["queries", "kw", "ks", "kb", "vw", "vs", "vb", "bias", "meta"],
            output_names=["partials", "sums", "maxs"],
            source=_PASS1.replace("SCALE", f"{128 ** -0.5:.9e}f"),
            ensure_row_contiguous=False,
        )
        _k["p2"] = mx.fast.metal_kernel(
            name="inkling_q8_decode_pass2",
            input_names=["partials", "sums", "maxs", "sinks", "meta"],
            output_names=["out"],
            source=_PASS2,
        )
    return _k["p1"], _k["p2"]


def blocks_for(n):
    return max(1, min(128, (n + 511) // 512))


MAX_QUERIES = 8


def supported(q, qk, kv):
    return (
        q.shape[2] <= MAX_QUERIES
        and q.shape[-1] == 128
        and kv.bits == 8
        and kv.group_size == 64
        and qk[0].dtype == mx.uint32
        and q.shape[1] // qk[0].shape[1] == 3
        and mx.default_device() == mx.gpu
    )


def q8_decode_attention(q, qk, qv, rel_logits, sinks, rel_extent):
    """q: (B, H, L, 128), L <= 8, the last L positions (causal); qk / qv: [weights
    (B, KV, N, 32) uint32, scales, biases (B, KV, N, 2)] (views of the cache);
    rel_logits: (B, H, L, rel_extent) bias by distance; sinks: (B, H, L) sink logits.
    Returns (B, H, L, 128) in q.dtype."""
    B, H, L, D = q.shape
    KV, N = qk[0].shape[1], qk[0].shape[2]
    G = H // KV
    blocks = blocks_for(N)
    parts = blocks * G
    p1, p2 = _kernels()
    meta = mx.array([N, parts, L], mx.int32)
    partials, sums, maxs = p1(
        inputs=[mx.contiguous(q.reshape(B * H, L, D)), *qk, *qv,
                mx.contiguous(rel_logits.reshape(B * H, L, rel_extent).astype(q.dtype)), meta],
        template=[("T", q.dtype), ("G", G), ("REL", rel_extent)],
        grid=(KV * 32, B * G * L, blocks),
        threadgroup=(32, G * L, 1),
        output_shapes=[(B * H * L, parts, D), (B * H * L, parts), (B * H * L, parts)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = p2(
        inputs=[partials, sums, maxs, mx.contiguous(sinks.reshape(B * H * L).astype(mx.float32)), meta],
        template=[("T", q.dtype)],
        grid=(B * H * L * D, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(B * H * L, D)],
        output_dtypes=[q.dtype],
    )[0]
    return out.reshape(B, H, L, D)


# --------------------------------------------------- sliding-window ring decode
_RING = r"""
    // Like pass 1 above for one query, over a BF16 ring buffer (B, KV, W + 1, 128):
    // physical slot t holds chronological index c = (t - oldest) mod W once full
    // (slots [0, n) are chronological before that), at distance n - 1 - c.
    const int n = meta[0], oldest = meta[2], full = meta[3];
    const uint3 tid = threadgroup_position_in_grid;
    const uint3 tpg = threadgroups_per_grid;
    const int simd_lid = thread_index_in_simdgroup;
    const int g = simdgroup_index_in_threadgroup;
    const int kv_head = tid.x, batch = tid.y, block = tid.z, blocks = tpg.z;
    const int num_q = tpg.x * G;
    const int base_head = batch * num_q + kv_head * G;
    const int chunk = (n + blocks - 1) / blocks;
    const int kstart = block * chunk, kend = min(n, kstart + chunk);
    const int sub = (chunk + G - 1) / G;
    const int s0 = kstart + g * sub, s1 = min(kend, s0 + sub);
    const int64_t ks = k_strides[2], vs = v_strides[2];
    const device T* kp = k + int64_t(batch) * k_strides[0] + kv_head * k_strides[1] + int64_t(s0) * ks + simd_lid * 4;
    const device T* vp = v + int64_t(batch) * v_strides[0] + kv_head * v_strides[1] + int64_t(s0) * vs + simd_lid * 4;

    float q[G][4];
    for (int j = 0; j < G; j++)
        for (int i = 0; i < 4; i++)
            q[j][i] = SCALE * float(queries[(base_head + j) * 128 + simd_lid * 4 + i]);
    float mx_[G], se[G], o[G][4];
    for (int j = 0; j < G; j++) {
        mx_[j] = -INFINITY; se[j] = 0;
        for (int i = 0; i < 4; i++) o[j][i] = 0;
    }
    for (int t = s0; t < s1; t++) {
        float kr[4], vr[4];
        for (int i = 0; i < 4; i++) { kr[i] = float(kp[i]); vr[i] = float(vp[i]); }
        kp += ks; vp += vs;
        const int c = full ? (t - oldest + W) % W : t;
        const int dist = n - 1 - c;
        for (int j = 0; j < G; j++) {
            float score = 0;
            for (int i = 0; i < 4; i++) score += q[j][i] * kr[i];
            score = simd_sum(score);
            if (dist < REL) score += float(bias[int64_t(base_head + j) * REL + dist]);
            float new_max = max(mx_[j], score);
            float factor = fast::exp(mx_[j] - new_max);
            float e = fast::exp(score - new_max);
            mx_[j] = new_max;
            se[j] = se[j] * factor + e;
            for (int i = 0; i < 4; i++) o[j][i] = o[j][i] * factor + e * vr[i];
        }
    }
    const int parts = blocks * G, part = block * G + g;
    for (int j = 0; j < G; j++) {
        const int row = base_head + j;
        device float* op = partials + (int64_t(row) * parts + part) * 128 + simd_lid * 4;
        for (int i = 0; i < 4; i++) op[i] = o[j][i];
        if (simd_lid == 0) { sums[row * parts + part] = se[j]; maxs[row * parts + part] = mx_[j]; }
    }
"""


def ring_decode_attention(q, keys, values, n, oldest, window, rel_logits, sinks, rel_extent, blocks=8):
    """One query per head over a SinkRingKVCache buffer (B, KV, W + 1, 128).
    n: valid slots; oldest: slot of the oldest entry once the ring is full.
    rel_logits: (B, H, rel_extent); sinks: (B, H). Returns (B, H, 1, 128)."""
    B, H, _, D = q.shape
    KV = keys.shape[1]
    G = H // KV
    if "ring" not in _k:
        _k["ring"] = mx.fast.metal_kernel(
            name="inkling_ring_decode_pass1",
            input_names=["queries", "k", "v", "bias", "meta"],
            output_names=["partials", "sums", "maxs"],
            source=_RING.replace("SCALE", f"{128 ** -0.5:.9e}f"),
            ensure_row_contiguous=False,
        )
    _, p2 = _kernels()
    blocks = max(1, min(blocks, n))
    parts = blocks * G
    full = int(n >= window)
    meta = mx.array([n, parts, oldest if full else 0, full], mx.int32)
    partials, sums, maxs = _k["ring"](
        inputs=[mx.contiguous(q.reshape(B * H, D)), keys, values,
                mx.contiguous(rel_logits.reshape(B * H, rel_extent).astype(q.dtype)), meta],
        template=[("T", q.dtype), ("G", G), ("REL", rel_extent), ("W", window)],
        grid=(KV * 32, B * G, blocks), threadgroup=(32, G, 1),
        output_shapes=[(B * H, parts, D), (B * H, parts), (B * H, parts)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = p2(
        inputs=[partials, sums, maxs, mx.contiguous(sinks.reshape(B * H).astype(mx.float32)), meta],
        template=[("T", q.dtype)],
        grid=(B * H * D, 1, 1), threadgroup=(D, 1, 1),
        output_shapes=[(B * H, D)], output_dtypes=[q.dtype],
    )[0]
    return out.reshape(B, H, 1, D)
