"""Fused prefill attention for Inkling's full-attention layers, with the relative
position bias computed only inside its band.

Inkling's full-attention logits are  q.k * scale + bias(i, j) + causal, where the
bias is a per-(query, head) profile over backward distance d = i - j that is zero
for d >= rel_extent (1024), plus one per-query sink logit with a zero value. The
mlx-vlm path materializes the whole (heads x queries x keys) additive mask, which
at long contexts is gigabytes per layer visit (3.2 GB for 1024 queries over 64k
keys) and is read back by SDPA: ~70 % of the attention time, and the reason chunks
had to shrink.

This kernel is MLX's NAX flash-attention kernel (steel_attention_nax.h,
attention_nax) with three changes:
  * the mask read is replaced by the band bias: a (heads x queries x rel_extent)
    table rel_logits = relative_states @ proj (~50 MB for 1024 queries), added only
    on key blocks that intersect the band; far blocks are plain causal attention;
  * the sink is per query (initial row max / sum of the online softmax) instead of
    per head, so no zero key / value is concatenated;
  * the runtime sizes come from a small int array, so one pipeline serves every
    chunk and context length; K / V may be strided views of the cache buffers.
Scores accumulate in FP32 on the matrix units exactly as in the built-in kernel.
"""

import mlx.core as mx

BQ, BK, WM, WN = 64, 32, 4, 1
_kernels = {}

def _inline(rel, seen):
    """Expand MLX's `#include "mlx/..."` headers from the installed wheel; metal_kernel
    sources have no include path for them (system headers stay as includes)."""
    import re
    from pathlib import Path

    root = Path(mx.__file__).parent / "include"
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in (root / rel).read_text().splitlines():
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(_inline(m.group(1), seen))
        elif line.strip() != "#pragma once":
            out.append(line)
    return "\n".join(out)


_HEADER = _inline("mlx/backend/metal/kernels/steel/attn/nax.h", set()) + r"""
using namespace mlx::steel;

struct BandMaxOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); } };
struct BandSumOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; } };
struct BandMulOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; } };
struct BandExpSubOp { template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); } };
"""

_SOURCE = r"""
  // meta: [qL, kL, gqa_factor]
  const int qL = meta[0], kL = meta[1], gqa = meta[2];
  const int NK = (kL + BK - 1) / BK;
  const int NQ_aligned = qL / BQ, NK_aligned = kL / BK;
  const int qL_rem = qL - NQ_aligned * BQ, kL_rem = kL - NK_aligned * BK;
  const int qL_off = kL - qL;
  const int H = Q_shape[1];

  uint simd_lane_id = thread_index_in_simdgroup;
  uint simd_group_id = simdgroup_index_in_threadgroup;
  uint3 tid = threadgroup_position_in_grid;
  ulong3 tidl{tid.x, tid.y, tid.z};

  const int64_t q_s0 = Q_strides[0], q_s1 = Q_strides[1], q_s2 = Q_strides[2];
  const int64_t k_s0 = K_strides[0], k_s1 = K_strides[1], k_s2 = K_strides[2];
  const int64_t v_s0 = V_strides[0], v_s1 = V_strides[1], v_s2 = V_strides[2];
  const int64_t o_s2 = BD, o_s1 = int64_t(qL) * BD, o_s0 = int64_t(H) * qL * BD;

  const device T* Qp = Q + tidl.z * q_s0 + tidl.y * q_s1 + tidl.x * BQ * q_s2;
  ulong kv_head = int(tid.y) / gqa;
  const device T* Kp = K + tidl.z * k_s0 + kv_head * k_s1;
  const device T* Vp = V + tidl.z * v_s0 + kv_head * v_s1;
  device T* Op = O + tidl.z * o_s0 + tidl.y * o_s1 + tidl.x * BQ * o_s2;
  // bias: [B, H, qL, REL] contiguous; sinks: [B, H, qL] contiguous
  const device T* biasp = bias + (int64_t(tid.z) * H + tid.y) * int64_t(qL) * REL;
  const device T* sinkp = sinks + (int64_t(tid.z) * H + tid.y) * int64_t(qL);

  const float scale2 = 0.08838834764831845f * 1.44269504089f;  // 128 ** -0.5 * log2(e)
  constexpr short kU = 16;
  constexpr int kNWarps = WM * WN;
  constexpr int TQ = BQ / (kNWarps * kU);
  constexpr int TD = BD / kU;
  constexpr int TV = BD / kU;
  constexpr short TK = BK / kU;
  static_assert(TQ == 1, "TQ");

  using otile_t = NAXTile<float, TQ, TV>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * simd_group_id;
  Qp += tm * int(q_s2);
  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<float, kRowsPT> max_score;
  metal::vec<float, kRowsPT> sum_score{0};
  // Per-query sink: one extra zero-valued logit per row.
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    const int r = int(tid.x) * BQ + tm + i * otile_t::kFragRowsJump + sm;
    max_score[i] = r < qL ? M_LOG2E_F * float(sinkp[r]) : 0.0f;
    sum_score[i] = 1;
  }

  int q_max = (tid.x + 1) * BQ + qL_off;
  int kb_lim = min(NK, (q_max + BK - 1) / BK);
  int q_min = max(0, int(tid.x) * BQ + qL_off);
  int kb_min_causal = q_min / BK;
  // First key block that can be within REL of this query block (band start).
  int kb_min_band = max(0, (q_min - REL + 1) / BK);
  // Sliding window: blocks entirely older than the window are skipped.
  int kb_first = SLIDING > 0 ? max(0, (q_min - SLIDING + 1) / BK) : 0;
  Kp += kb_first * BK * int(k_s2);
  Vp += kb_first * BK * int(v_s2);

  const bool is_last_q = int(tid.x) == NQ_aligned;
  const short lim_rows_q = qL_rem - tm;
  const short lim_rows_k = kL_rem;

  for (int kb = kb_first; kb < kb_lim; kb++) {
    const int is_last_k = (kb == NK_aligned);
    using stile_t = NAXTile<float, TQ, TK>;
    stile_t Stile;
    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;
          const int Q_load_off = iq * kU * int(q_s2) + id * kU;
          const int K_load_off = ik * kU * int(k_s2) + id * kU;
          if (!ALIGN_Q && is_last_q) Qtile.load_rows(Qp + Q_load_off, int(q_s2), lim_rows_q - iq * kU);
          else Qtile.load(Qp + Q_load_off, int(q_s2));
          if (!ALIGN_K && is_last_k) Ktile.load_rows(Kp + K_load_off, int(k_s2), lim_rows_k - ik * kU);
          else Ktile.load(Kp + K_load_off, int(k_s2));
          stile_t::NAXFrag_t::mma(Stile.frag_at(iq, ik), Stile.frag_at(iq, ik + 1), Qtile.frag_at(0, 0),
                                  metal::false_type{}, Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) Stile.elems()[ii] *= scale2;

    constexpr auto neg_inf = Limits<float>::finite_min;
    if (!ALIGN_K && is_last_k) {
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Causal mask and band bias (only on blocks that can intersect the band).
    if (kb >= kb_min_band) {
      const int base_row = int(tid.x) * BQ + tm;  // local query row
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const int r = base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const int c = base_col + ik * kU + jj + sn;
              const int dist = r + qL_off - c;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              if (dist < 0 || (SLIDING > 0 && dist >= SLIDING)) {
                fg[loc] = neg_inf;
              } else if (dist < REL && r < qL) {
                fg[loc] += M_LOG2E_F * float(biasp[int64_t(r) * REL + dist]);
              }
            }
          }
        }
      }
    } else if (kb >= kb_min_causal) {
      const int base_row = int(tid.x) * BQ + qL_off + tm;
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r = base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    metal::vec<float, kRowsPT> new_max;
    metal::vec<float, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
    Stile.template row_reduce<BandMaxOp>(new_max);
    Stile.template row_bin_op<BandExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) sum_score[i] = sum_score[i] * factor[i];
    Stile.template row_reduce<BandSumOp>(sum_score);
    Otile.template row_bin_op<BandMulOp>(factor);
    simdgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TV; id += 2) {
        if (BD == 128 && id == 4) threadgroup_barrier(mem_flags::mem_none);
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;
          const int V_load_off = ik * kU * int(v_s2) + id * kU;
          if (!ALIGN_K && is_last_k) Vtile.load_rows(Vp + V_load_off, int(v_s2), lim_rows_k - ik * kU);
          else Vtile.load(Vp + V_load_off, int(v_s2));
          otile_t::NAXFrag_t::mma(Otile.frag_at(iq, id), Otile.frag_at(iq, id + 1), Stile.frag_at(iq, ik),
                                  metal::false_type{}, Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
        }
      }
    }
    Kp += BK * int(k_s2);
    Vp += BK * int(v_s2);
  }

  threadgroup_barrier(mem_flags::mem_none);
  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
  Otile.template row_bin_op<BandMulOp>(rcp);

  Op += tm * int(o_s2);
  if (!ALIGN_Q && is_last_q) {
    if (lim_rows_q <= 0) return;
    Otile.store_rows(Op, int(o_s2), lim_rows_q);
  } else {
    Otile.store(Op, int(o_s2));
  }
"""


def _kernel():
    if "k" not in _kernels:
        _kernels["k"] = mx.fast.metal_kernel(
            name="inkling_band_attention_nax",
            input_names=["Q", "K", "V", "bias", "sinks", "meta"],
            output_names=["O"],
            header=_HEADER,
            source=_SOURCE,
            ensure_row_contiguous=False,
        )
    return _kernels["k"]


def available():
    if "ok" not in _kernels:
        try:
            q = mx.zeros((1, 2, 64, 128), mx.bfloat16)
            k = mx.zeros((1, 1, 64, 128), mx.bfloat16)
            mx.eval(band_attention(q, k, k, mx.zeros((1, 2, 64, 16), mx.bfloat16), mx.zeros((1, 2, 64), mx.bfloat16), 16, 0))
            _kernels["ok"] = True
        except Exception:  # no NAX / MetalPerformancePrimitives on this GPU
            _kernels["ok"] = False
    return _kernels["ok"] and mx.default_device() == mx.gpu


def band_attention(q, k, v, rel_logits, sinks, rel_extent, sliding=0):
    """q: (B, H, qL, 128); k, v: (B, KV, kL, 128), the queries being the last qL
    keys (causal, lower-right aligned); rel_logits: (B, H, qL, rel_extent), the
    bias at backward distance d for d < rel_extent; sinks: (B, H, qL) logits of the
    per-query zero sink; sliding: window size (keys at distance >= sliding are
    masked) or 0 for full attention. Returns (B, H, qL, 128) in q.dtype."""
    B, H, qL, D = q.shape
    kL = k.shape[2]
    assert D == 128 and rel_logits.shape[-1] == rel_extent
    meta = mx.array([qL, kL, H // k.shape[1]], mx.int32)
    nq = (qL + BQ - 1) // BQ
    return _kernel()(
        inputs=[mx.contiguous(q), k, v, mx.contiguous(rel_logits.astype(q.dtype)),
                mx.contiguous(sinks.astype(q.dtype)), meta],
        template=[("T", q.dtype), ("BQ", BQ), ("BK", BK), ("BD", D), ("WM", WM), ("WN", WN),
                  ("REL", rel_extent), ("SLIDING", int(sliding)),
                  ("ALIGN_Q", qL % BQ == 0), ("ALIGN_K", kL % BK == 0)],
        grid=(nq * 32, H * WM, B * WN),
        threadgroup=(32, WM, WN),
        output_shapes=[(B, H, qL, D)],
        output_dtypes=[q.dtype],
    )[0]
