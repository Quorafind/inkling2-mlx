"""Decode-latency fusions for Inkling 2 Mini.

* FusedAttention: q / gate / r / sink / k / v projections share one weight matrix.
  When all inputs are the same tensor (non-message visits) that is one GEMV instead
  of six; otherwise three (query group, key, value) on row slices of it.
* gates(): the q / k / v message gates of a visit as one batched, compiled
  computation instead of three separate ~15-op chains.
* The block attention residual uses the fused Metal kernel in kernels.py.

Everything here is numerically the same computation as the unfused path (only
reduction order changes); on the CPU the unfused reference functions are used.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import BatchRotatingKVCache, RotatingKVCache
from mlx_vlm.models.inkling.language import banded_additive_mask
from mlx_vlm.models.inkling2.language import Attention, l2_clip

from . import band_attention as band
from . import decode_attention as q8dec
from . import decode_kernels as DK
from .caches import QuantSinkKVCache, SinkKVCache, SinkRingKVCache

_PROJ = ("q_proj", "gate_proj", "r_proj", "sink_proj", "k_proj", "v_proj")
RING_DECODE = True  # single-token decode attends to the ring buffer in place (caches.py)
BAND_MIN_QUERIES = 16  # full-attention calls with at least this many queries use band_attention


@mx.compile
def _gated_output(out, gate):
    return (out.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(out.dtype)


def _make_gates(eps):
    @mx.compile
    def gates(h, m, down, fg):
        """h, m: (G, N, H) FP32; down: (G, r, 2H); fg: (G, 2H, r) FP32 [forget; gain]."""
        H = h.shape[-1]
        z = mx.concatenate([mx.fast.rms_norm(h, None, eps), mx.fast.rms_norm(m, None, eps)], axis=-1)
        z = (z.astype(down.dtype) @ down.swapaxes(-1, -2)).astype(mx.float32)
        fg_out = mx.tanh(z @ fg.swapaxes(-1, -2))
        return (1 - fg_out[..., :H]) * h + fg_out[..., H:] * m

    return gates


gates = _make_gates(1e-6)


def stack_gates(gate_modules):
    down = mx.stack([g.down_weight for g in gate_modules])
    fg = mx.stack([mx.concatenate([g.forget_weight, g.gain_weight], axis=0) for g in gate_modules])
    return down, fg.astype(mx.float32)


class FusedAttention(Attention):
    fused = False

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        from .prefill_kernels import speed_up_convs

        speed_up_convs(self.k_sconv, self.v_sconv)

    def fuse(self):
        mods = [self[n] for n in _PROJ]
        self._sizes = [m.weight.shape[0] for m in mods]
        first = mods[0]
        if isinstance(first, nn.QuantizedLinear):
            same = all(
                isinstance(m, nn.QuantizedLinear)
                and (m.bits, m.group_size, m.mode) == (first.bits, first.group_size, first.mode)
                for m in mods
            )
            if not same:
                raise ValueError("cannot fuse projections of different precisions")
            out = sum(self._sizes)
            fused = nn.QuantizedLinear(
                first.scales.shape[-1] * first.group_size, out, bias=False,
                group_size=first.group_size, bits=first.bits, mode=first.mode,
            )
            fused.weight = mx.concatenate([m.weight for m in mods])
            fused.scales = mx.concatenate([m.scales for m in mods])
            if first.get("biases") is not None:
                fused.biases = mx.concatenate([m.biases for m in mods])
        else:
            if not all(type(m) is nn.Linear and "bias" not in m for m in mods):
                raise ValueError("cannot fuse these projections")
            fused = nn.Linear(first.weight.shape[1], sum(self._sizes), bias=False)
            fused.weight = mx.concatenate([m.weight for m in mods])
        for n in _PROJ:
            delattr(self, n)
        self.qkv_all = fused
        mx.eval(fused.parameters())
        self.fused = True

    def _quantized_decode(self, q, qk, qv, mask, sinks, kv, gate, dtype):
        """One query against a quantized cache: softmax([q.K^T * scale + mask, sink]) @ V,
        the sink being a zero key / value with its own logit."""
        B, H, L, D = q.shape
        rep = H // self.n_kv
        S = qk[0].shape[2]
        # FP32 scores and probabilities (fused SDPA also keeps them unrounded).
        qs = (q.astype(mx.float32) * self.head_dim**-0.5).reshape(B, self.n_kv, rep, L, D)
        expand = lambda arrs: [a[:, :, None] for a in arrs]  # noqa: E731 (broadcast over rep)
        scores = mx.quantized_matmul(qs, *expand(qk), transpose=True, group_size=kv.group_size, bits=kv.bits)
        scores = scores + mask.reshape(B, self.n_kv, rep, L, S).astype(mx.float32)
        logits = mx.concatenate([scores, sinks.reshape(B, self.n_kv, rep, L, 1).astype(scores.dtype)], axis=-1)
        probs = mx.softmax(logits, axis=-1, precise=True)[..., :S]
        out = mx.quantized_matmul(probs, *expand(qv), transpose=False, group_size=kv.group_size, bits=kv.bits)
        out = out.reshape(B, H, L, D).astype(dtype)
        out = _gated_output(out.transpose(0, 2, 1, 3).reshape(B, L, -1), gate)
        return self.o_proj(out)

    def _use_band(self, length, b, kv):
        return (
            length >= BAND_MIN_QUERIES
            and b == 1
            and self.head_dim == 128
            and getattr(kv, "left_padding", None) is None
            and band.available()
        )

    def _band(self, q, k, v, r, rel_proj, sinks, gate, dtype):
        """Prefill through the band-bias flash kernel (band_attention.py): no
        (heads x queries x keys) mask, no zero K/V concatenation."""
        b, _, length, _ = q.shape
        rel_logits = (r @ rel_proj).transpose(0, 2, 1, 3)  # (B, heads, L, rel_extent)
        out = band.band_attention(q, k, v, rel_logits, sinks[..., 0], self.rel_extent, self.sliding)
        out = _gated_output(out.transpose(0, 2, 1, 3).reshape(b, length, -1), gate)
        return self.o_proj(out)

    def _rows(self, x, a, b):
        f = self.qkv_all
        if isinstance(f, nn.QuantizedLinear):
            return mx.quantized_matmul(
                x, f.weight[a:b], f.scales[a:b], f.get("biases")[a:b] if f.get("biases") is not None else None,
                transpose=True, group_size=f.group_size, bits=f.bits, mode=f.mode,
            )
        return x @ f.weight[a:b].T

    def __call__(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
        if not self.fused:
            return super().__call__(x, key_input, value_input, cache=cache, conv_mask=conv_mask)
        b, length, _ = x.shape
        cfg = self.config
        kv, conv = cache if cache is not None else (None, None)
        nq, ng, nr, ns, nk, nv = self._sizes
        o1, o2, o3 = nq, nq + ng, nq + ng + nr
        o4 = o3 + ns
        o5 = o4 + nk
        same = (key_input is None or key_input is x) and (value_input is None or value_input is x)
        if same:
            y = self.qkv_all(x)
            k, v = y[..., o4:o5], y[..., o5 : o5 + nv]
            key_input = value_input = x
        else:
            y = self._rows(x, 0, o4)
            k = self._rows(key_input, o4, o5)
            v = self._rows(value_input, o5, o5 + nv)
        q, gate, r, sinks = y[..., :o1], y[..., o1:o2], y[..., o2:o3], y[..., o3:o4]

        if self.is_sliding:
            k = self.k_sconv(k, conv, conv_mask)
            v = self.v_sconv(v, conv, conv_mask)
        else:
            k = self.k_sconv(k, key_input, conv, conv_mask)
            v = self.v_sconv(v, value_input, conv, conv_mask)
        log_scaling = (
            (cfg.log_scaling_alpha, cfg.log_scaling_n_floor)
            if not self.is_sliding and cfg.log_scaling_n_floor is not None else None
        )
        offset = kv.offset if kv is not None else 0
        if b == 1 and isinstance(offset, int) and self.head_dim == 128 and DK.active() and y.ndim == 3:
            # clips, value norm, log scale and layout in one kernel (decode_kernels.attn_prep)
            q, k, v, r = DK.attn_prep(
                y, k, v, self.q_scale, self.rel_scale, o2, self.n_heads, self.n_kv, cfg.d_rel,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps, offset, log_scaling,
            )
            prepped = True
        else:
            prepped = False
            q = l2_clip(q.reshape(b, length, self.n_heads, self.head_dim), cfg.qk_l2_clip_threshold, self.q_scale)
            k = l2_clip(k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold)
            v = v.reshape(b, length, self.n_kv, self.head_dim)
            if cfg.use_value_norm_no_gamma:
                v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(x.dtype)
            r = l2_clip(r.reshape(b, length, self.n_heads, cfg.d_rel), cfg.rel_l2_clip_threshold, self.rel_scale)
            if log_scaling is not None:
                positions = mx.arange(length) + mx.array(offset)[..., None] + 1
                tau = 1 + cfg.log_scaling_alpha * mx.log(mx.maximum(positions / cfg.log_scaling_n_floor, 1))
                q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
                r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        if not prepped:
            q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        sinks = sinks.transpose(0, 2, 1)[..., None]  # (B, heads, L, 1): the sink logits
        rel_proj = self.rel_proj.astype(x.dtype)
        if RING_DECODE and isinstance(kv, SinkRingKVCache) and length == 1:
            # Decode on a ring: attend to the physical buffer (slot W is the zero sink);
            # build the mask chronologically and rotate it into the physical layout.
            k, v, oldest, n = kv.update_and_fetch_with_sink(k, v)
            W = kv.max_size
            if b == 1 and DK.active() and self.head_dim == 128:
                # fused ring-buffer decode attention: no mask, roll or concatenation
                rel_logits = (r @ rel_proj).reshape(b, self.n_heads, self.rel_extent)
                out = q8dec.ring_decode_attention(q, k, v, n, oldest, W, rel_logits,
                                                  sinks.reshape(b, self.n_heads), self.rel_extent)
                out = _gated_output(out.transpose(0, 2, 1, 3).reshape(b, length, -1), gate)
                return self.o_proj(out)
            chrono = banded_additive_mask(r, rel_proj, n - 1, n, self.sliding, self.rel_extent,
                                          shape_ref=k[..., :n, :])
            if n < W:
                pad = mx.full((*chrono.shape[:-1], W - n), -1e30, chrono.dtype)
                phys = mx.concatenate([chrono, pad], axis=-1)
            else:
                phys = mx.roll(chrono, oldest, axis=-1) if oldest else chrono
            mask = mx.concatenate([phys, sinks.astype(phys.dtype)], axis=-1)
        elif isinstance(kv, QuantSinkKVCache):
            qk, qv = kv.update_and_fetch_quant(k, v)
            size = kv.offset
            mask = banded_additive_mask(r, rel_proj, size - length, size, self.sliding, self.rel_extent,
                                        shape_ref=qk[0])
            if b == 1 and q8dec.supported(q, qk, kv):
                # fused split-K decode / verify over the 8-bit cache (decode_attention.py)
                rel_logits = (r @ rel_proj).transpose(0, 2, 1, 3)  # (B, heads, L, rel_extent)
                out = q8dec.q8_decode_attention(q, qk, qv, rel_logits, sinks[..., 0], self.rel_extent)
                out = _gated_output(out.transpose(0, 2, 1, 3).reshape(b, length, -1), gate)
                return self.o_proj(out)
            if length == 1:
                return self._quantized_decode(q, qk, qv, mask, sinks, kv, gate, x.dtype)
            k, v = kv.dequantized(qk, qv, x.dtype)
            if self._use_band(length, b, kv):
                return self._band(q, k, v, r, rel_proj, sinks, gate, x.dtype)
            mask = mx.concatenate([mask, sinks.astype(mask.dtype)], axis=-1)
            zeros = mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)
            k = mx.concatenate([k, zeros], axis=2)
            v = mx.concatenate([v, zeros], axis=2)
        elif isinstance(kv, SinkKVCache):
            k, v = kv.update_and_fetch_with_sink(k, v)  # last slot is the zero sink
            size = k.shape[2] - 1
            if self._use_band(length, b, kv):
                return self._band(q, k[..., :size, :], v[..., :size, :], r, rel_proj, sinks, gate, x.dtype)
            mask = banded_additive_mask(r, rel_proj, size - length, size, self.sliding, self.rel_extent,
                                        shape_ref=k[..., :size, :])
            mask = mx.concatenate([mask, sinks.astype(mask.dtype)], axis=-1)
        else:
            if kv is None and self._use_band(length, b, kv):  # one-shot forward, no cache
                return self._band(q, k, v, r, rel_proj, sinks, gate, x.dtype)
            if kv is not None:
                k, v = kv.update_and_fetch(k, v)
                if isinstance(kv, SinkRingKVCache) and self._use_band(length, b, kv):
                    # multi-token update returns the window chronologically (W - 1 + L keys)
                    return self._band(q, k, v, r, rel_proj, sinks, gate, x.dtype)
                if isinstance(kv, RotatingKVCache) and not isinstance(kv, SinkRingKVCache):
                    k, v = kv._temporal_order(k), kv._temporal_order(v)
                elif isinstance(kv, BatchRotatingKVCache) and kv.rotated:
                    k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
            size = k.shape[2]
            mask = banded_additive_mask(
                r, rel_proj, size - length, size, self.sliding, self.rel_extent,
                left_padding=getattr(kv, "left_padding", None), shape_ref=k,
            )
            # A query-dependent sink: a zero key / value whose logit is the sink.
            mask = mx.concatenate([mask, sinks.astype(mask.dtype)], axis=-1)
            zeros = mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)
            k = mx.concatenate([k, zeros], axis=2)
            v = mx.concatenate([v, zeros], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        out = _gated_output(out.transpose(0, 2, 1, 3).reshape(b, length, -1), gate)
        return self.o_proj(out)
