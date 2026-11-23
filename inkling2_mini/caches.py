"""KV caches that hold Inkling's attention-sink slot in place.

Inkling attention has a per-query sink: an extra all-zero key / value whose logit is
the sink projection. The straightforward way (mlx-vlm) concatenates a zero slot onto
the fetched K and V of every layer visit, and sliding-window layers also reorder the
ring buffer into temporal order first, so each decoded token copies every cached K/V
once or twice (48 visits per token). These caches keep a zero slot inside the buffer
and let single-token decode attend to the buffer in its physical layout; only the
small additive mask is permuted. The arithmetic is unchanged.

* SinkKVCache (full-attention layers): a KVCache whose slot right after the last
  entry is kept zero, so [entries, sink] is a view.
* SinkRingKVCache (sliding layers): a W-slot ring plus a permanent zero slot W.
  Multi-token updates (prefill, verification, rollback replay) keep the standard
  RotatingKVCache contract (chronological keys, W - 1 + S of them) and rebuild the
  ring in temporal order; single-token decode writes in place and returns the whole
  physical buffer with its layout.

Both stay compatible with mlx-vlm's speculative cache transactions (offset trim for
SinkKVCache, snapshot / replay for the rotating one) and with batch merging.
"""

import mlx.core as mx
from mlx_vlm.models.cache import KVCache, RotatingKVCache


class SinkKVCache(KVCache):
    def update_and_fetch_with_sink(self, keys, values):
        prev = self.offset
        S = keys.shape[2]
        need = prev + S + 1
        if self.keys is None or need > self.keys.shape[2]:
            B, H, _, D = keys.shape
            have = 0 if self.keys is None else prev
            n_steps = (need - have + self.step - 1) // self.step
            new_k = mx.zeros((B, H, n_steps * self.step, D), keys.dtype)
            new_v = mx.zeros((B, H, n_steps * self.step, values.shape[3]), values.dtype)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                self.keys = mx.concatenate([self.keys[..., :prev, :], new_k], axis=2)
                self.values = mx.concatenate([self.values[..., :prev, :], new_v], axis=2)
        self.offset += S
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        # The slot after the last entry is the sink; a rollback may have left data there.
        self.keys[..., self.offset : self.offset + 1, :] = 0
        self.values[..., self.offset : self.offset + 1, :] = 0
        return self.keys[..., : self.offset + 1, :], self.values[..., : self.offset + 1, :]


class SinkRingKVCache(RotatingKVCache):
    def __init__(self, max_size):
        super().__init__(max_size=max_size, keep=0)

    # Layout: when offset <= W the entries sit chronologically in slots [0, offset);
    # once full, the oldest entry is at slot _idx. Slot W is always zero.
    def _temporal_order(self, v):
        W = self.max_size
        if self.offset <= W:
            return v[..., : self.offset, :]
        idx = self._idx
        if idx == 0:
            return v[..., :W, :]
        return mx.concatenate([v[..., idx:W, :], v[..., :idx, :]], axis=2)

    def _rebuild(self, chronological_k, chronological_v):
        W = self.max_size
        n = chronological_k.shape[2]
        keep = min(n, W)
        B, H, _, D = chronological_k.shape
        pad = W + 1 - keep
        self.keys = mx.concatenate(
            [chronological_k[..., n - keep :, :], mx.zeros((B, H, pad, D), chronological_k.dtype)], axis=2
        )
        self.values = mx.concatenate(
            [chronological_v[..., n - keep :, :], mx.zeros((B, H, pad, chronological_v.shape[3]), chronological_v.dtype)],
            axis=2,
        )
        self._idx = keep % W

    def update_and_fetch(self, keys, values):
        """Standard RotatingKVCache contract: returns, chronologically, the last
        W - 1 cached entries followed by the S new ones."""
        W = self.max_size
        if self.keys is None or self.offset == 0:
            out_k, out_v = keys, values
            all_k, all_v = keys, values
        else:
            old_k, old_v = self._temporal_order(self.keys), self._temporal_order(self.values)
            ctx = min(old_k.shape[2], W - 1)
            all_k = mx.concatenate([old_k, keys], axis=2)
            all_v = mx.concatenate([old_v, values], axis=2)
            start = old_k.shape[2] - ctx
            out_k, out_v = all_k[..., start:, :], all_v[..., start:, :]
        self.offset += keys.shape[2]
        self._rebuild(all_k, all_v)
        return out_k, out_v

    def update_and_fetch_with_sink(self, keys, values):
        """Single-token decode. Returns (keys, values, next_slot, count): the whole
        physical buffer (W + 1 slots, slot W zero), the slot holding the oldest entry
        once full, and how many slots are valid."""
        W = self.max_size
        if self.keys is None:
            B, H, _, D = keys.shape
            self.keys = mx.zeros((B, H, W + 1, D), keys.dtype)
            self.values = mx.zeros((B, H, W + 1, values.shape[3]), values.dtype)
            self._idx = 0
        i = self._idx
        self.keys[..., i : i + 1, :] = keys
        self.values[..., i : i + 1, :] = values
        self.offset += 1
        self._idx = (i + 1) % W
        return self.keys, self.values, self._idx, min(self.offset, W)

    def size(self):
        return min(self.offset, self.max_size)

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v


class QuantSinkKVCache(KVCache):
    """Full-attention cache holding K and V affine-quantized along head_dim.

    Only the 8 full-attention visits grow with context (32 KB per token in BF16);
    8 bits halves that, 4 bits quarters it. Decode attends with two quantized
    matmuls (QK^T, PV) and the sink as an extra logit column; multi-token calls
    dequantize and use the fused SDPA kernel. Rollback is the KVCache offset trim.
    """

    def __init__(self, bits=8, group_size=64):
        super().__init__()
        self.bits, self.group_size = bits, group_size
        self.qk = self.qv = None

    def _grow(self, like_k, like_v, need):
        def alloc(q, cap):
            return [mx.zeros((*a.shape[:2], cap, a.shape[3]), a.dtype) for a in q]

        cap = ((need + self.step - 1) // self.step) * self.step
        if self.qk is None:
            self.qk, self.qv = alloc(like_k, cap), alloc(like_v, cap)
            return
        prev = self.offset
        new_k, new_v = alloc(like_k, cap - prev), alloc(like_v, cap - prev)
        self.qk = [mx.concatenate([a[..., :prev, :], n], axis=2) for a, n in zip(self.qk, new_k)]
        self.qv = [mx.concatenate([a[..., :prev, :], n], axis=2) for a, n in zip(self.qv, new_v)]

    def update_and_fetch_quant(self, keys, values):
        if keys.shape[-1] % self.group_size:
            self.group_size = 32  # small test models (head_dim 32)
        qk = mx.quantize(keys, self.group_size, self.bits)
        qv = mx.quantize(values, self.group_size, self.bits)
        prev, S = self.offset, keys.shape[2]
        if self.qk is None or prev + S > self.qk[0].shape[2]:
            self._grow(qk, qv, prev + S)
        for buf, new in zip(self.qk, qk):
            buf[..., prev : prev + S, :] = new
        for buf, new in zip(self.qv, qv):
            buf[..., prev : prev + S, :] = new
        self.offset += S
        return [a[..., : self.offset, :] for a in self.qk], [a[..., : self.offset, :] for a in self.qv]

    def dequantized(self, qk, qv, dtype):
        def deq(q):
            return mx.dequantize(*q, group_size=self.group_size, bits=self.bits).astype(dtype)

        return deq(qk), deq(qv)

    def update_and_fetch(self, keys, values):
        # The standard contract (used by mlx-vlm helpers): chronological BF16 views.
        qk, qv = self.update_and_fetch_quant(keys, values)
        return self.dequantized(qk, qv, keys.dtype)

    @property
    def state(self):
        if self.qk is None:
            return None, None
        return [a[..., : self.offset, :] for a in self.qk], [a[..., : self.offset, :] for a in self.qv]

    @property
    def nbytes(self):
        return 0 if self.qk is None else sum(a.nbytes for a in (*self.qk, *self.qv))
