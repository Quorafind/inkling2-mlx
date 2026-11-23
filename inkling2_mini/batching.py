"""Continuous batching: requests join and leave a running decode batch.

A request is prefilled on its own, into its own single-request caches (identical
numerics to plain generation). Its 48 per-visit caches are then converted into one-row
batch caches and appended to the running batch (`CacheList.extend`); a finished row is
extracted back into single-request caches (so the server can keep it for KV reuse)
and filtered out. All rows of a batch decode in lockstep: one weight read serves every
row.

Cache formats:

* single request: SinkKVCache or QuantSinkKVCache (full attention), SinkRingKVCache
  (sliding window), ArraysCache (short convolutions).
* batch: mlx-vlm's BatchKVCache (left-padded, BF16), BatchRotatingKVCache, ArraysCache.

8-bit KV is dequantized on the way in and quantized again on the way out; tokens decoded
inside the batch are BF16 until then.

Pipelining: each step builds the next forward pass before reading the previous step's
samples, so CPU-side work (graph building, detokenizing, sending deltas) overlaps the
GPU. The batch's next inputs are therefore one lazy (B, 1) array, the previous step's
output itself (reading a derived array instead, e.g. per-row slices concatenated, waits
for part of the current step too: 7 % slower at 8 rows); a row's `emitted` says whether
its next input token has already been handed to the client (a freshly added row's
pending token has).
"""

import numpy as np

import mlx.core as mx
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, BatchRotatingKVCache, CacheList, KVCache

from .caches import QuantSinkKVCache, SinkKVCache, SinkRingKVCache
from .generate import GenState, make_sampler


# ------------------------------------------------------------ cache conversion
def _mergeable(cl):
    kv, conv = cl[0], cl[1]
    if isinstance(kv, QuantSinkKVCache):
        plain = KVCache()
        if kv.qk is not None and kv.offset:
            qk, qv = kv.state
            plain.keys, plain.values = kv.dequantized(qk, qv, qk[1].dtype)  # the activations' dtype
        plain.offset = kv.offset
        kv = plain
    return CacheList(kv, conv)


def to_batch(cache):
    """Single-request caches (any KV precision) -> one-row batch caches."""
    return [CacheList.merge([_mergeable(c)]) for c in cache]


def extract_row(batch, i, kv_bits=None):
    """Row i of batch caches -> single-request caches (8/4-bit full-attention KV with
    kv_bits, BF16 otherwise)."""
    out = []
    for cl in batch:
        kv, conv = cl[0], cl[1]
        if isinstance(kv, BatchRotatingKVCache):
            r = kv.extract(i)  # chronological, at most the window
            s = SinkRingKVCache(kv.max_size)
            if r.keys is not None and r.keys.shape[2]:
                s._rebuild(r.keys, r.values)
            s.offset = int(r.offset)
        elif isinstance(kv, BatchKVCache):
            r = kv.extract(i)
            n = 0 if r.keys is None else r.keys.shape[2]
            if kv_bits:
                s = QuantSinkKVCache(bits=kv_bits)
                if n:
                    if r.keys.shape[-1] % s.group_size:
                        s.group_size = 32
                    s.qk = list(mx.quantize(r.keys, s.group_size, kv_bits))
                    s.qv = list(mx.quantize(r.values, s.group_size, kv_bits))
            else:
                s = SinkKVCache()
                if n:
                    s.keys, s.values = r.keys, r.values
            s.offset = n
        else:
            raise TypeError(f"cannot extract a row from {type(kv).__name__}")
        c = ArraysCache(len(conv.cache))
        c.cache = [None if a is None else mx.contiguous(a[i : i + 1]) for a in conv.cache]
        out.append(CacheList(s, c))
    mx.eval([a for cl in out for a in _cache_arrays(cl)])
    return out


def _cache_arrays(cl):
    kv, conv = cl[0], cl[1]
    arrays = [a for a in conv.cache if a is not None]
    if isinstance(kv, QuantSinkKVCache):
        arrays += [*(kv.qk or []), *(kv.qv or [])]
    elif kv.keys is not None:
        arrays += [kv.keys, kv.values]
    return arrays


def row_bytes(text_config, length, passes):
    """BF16 batch-cache bytes of one row padded to `length` tokens: the sliding rings
    are fixed, the full-attention visits grow with the (padded) length."""
    cfg = text_config
    per = cfg.num_key_value_heads * cfg.head_dim * 2 * 2  # K and V, BF16
    n = cfg.visits(passes)
    layers = [a if a < cfg.num_hidden_layers else cfg.loop_start + (a - cfg.num_hidden_layers) % cfg.num_loop_layers
              for a in range(n)]
    sliding = sum(1 for layer in layers if cfg.layer_is_sliding(layer))
    return per * (sliding * cfg.sliding_window_size + (n - sliding) * length)


# ------------------------------------------------------------------- sampling
class RowSampler:
    """One row's sampling settings; seeded rows draw from their own key."""

    def __init__(self, temperature=1.0, top_p=1.0, top_k=0, seed=None):
        self.params = (float(temperature), float(top_p), int(top_k or 0))
        self.greedy = self.params[0] <= 0
        self.sample = make_sampler(*self.params)
        self.key = mx.random.key(int(seed)) if seed is not None and not self.greedy else None

    def next_key(self):
        if self.key is None:
            return None
        self.key, sub = mx.random.split(self.key)
        return sub

    def __call__(self, logits):
        return self.sample(logits, key=self.next_key()).reshape(-1)


def sample_rows(logits, samplers):
    """logits (B, V) -> (B,) tokens, rows grouped by identical settings."""
    if all(s.greedy for s in samplers):
        return mx.argmax(logits, axis=-1)
    groups = {}
    for i, s in enumerate(samplers):
        groups.setdefault(("own", i) if s.key is not None else s.params, []).append(i)
    if len(groups) == 1:
        return samplers[0](logits)
    parts, order = [], []
    for idx in groups.values():
        parts.append(samplers[idx[0]](logits[mx.array(idx)]))
        order += idx
    return mx.concatenate(parts)[mx.array(np.argsort(order))]


# ---------------------------------------------------------------------- batch
class Row:
    """A decoding row. `state` holds its single-request GenState while it is outside
    the batch (and the draft cache / unconsumed draft features while inside)."""

    def __init__(self, sampler, state):
        self.sampler = sampler
        self.state = state
        self.emitted = True  # whether the row's next input token was handed out already
        self.length = state.length
        self.feats = [] if state.hidden is None else [state.hidden]


class Batch:
    """Rows decoding in lockstep with shared batch caches (one pass count)."""

    FEATS_CONSOLIDATE = 64

    def __init__(self, bot, passes, capture=None):
        self.bot, self.passes, self.capture = bot, passes, capture
        self.rows, self.cache = [], None
        self.next = None  # (B, 1) int32: every row's next input token
        self.steps = 0

    def __len__(self):
        return len(self.rows)

    def max_length(self):
        return max((r.length for r in self.rows), default=0)

    def add(self, row):
        """row.state: single-request caches, every emitted token but the last processed
        (state.pending == [last])."""
        st = row.state
        if len(st.pending) != 1:
            raise ValueError("a row joins the batch with exactly one pending token")
        part = to_batch(st.cache)
        if self.cache is None:
            self.cache = part
        else:
            for c, p in zip(self.cache, part):
                c.extend(p)
        token = mx.array([[st.pending[0]]], mx.int32)
        self.next = token if self.next is None else mx.concatenate([self.next, token], axis=0)
        row.emitted, row.length = True, st.length
        row.feats = [] if st.hidden is None else [st.hidden]
        st.cache, st.hidden, st.pending = None, None, []  # the batch owns them now
        self.rows.append(row)

    def step(self):
        """One decode step for every row. Returns [(row, token)] for the tokens to
        emit now: each row's input of this step, unless it was emitted already."""
        lm = self.bot.model.language_model
        inp = self.next
        if self.capture is not None:
            out, caps = lm.model(inp, cache=self.cache, capture=self.capture)
            feats = mx.concatenate(caps, axis=-1)
        else:
            out, feats = lm.model(inp, cache=self.cache), None
        nxt = sample_rows(lm.logits(out).reshape(len(self.rows), -1), [r.sampler for r in self.rows])
        nxt = nxt.reshape(-1, 1).astype(mx.int32)
        mx.async_eval(nxt, *([] if feats is None else [feats]))
        values = [v[0] for v in inp.tolist()]  # the previous step's samples (pipelined)
        emit = []
        for i, r in enumerate(self.rows):
            r.length += 1
            if feats is not None:
                r.feats.append(feats[i : i + 1])
                if len(r.feats) >= self.FEATS_CONSOLIDATE:
                    r.feats = [mx.concatenate(r.feats, axis=1)]
            if not r.emitted:
                emit.append((r, values[i]))
            r.emitted = False
        self.next = nxt
        self.steps += 1
        if self.steps % 256 == 0:
            mx.clear_cache()
        return emit

    def _token(self, row):
        return self.next[self.rows.index(row), 0].item()

    def settle(self, row):
        """Reads a row's in-flight sample: returns it (to be emitted) or None."""
        if row.emitted:
            return None
        row.emitted = True
        return self._token(row)

    def remove(self, rows, keep_state=()):
        """Takes rows out of the batch. Rows in keep_state get their single-request
        GenState back in row.state (processed: every token this batch has processed for
        them; pending: their unprocessed input, if it was emitted)."""
        index = {id(r): i for i, r in enumerate(self.rows)}
        for r in keep_state:
            i = index[id(r)]
            st = r.state
            st.cache = extract_row(self.cache, i, self.bot.kv_bits)
            st.length = r.length
            st.pending = [self._token(r)] if r.emitted else []
            st.hidden = mx.concatenate(r.feats, axis=1) if r.feats else None
            if st.hidden is not None:
                mx.eval(st.hidden)
            r.feats = []
        gone = {id(r) for r in rows}
        keep = [i for i, r in enumerate(self.rows) if id(r) not in gone]
        if not keep:
            self.rows, self.cache, self.next = [], None, None
        elif len(keep) < len(self.rows):
            sel = mx.array(keep)
            for c in self.cache:
                c.filter(sel)
            self.next = self.next[sel]
            self.rows = [self.rows[i] for i in keep]
        mx.clear_cache()


def new_row_state(state):
    """A GenState shell for a row (the batch owns the caches while it decodes)."""
    return GenState(cache=state.cache, length=state.length, pending=list(state.pending),
                    draft_cache=state.draft_cache, hidden=state.hidden)
