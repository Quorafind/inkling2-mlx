"""DSpark speculative decoding for Inkling 2 Mini.

The checkpoint ships a 3-layer DSpark draft (`mtp.safetensors`). It reads the
hidden states entering decoder *visits* 5, 23 and 41 (pre-loop layer 5, pass-2
layer 11, pass-3 layer 17) plus the final normalized state (visit 48), proposes
`block_size - 1 = 7` tokens per round with bidirectional block attention and a
recurrent Markov logit head, and the target verifies them in one forward.

The draft network itself is mlx-vlm's port of vLLM's Inkling 2 DSpark; only the
target binding differs (48 visits instead of 48 layers).

Acceptance: the target samples every verified position with the user's sampler,
and a draft token is accepted while it equals the target's own sample. Every
emitted token is therefore a target sample, so the output distribution is exactly
that of ordinary decoding at any temperature (greedy output is the same up to
floating-point ties).
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.speculative.drafters.inkling2_dspark.config import ModelConfig as _DraftConfig
from mlx_vlm.speculative.drafters.inkling2_dspark.dspark import MarkovHead
from mlx_vlm.speculative.drafters.inkling2_dspark.dspark import Model as _DSpark


def _as_bf16_weight(linear):
    if isinstance(linear, nn.QuantizedLinear):
        return mx.dequantize(
            linear.weight, linear.scales, linear.get("biases"),
            group_size=linear.group_size, bits=linear.bits, mode=linear.mode,
        ).astype(mx.bfloat16)
    return linear.weight


class _DraftTargetView:
    """What the draft needs from the target: its embedding and a vocabulary head.

    The draft's head is a 4-bit copy of the target's lm_head: the draft reads the
    whole vocabulary matrix every round, and its precision only affects which tokens
    get proposed. Verification always uses the target's own head.
    """

    def __init__(self, target, head):
        self.model = target.model
        self.lm_head = head
        self.sharding_group = None


class _QuantMarkovHead(MarkovHead):
    """Markov head whose folded (vocab x rank) projection is a 4-bit matmul."""

    def latent_step(self, state, previous, context):
        rank = state.shape[-1]
        inputs = mx.concatenate([state, self.markov_w1(previous)], axis=-1)
        raw = (
            inputs.astype(mx.float32) @ self.rnn_proj.weight[:, : 2 * rank].astype(mx.float32).T
            + context
        ).astype(state.dtype)
        gate, candidate, output = mx.split(raw, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(state.dtype)
        candidate = mx.tanh(candidate.astype(mx.float32)).astype(state.dtype)
        state = gate * state + (1 - gate) * candidate
        latent = mx.tanh(output.astype(mx.float32)).astype(state.dtype)
        return state, latent

    def step(self, state, previous, context):
        state, latent = self.latent_step(state, previous, context)
        return state, self.fold_q(latent)

    def gather_fold(self, ids):
        """The folded projection's rows for token ids (n,), dequantized: (n, rank), (n,)."""
        q = self.fold_q
        w = mx.dequantize(q.weight[ids], q.scales[ids], q.get("biases")[ids] if "biases" in q else None,
                          group_size=q.group_size, bits=q.bits, mode=q.mode)
        return w, q.bias[ids]


_chunk_topk = {}


def chunk_topk(logits, valid, chunks=512, k=8, threads=64):
    """Top-k of each of `chunks` contiguous vocabulary slices, per row: the screening
    shortlist (values float32, token ids int32), shapes (rows, chunks * k), ordered by
    slice. High-scoring tokens cluster at neighbouring ids, so k = 1 (a slice max)
    is a poor screen. logits: (rows, V) with the first `valid` entries real."""
    rows, stride = logits.shape
    key = (chunks, k, threads)
    if key not in _chunk_topk:
        _chunk_topk[key] = mx.fast.metal_kernel(
            name=f"inkling_dspark_chunk_top{k}_{chunks}_{threads}",
            input_names=["logits", "dims"],
            output_names=["vals", "ids"],
            source=f"""
    const int C = {chunks}, K = {k}, T = {threads};
    const int V = dims[0], S = dims[1];
    const int c = threadgroup_position_in_grid.x, r = threadgroup_position_in_grid.y;
    const int t = thread_position_in_threadgroup.x;
    const int lane = t % 32, sg = t / 32;
    const int chunk = (V + C - 1) / C;
    const int v0 = c * chunk, v1 = min(V, v0 + chunk);
    // Per-thread sorted top-K (descending; ties keep the lower id first).
    float lv[K];
    int li[K];
    for (int j = 0; j < K; ++j) {{ lv[j] = -INFINITY; li[j] = 0x7fffffff; }}
    for (int v = v0 + t; v < v1; v += T) {{
        float x = float(logits[(size_t)r * S + v]);
        if (x > lv[K - 1]) {{
            int p = K - 1;
            while (p > 0 && x > lv[p - 1]) {{ lv[p] = lv[p - 1]; li[p] = li[p - 1]; --p; }}
            lv[p] = x; li[p] = v;
        }}
    }}
    // K rounds of a threadgroup argmax over each thread's next unused entry.
    threadgroup float sv[T / 32];
    threadgroup int si[T / 32];
    int head = 0;
    for (int round = 0; round < K; ++round) {{
        float bv = head < K ? lv[head] : -INFINITY;
        int bid = head < K ? li[head] : 0x7fffffff;
        for (ushort off = 16; off > 0; off >>= 1) {{
            float ov = simd_shuffle_down(bv, off);
            int oi = simd_shuffle_down(bid, off);
            if (ov > bv || (ov == bv && oi < bid)) {{ bv = ov; bid = oi; }}
        }}
        if (lane == 0) {{ sv[sg] = bv; si[sg] = bid; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float wv = sv[0];
        int wid = si[0];
        for (int s = 1; s < T / 32; ++s)
            if (sv[s] > wv || (sv[s] == wv && si[s] < wid)) {{ wv = sv[s]; wid = si[s]; }}
        if (t == 0) {{
            const int o = (r * C + c) * K + round;
            vals[o] = wv;
            ids[o] = wid < V ? wid : v0;
        }}
        if (head < K && li[head] == wid) ++head;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}""",
        )
    vals, ids = _chunk_topk[key](
        inputs=[logits, mx.array([valid, stride], mx.int32)],
        grid=(chunks * threads, rows, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, chunks * k), (rows, chunks * k)],
        output_dtypes=[mx.float32, mx.int32],
    )
    return vals, ids


class DSparkDrafter(_DSpark):
    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        cfg = target.config
        if (
            cfg.hidden_size != self.config.hidden_size
            or cfg.vocab_size != self.config.vocab_size
            or cfg.num_cache_layers != self.config.target_layer_ids[-1]
        ):
            raise ValueError("this DSpark head does not match the Inkling 2 Mini target")

    def bind_quantized(self, target_model, bits=4, group_size=64):
        self.bind(target_model)
        target = target_model.language_model
        head = nn.Linear(target.config.hidden_size, target.config.vocab_size, bias=False)
        head.weight = _as_bf16_weight(target.lm_head)
        head = nn.QuantizedLinear.from_linear(head, group_size, bits)
        mh = self.markov_head
        fold = nn.Linear(mh._fold_weight.shape[1], mh._fold_weight.shape[0])
        fold.weight, fold.bias = mh._fold_weight, mh._fold_bias
        mh.fold_q = nn.QuantizedLinear.from_linear(fold, group_size, bits)
        mh._fold_weight = mh._fold_bias = None
        mh.__class__ = _QuantMarkovHead
        mx.eval(head.parameters(), mh.fold_q.parameters())
        self._target = _DraftTargetView(target, head)

    def reset(self, target_model=None):
        # Bound once at load; a fresh draft KV per request.
        self.accept_lens, self.draft_lens = [], []
        return self.make_cache()

    # Screened Markov steps (Strata-style): score only a shortlist of this many candidates
    # per position; 0 = the full vocabulary, as in mlx-vlm's draft_block (the default).
    # Measured (tests/check_screened_draft.py): 4096 candidates as 256 slices x top-16
    # match the full loop's drafts at 98.8 % of positions and save 1.0 ms of the 2.6 ms
    # Markov loop per round, about 1 % of a code round, which the 1.2 % of changed deep
    # drafts give back in acceptance. Off until a faster exact top-k exists.
    SCREEN = 0
    SCREEN_K = 16  # per vocabulary slice: 256 slices x top-16

    def draft_with_confidence(self, last_bonus, hidden, cache, block_size, screen=None):
        """Greedy drafts plus the draft's own probability of each drafted token,
        shapes (1, block_size - 1). The draft consumes `hidden` into its cache."""
        screen = self.SCREEN if screen is None else screen
        if not screen or not isinstance(self.markov_head, _QuantMarkovHead):
            conf = []

            def sampler(logits):
                lf = logits.astype(mx.float32)
                conf.append(mx.exp(mx.max(lf, axis=-1) - mx.logsumexp(lf, axis=-1)))
                return mx.argmax(lf, axis=-1)

            tokens = self.draft_block(last_bonus, hidden, cache, block_size, sampler, mx.int32)
            return tokens, mx.stack(conf, axis=-1)
        feats = self.draft_features(last_bonus, hidden, cache, block_size)
        return self.markov_screened(*feats, screen=screen, k=self.SCREEN_K)

    def draft_features(self, last_bonus, hidden, cache, block_size):
        """The draft layers and vocabulary head (mlx-vlm's draft_block up to the
        Markov loop): base logits (count, V), Markov context, anchor."""
        count = int(block_size) - 1
        if not 1 <= count < self.config.block_size:
            raise ValueError("Requested DSpark block exceeds the trained depth")
        anchor = mx.array([int(last_bonus)], mx.int32)
        embed = self._target.model.embed(anchor[:, None])
        masks = mx.broadcast_to(self.mask_embed.astype(embed.dtype),
                                (1, self.config.block_size - 1, embed.shape[-1]))
        x, residual = mx.concatenate([embed, masks], axis=1), None
        for i, (layer, entry) in enumerate(zip(self.layers, cache)):
            x, residual = layer(x, residual, self.mix(hidden, i, layer.dflare_norm), entry)
        x = self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(embed.dtype)[:, 1 : count + 1]
        base = self._target.lm_head(x).astype(x.dtype).reshape(count, -1)
        f_rnn = self.mix(hidden[:, -1:], -1, self.markov_head.dflare_norm)
        context = self.markov_head.context(x, f_rnn)[0]
        return base, context, anchor

    def _softcap(self, z):
        cap = self.config.final_logit_softcapping
        return cap * mx.tanh(z / cap) if cap else z

    def markov_full(self, base, context, anchor, return_logits=False):
        """Reference Markov loop over the whole vocabulary (same math as draft_block,
        including its BF16 rounding of the capped logits, which makes near-top ties
        common: the argmax then takes the lowest token id)."""
        mh, valid = self.markov_head, self.config.unpadded_vocab_size
        state = mx.zeros((1, self.config.markov_rank), base.dtype)
        previous, tokens, conf, rows = anchor, [], [], []
        for i in range(base.shape[0]):
            state, bias = mh.step(state, previous, context[None, i])
            logits = self._softcap((base[None, i] + bias).astype(mx.float32)).astype(base.dtype)
            lf = logits[:, :valid].astype(mx.float32)
            previous = mx.argmax(lf, axis=-1)
            tokens.append(previous)
            conf.append(mx.exp(mx.max(lf, axis=-1) - mx.logsumexp(lf, axis=-1)))
            rows.append(lf[0])
        out = mx.stack(tokens, axis=-1), mx.stack(conf, axis=-1)
        return out + (mx.stack(rows),) if return_logits else out

    def markov_screened(self, base, context, anchor, screen=4096, k=16):
        """Strata-style screened drafting: the state-dependent Markov bias is computed
        only for a shortlist per position instead of the full 200k vocabulary.

        The Markov bias is fold_W @ latent + fold_b. Its static part fold_b is as large
        as the dynamic part (std 2.0 vs 1.2-1.6) and is added before screening: the
        shortlist is the max of each of `screen` vocabulary slices of base + fold_b.
        Screening on the base logits alone misses the full loop's token at 10-30 % of
        positions 4-6; with fold_b its p90 rank is 18 and p99 about 430. Slice maxima
        cover it badly (76 % at 512 slices: top tokens sit at neighbouring ids), so
        the shortlist is the top-k of each slice. The draft token is the best
        shortlisted token; its confidence adds the static-logit mass outside the
        shortlist (whose dynamic bias is not computed)."""
        mh, valid = self.markov_head, self.config.unpadded_vocab_size
        count = base.shape[0]
        static = base + mh.fold_q.bias.astype(base.dtype)
        cand_v, cand_i = chunk_topk(static, valid, screen // k, k)
        w, _ = mh.gather_fold(cand_i.reshape(-1))
        w = w.reshape(count, cand_i.shape[1], -1)
        lse_all = mx.logsumexp(self._softcap(static[:, :valid].astype(mx.float32)), axis=-1)
        lse_cand = mx.logsumexp(self._softcap(cand_v), axis=-1)
        rest = lse_all + mx.log(mx.maximum(1.0 - mx.exp(lse_cand - lse_all), 1e-30))
        state = mx.zeros((1, self.config.markov_rank), base.dtype)
        previous, tokens, conf = anchor, [], []
        for i in range(count):
            state, latent = mh.latent_step(state, previous, context[None, i])
            dynamic = (w[i] @ latent.reshape(-1, 1)).reshape(-1)
            logits = self._softcap((cand_v[i].astype(base.dtype) + dynamic).astype(mx.float32))
            j = mx.argmax(logits)
            previous = cand_i[i][j].reshape(1)
            tokens.append(previous)
            conf.append(mx.exp(logits[j] - mx.logaddexp(mx.logsumexp(logits), rest[i])).reshape(1))
        return mx.stack(tokens, axis=-1), mx.stack(conf, axis=-1)


def load_drafter(model_dir, target_model, head_bits=4, layer_bits=8):
    """head_bits: the draft-side copy of lm_head and the folded Markov projection.
    layer_bits: the 3 draft layers' projections (8: 10.4 -> 8.9 ms per round with
    unchanged acceptance; 6 is no faster, 4 drafts slightly worse). The draft only
    proposes tokens, so neither changes the output."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    text = dict(config["text_config"])
    target = target_model.language_model.config
    # The draft's target_layer_ids index layer *visits*; the last one is the final state.
    text["num_hidden_layers"] = target.num_cache_layers
    text["mtp_config"] = config["mtp_config"]
    drafter = DSparkDrafter(_DraftConfig(text_config=text))
    weights = mx.load(str(model_dir / "mtp.safetensors"))
    drafter.load_weights(list(weights.items()), strict=True)
    if layer_bits and layer_bits < 16:
        for layer in drafter.layers:
            nn.quantize(layer, group_size=64, bits=layer_bits,
                        class_predicate=lambda _, m: isinstance(m, nn.Linear))
    mx.eval(drafter.parameters())
    if head_bits:
        drafter.bind_quantized(target_model, bits=head_bits)
    else:
        drafter.bind(target_model)  # folds the Markov head through the target's lm_head
    return drafter


def greedy(logits):
    return mx.argmax(logits, axis=-1)


class BlockPolicy:
    """Pick the verify block size that maximizes expected tokens per second.

    Draft acceptance is modelled as geometric with per-position probability p,
    estimated online with exponential forgetting. Round costs are tracked online
    per block size; unmeasured sizes are scaled by the measured ones. Block size 1
    means plain decoding, chosen whenever speculation would not pay off; plain
    segments end with a small probe round so p keeps tracking the text (code and
    lists draft well, free prose does not).

    (A per-depth conditional-acceptance model was tried and did not do better.)
    """

    PRIOR_MS = {1: 54.0, 2: 80.0, 3: 92.0, 4: 103.0, 5: 112.0, 6: 120.0, 7: 129.0, 8: 138.0}

    def __init__(self, max_block=8, fixed=False, decay=0.85, plain_segment=24, probe=3):
        self.max_block = max_block
        self.fixed = fixed
        self.decay = decay
        self.plain_segment = plain_segment
        self.probe = min(probe, max_block)
        self.prior = {k: v / 1e3 for k, v in self.PRIOR_MS.items() if k <= max_block}
        self.cost = dict(self.prior)
        self.seen = set()
        self.succ, self.fail = 2.0, 1.0  # prior p = 2/3
        self.after_plain = False

    @property
    def p(self):
        return self.succ / (self.succ + self.fail)

    def expected_tokens(self, bs):
        p = self.p
        return sum(p**i for i in range(bs))

    def choose(self):
        if self.fixed:
            return self.max_block
        if self.after_plain:
            self.after_plain = False
            return self.probe
        return max(self.cost, key=lambda bs: self.expected_tokens(bs) / self.cost[bs])

    def _update_cost(self, bs, seconds):
        self.cost[bs] = seconds if bs not in self.seen else 0.7 * self.cost[bs] + 0.3 * seconds
        self.seen.add(bs)
        scale = sum(self.cost[k] / self.prior[k] for k in self.seen) / len(self.seen)
        for k in self.cost:
            if k not in self.seen:
                self.cost[k] = self.prior[k] * scale

    def observe(self, bs, accepted, seconds):
        self.succ = self.decay * self.succ + accepted
        self.fail = self.decay * self.fail + (1 if accepted < bs - 1 else 0)
        self._update_cost(bs, seconds)

    def observe_plain(self, seconds_per_token):
        self._update_cost(1, seconds_per_token)
        self.after_plain = True


class ConfidencePolicy:
    """Per-round verify length from the draft's confidence (DSpark's prefix scheduler).

    Every round drafts the full block (the draft's cost barely depends on how many
    positions are read out), then verifies the prefix L that maximizes
        E[tokens | L] - cost(L) / plain_cost,
    i.e. a round runs only if it beats plain decoding over the same time. The draft's
    probability of each drafted token is mapped to an acceptance probability through
    a binned table that is updated online. If no L >= 2 pays, the round falls back to
    a pipelined plain segment; segments double after each fallback (24, then 48) and
    reset once a verify round accepts something, so free prose costs one draft per
    segment instead of a losing verify round. Longer caps (96, 256) gain nothing on
    prose and miss code that follows a prose stretch.

    The checkpoint's draft has no trained confidence head; its token probability
    separates accepted from rejected positions with AUC 0.79-0.92 (prose / code).
    """

    EDGES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995)
    # Conditional acceptance per bin, 3,519 drafted positions over English / Chinese
    # prose, code and thinking, T = 0 and 1 (6-bit dense, block 8).
    PRIOR_ACCEPT = (0.05, 0.08, 0.14, 0.19, 0.24, 0.33, 0.39, 0.44, 0.59, 0.69, 0.83, 0.94)
    # Verify round cost after the draft (forward over L tokens, sampling, commit) and a
    # pipelined plain step, both at short context on the M6; replaced by measurements.
    PRIOR_VERIFY_MS = {2: 55.0, 3: 65.0, 4: 78.0, 5: 89.0, 6: 93.0, 7: 100.0, 8: 108.0}
    PRIOR_PLAIN_MS = 33.0

    def __init__(self, max_block=8, segment=24, max_segment=48, strength=8.0, window=100):
        self.max_block = max_block
        self.base_segment, self.max_segment = segment, max_segment
        self.segment = segment
        self.n = [strength] * len(self.PRIOR_ACCEPT)
        self.a = [strength * p for p in self.PRIOR_ACCEPT]
        self.prior = {k: v / 1e3 for k, v in self.PRIOR_VERIFY_MS.items() if k <= max_block}
        self.cost = dict(self.prior)
        self.seen = set()
        self.plain_cost = self.PRIOR_PLAIN_MS / 1e3
        self.window = window
        self.fallbacks = 0

    def choose(self):
        return self.max_block  # always draft; the length is cut after drafting

    def _bin(self, p):
        for i, edge in enumerate(self.EDGES):
            if p < edge:
                return i
        return len(self.EDGES)

    def accept_prob(self, p):
        b = self._bin(p)
        return self.a[b] / self.n[b]

    def choose_length(self, conf, limit=None):
        """conf: the draft's probabilities of its tokens. Returns 1 for plain decoding."""
        limit = min(limit or self.max_block, self.max_block, len(conf) + 1)
        best, best_value = 1, 0.0
        survive, expected = 1.0, 1.0
        for L in range(2, limit + 1):
            survive *= self.accept_prob(conf[L - 2])
            expected += survive
            value = expected - self.cost[L] / self.plain_cost
            if value > best_value:
                best, best_value = L, value
        if best == 1:
            self.fallbacks += 1
        return best

    def next_segment(self):
        """Length of the plain segment after a fallback; the next one doubles."""
        seg = self.segment
        self.segment = min(2 * seg, self.max_segment)
        return seg

    def _update_cost(self, bs, seconds):
        self.cost[bs] = seconds if bs not in self.seen else 0.7 * self.cost[bs] + 0.3 * seconds
        self.seen.add(bs)
        scale = sum(self.cost[k] / self.prior[k] for k in self.seen) / len(self.seen)
        for k in self.cost:
            if k not in self.seen:
                self.cost[k] = self.prior[k] * scale

    def observe(self, bs, accepted, seconds, conf=()):
        """seconds: the verify round after the draft."""
        for i in range(min(accepted + 1, bs - 1, len(conf))):
            b = self._bin(conf[i])
            self.n[b] += 1
            self.a[b] += i < accepted
            if self.n[b] > self.window:  # keep adapting (code accepts more per confidence)
                self.n[b] /= 2
                self.a[b] /= 2
        if accepted:
            self.segment = self.base_segment
        self._update_cost(bs, seconds)

    def observe_plain(self, seconds_per_token):
        self.plain_cost = 0.7 * self.plain_cost + 0.3 * seconds_per_token
