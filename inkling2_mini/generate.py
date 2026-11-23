"""Prompting, sampling and streaming generation for Inkling 2 Mini."""

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import mlx.core as mx
from mlx.utils import tree_flatten

from .loader import load

# Adaptive DSpark verify length: "confidence" (speculative.ConfidencePolicy, per-round
# length from the draft's token probabilities) or "geometric" (the earlier BlockPolicy,
# one global acceptance rate). Kept switchable for tests/bench_spec_policy.py.
SPEC_POLICY = "confidence"

MESSAGE_MODEL = "<|message_model|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINKING = "<|content_thinking|>"
CONTENT_TOOL_JSON = "<|content_invoke_tool_json|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"


@dataclass
class Event:
    kind: str  # "thinking" | "text" | "tool" | "done"
    text: str = ""
    stats: Optional[dict] = None


@dataclass
class GenState:
    """Everything needed to continue a conversation without re-prefilling it."""

    cache: list
    length: int = 0  # tokens processed into `cache`
    pending: list = field(default_factory=list)  # emitted, not yet processed
    draft_cache: Optional[list] = None
    hidden: Optional[mx.array] = None  # draft features of processed tokens the draft has not seen

    def clone(self):
        """An independent copy: continuing either one leaves the other untouched.
        MLX updates arrays functionally, so a new array object per buffer suffices
        (the first in-place update of a shared buffer copies it)."""
        return GenState(cache=_clone(self.cache), length=self.length, pending=list(self.pending),
                        draft_cache=_clone(self.draft_cache), hidden=self.hidden)

    def nbytes(self):
        return sum(a.nbytes for a in _arrays([self.cache, self.draft_cache, self.hidden]))


def _clone(obj):
    if isinstance(obj, mx.array):
        return mx.array(obj)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        new = copy.copy(obj)
        for k, v in vars(obj).items():
            setattr(new, k, _clone(v))
        return new
    return obj


def _arrays(obj, seen=None):
    seen = set() if seen is None else seen
    if isinstance(obj, mx.array):
        if id(obj) not in seen:
            seen.add(id(obj))
            yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _arrays(x, seen)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _arrays(x, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        for x in vars(obj).values():
            yield from _arrays(x, seen)


@dataclass
class Reply:
    thinking: str = ""
    text: str = ""
    tool: str = ""
    tokens: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def make_sampler(temperature=1.0, top_p=1.0, top_k=0):
    """Returns sample(logits, key=None) -> token ids; `key` gives the row its own random
    stream (per-request seeds inside a batch)."""

    def sample(logits, key=None):
        logits = logits.reshape(-1, logits.shape[-1])
        if temperature <= 0:
            return mx.argmax(logits, axis=-1)
        logits = logits / temperature
        if top_k and top_k > 0:
            kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
            logits = mx.where(logits < kth, -mx.inf, logits)
        if top_p < 1.0:
            order = mx.argsort(-logits, axis=-1)
            sorted_logits = mx.take_along_axis(logits, order, axis=-1)
            probs = mx.softmax(sorted_logits, axis=-1)
            # keep the smallest prefix whose mass reaches top_p (always keep the first)
            keep = (mx.cumsum(probs, axis=-1) - probs) < top_p
            sorted_logits = mx.where(keep, sorted_logits, -mx.inf)
            inverse = mx.argsort(order, axis=-1)
            logits = mx.take_along_axis(sorted_logits, inverse, axis=-1)
        return mx.random.categorical(logits, key=key)

    return sample


def _cache_arrays(cache):
    return [v for _, v in tree_flatten([c.state for c in cache]) if isinstance(v, mx.array)]


class Inkling2Mini:
    # Pass counts a variable-loop checkpoint was trained with (TML-d24-varloop-research).
    VARLOOP_PASSES = range(1, 6)

    def __init__(self, model_dir, verbose=True, speculative=True, dense_bits=6, kv_bits=8, passes=None):
        """dense_bits: 6 (default) quantizes attention projections + lm_head to affine 6-bit
        (KL at the model's own BF16 noise, fewest bytes per token); 8 is near-lossless;
        16 / None keeps the checkpoint's native BF16 for them.
        passes: loop passes for new states (default: the checkpoint's, 3). Variable-loop
        checkpoints take 1 to 5; Inkling 2 Mini is only trained for 3."""
        from mlx_vlm.models.inkling2.processing_inkling2 import Inkling2Processor

        self.model_dir = Path(model_dir)
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        mx.set_wired_limit(info["max_recommended_working_set_size"])
        self.model = load(self.model_dir, verbose=verbose, dense_bits=dense_bits)
        self.text_config = self.model.language_model.config
        self.passes = self.check_passes(passes or self.text_config.num_passes)
        self.text_config.loop_config["num_passes"] = self.passes  # default for cache-less calls
        if verbose and (self.text_config.is_varloop or passes):
            kind = "variable-loop checkpoint (1-5 passes), " if self.text_config.is_varloop else ""
            print(f"[inkling2-mini] {kind}{self.passes} loop pass{'es' if self.passes > 1 else ''}")
        # Full-attention KV caches: 8-bit (default; fused decode kernel, half the memory) or
        # None / 16 for BF16. Batched decoding always uses BF16 caches.
        self.kv_bits = kv_bits if kv_bits and kv_bits < 16 else None
        self.drafter = None
        if speculative and (self.model_dir / "mtp.safetensors").exists():
            from .speculative import load_drafter

            self.drafter = load_drafter(self.model_dir, self.model)
        self.processor = Inkling2Processor.from_pretrained(str(self.model_dir))
        self.tokenizer = self.processor.tokenizer
        tok = self.tokenizer.convert_tokens_to_ids
        self.ids = {
            name: tok(name)
            for name in (MESSAGE_MODEL, CONTENT_TEXT, CONTENT_THINKING, CONTENT_TOOL_JSON, END_MESSAGE, END_SAMPLING)
        }
        self.eos_id = self.model.config.eos_token_id

    def check_passes(self, passes):
        passes = int(passes)
        cfg = self.text_config
        if cfg.is_varloop and passes not in self.VARLOOP_PASSES:
            raise ValueError(f"this variable-loop checkpoint was trained with 1 to 5 passes, not {passes}")
        if passes < 1 or passes > 8:
            raise ValueError(f"passes must be 1 to 8, not {passes}")
        return passes

    def state_passes(self, state):
        return self.text_config.passes_for(len(state.cache))

    def capture_ids(self, state, drafter=None):
        """Draft feature visits for this state's pass count (None without a drafter)."""
        drafter = self.drafter if drafter is None else drafter
        if drafter is None:
            return None
        return drafter.capture_ids(self.text_config, self.state_passes(state))

    # ---------------------------------------------------------------- prompt
    def render(self, messages, reasoning_effort=None, tools=None) -> str:
        kwargs = {}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False, **kwargs
        )

    def encode(self, messages, images=None, reasoning_effort=None, tools=None):
        prompt = self.render(messages, reasoning_effort, tools)
        inputs = self.processor(images=images or None, text=prompt)
        return inputs["input_ids"], inputs.get("pixel_values"), inputs.get("image_grid_thw")

    # ------------------------------------------------------------ generation
    # Each chunk's relative-position mask is (heads x chunk x context) BF16; at long
    # contexts the chunk shrinks so that mask stays under this budget.
    MASK_BUDGET_BYTES = 1 << 30

    def _chunk(self, step, ctx):
        from . import band_attention

        if band_attention.available():
            return step  # full-attention layers build no (queries x context) mask
        heads = self.model.language_model.config.num_attention_heads
        return max(64, min(step, self.MASK_BUDGET_BYTES // (2 * heads * (ctx + step))))

    def _prefill(self, h, cache, step, capture, ctx=0):
        """Returns last-position logits and, with capture, the draft features of all
        positions. ctx: tokens already in the cache (sizes the chunks)."""
        it = self.prefill_chunks(h, cache, step, capture, ctx)
        while True:
            try:
                next(it)
            except StopIteration as done:
                return done.value

    def prefill_chunks(self, h, cache, step, capture, ctx=0):
        """_prefill as a generator: yields the number of tokens processed after every
        chunk but the last, so a scheduler can run other work in between; returns
        (last-position logits, draft features or None)."""
        lm = self.model.language_model
        feats, start, n = [], 0, h.shape[1]
        while True:
            size = self._chunk(step, ctx + start)
            chunk = h[:, start : start + size]
            out = lm.model(cache=cache, input_embeddings=chunk, capture=capture)
            if capture is not None:
                out, caps = out
                feats.append(mx.concatenate(caps, axis=-1))
            start += chunk.shape[1]
            if start >= n:
                break
            mx.eval(out, feats, _cache_arrays(cache))
            mx.clear_cache()
            yield start
        logits = lm.logits(out[:, -1:])
        return logits, (mx.concatenate(feats, axis=1) if capture is not None else None)

    def prefill(self, input_ids, state, prefill_step=1024):
        """Process text tokens into `state` without sampling (prefix snapshots). The
        state then continues with generate_tokens(more_ids, state=state)."""
        capture = self.capture_ids(state)
        h = self.model.embed(input_ids)
        if state.pending:
            h = mx.concatenate([self.model.embed(mx.array([state.pending])), h], axis=1)
        _, fresh = self._prefill(h, state.cache, prefill_step, capture, ctx=state.length)
        state.length += h.shape[1]
        state.pending = []
        if capture is not None:
            state.hidden = fresh if state.hidden is None else mx.concatenate([state.hidden, fresh], axis=1)
            mx.eval(state.hidden)
        mx.eval(_cache_arrays(state.cache))
        return state

    def new_state(self, kv_bits=None, passes=None):
        """kv_bits: None = the bot's default (16 = BF16); passes: None = the bot's default."""
        kv_bits = kv_bits if kv_bits is not None else self.kv_bits
        passes = self.check_passes(passes) if passes else self.passes
        return GenState(cache=self.model.make_cache(kv_bits=kv_bits if kv_bits and kv_bits < 16 else None,
                                                    passes=passes))

    def generate_tokens(
        self,
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
        max_tokens=2048,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        prefill_step=1024,
        seed=None,
        speculative=True,
        block_size=None,
        state=None,
        should_pause=None,
    ) -> Iterator[tuple]:
        """Yields (token_id, stats) pairs; stats is filled in on the last token.

        should_pause: optional callable, polled between tokens. When it returns True the
        generator stops early without final stats, leaving `state` consistent (every
        emitted token but the last processed, the last in `pending`), so the
        conversation can be continued later or in another decoding mode (the server's
        continuous batching).

        state: a GenState to continue (multi-turn KV reuse). Its caches hold every
        token processed so far; `pending` tokens were emitted but not yet processed
        and are prefilled first. The target never processes past the last emitted
        token (rings and conv states cannot be rewound), and on return the state
        describes exactly the emitted sequence."""
        if seed is not None:
            mx.random.seed(seed)
        lm = self.model.language_model
        sample = make_sampler(temperature, top_p, top_k)
        state = state if state is not None else self.new_state()
        cache = state.cache
        drafter = self.drafter if speculative else None
        capture = self.capture_ids(state, drafter) if drafter is not None else None
        pause = should_pause or (lambda: False)

        t0 = time.perf_counter()
        h = self.model.embed(input_ids, pixel_values, image_grid_thw)
        if state.pending:
            h = mx.concatenate([self.model.embed(mx.array([state.pending])), h], axis=1)
        n_prompt = h.shape[1]
        logits, fresh = self._prefill(h, cache, prefill_step, capture, ctx=state.length)
        state.length += n_prompt
        state.pending = []
        if capture is not None:
            hidden = fresh if state.hidden is None else mx.concatenate([state.hidden, fresh], axis=1)
        else:
            hidden = None
            state.hidden = None  # plain decoding does not track draft features
        y = sample(logits)
        mx.async_eval(y)
        y.item()
        t_prefill = time.perf_counter() - t0
        t1 = time.perf_counter()
        spec = dict(rounds=0, drafted=0, accepted=0)

        def finish(n):
            dt = time.perf_counter() - t1
            stats = dict(
                prompt_tokens=n_prompt,
                context_tokens=state.length + len(state.pending),
                prefill_s=t_prefill,
                prefill_tps=n_prompt / t_prefill,
                gen_tokens=n,
                gen_tps=n / dt if dt > 0 else 0.0,
                peak_gb=mx.get_peak_memory() / 1e9,
            )
            if drafter is not None and spec["rounds"]:
                stats["tokens_per_round"] = (spec["accepted"] + spec["rounds"]) / spec["rounds"]
                stats["acceptance"] = spec["accepted"] / max(spec["drafted"], 1)
            if drafter is not None and getattr(policy, "fallbacks", 0):
                stats["plain_fallbacks"] = policy.fallbacks
            return stats

        if drafter is None:
            n = 0
            while True:
                n += 1
                done = n >= max_tokens
                paused = not done and pause()
                if not done and not paused:  # y is processed now, before we know whether it is EOS
                    next_y = sample(lm(y.reshape(1, 1), cache=cache))
                    mx.async_eval(next_y)
                    state.length += 1
                token = y.item()
                if done or paused:
                    state.pending = [token]
                done = done or token == self.eos_id
                yield token, (finish(n) if done else None)
                if done or paused:
                    return
                if n % 256 == 0:
                    mx.clear_cache()
                y = next_y

        # ---- DSpark speculative decoding: draft -> verify -> accept -> commit
        from mlx_vlm.speculative.cache_state import start_speculative_cache

        from .speculative import BlockPolicy, ConfidencePolicy, greedy

        max_block = min(block_size or drafter.config.block_size, drafter.config.block_size)
        gated = block_size is None and SPEC_POLICY == "confidence"
        policy = ConfidencePolicy(max_block) if gated else BlockPolicy(max_block, fixed=block_size is not None)
        if state.draft_cache is None:
            state.draft_cache = drafter.reset()
        draft_cache = state.draft_cache
        b = y.item()
        n = 1
        state.pending, state.hidden = [b], hidden
        if b == self.eos_id or n >= max_tokens:
            yield b, finish(n)
            return
        yield b, None

        while True:
            if pause():  # between rounds the state is consistent: b is pending
                return
            bs = min(policy.choose(), max_tokens - n + 1)
            t_round = time.perf_counter()
            conf = ()
            segment = policy.plain_segment if bs <= 1 else None
            if bs > 1 and gated:
                # Draft the full block, then verify only the prefix the draft's own
                # confidence says will pay (or fall back to plain decoding).
                drafts, conf = drafter.draft_with_confidence(b, hidden, draft_cache, max_block)
                mx.eval(drafts, conf)
                hidden = state.hidden = None  # consumed by the draft
                conf = conf.reshape(-1).tolist()
                policy.context = state.length
                bs = policy.choose_length(conf, limit=max_tokens - n + 1)
                drafts = drafts[:, : bs - 1]
                if bs <= 1:
                    segment = policy.next_segment()
                t_round = time.perf_counter()
            elif bs > 1:
                drafts = drafter.draft_block(b, hidden, draft_cache, bs, greedy, mx.int32)
            if bs <= 1:
                # Plain pipelined decoding; keep collecting draft features meanwhile.
                k = min(segment, max_tokens - n)
                feats = [] if hidden is None else [hidden]
                out, caps = lm.model(mx.array([[b]], mx.int32), cache=cache, capture=capture)
                state.length += 1
                nxt = sample(lm.logits(out)).reshape(1, 1)
                feats.append(mx.concatenate(caps, axis=-1))
                mx.async_eval(nxt, feats[-1])
                i = 0
                while i < k:
                    if i < k - 1 and pause():
                        k = i + 1  # end the segment at this token: it stays pending
                    if i < k - 1:  # process the token before reading it (pipelining)
                        out, caps = lm.model(nxt, cache=cache, capture=capture)
                        state.length += 1
                        nxt2 = sample(lm.logits(out)).reshape(1, 1)
                        f = mx.concatenate(caps, axis=-1)
                        mx.async_eval(nxt2, f)
                        feats.append(f)
                    token = nxt.item()
                    n += 1
                    done = token == self.eos_id or n >= max_tokens
                    if done:
                        state.pending = [] if i < k - 1 else [token]
                        state.hidden = mx.concatenate(feats, axis=1)
                        yield token, finish(n)
                        return
                    yield token, None
                    if i < k - 1:
                        nxt = nxt2
                    i += 1
                b = token
                hidden = mx.concatenate(feats, axis=1)
                state.pending, state.hidden = [b], hidden
                policy.observe_plain((time.perf_counter() - t_round) / k)
                continue

            verify = mx.concatenate([mx.array([[b]], mx.int32), drafts.reshape(1, -1)], axis=1)
            tx = start_speculative_cache(cache, bs)
            try:
                out, caps = lm.model(verify, cache=cache, capture=capture)
                target = sample(lm.logits(out)).reshape(-1)
                feats = mx.concatenate(caps, axis=-1)
                mx.async_eval(target, feats)
                d, t = drafts.reshape(-1).tolist(), target.tolist()
                acc = 0
                while acc < bs - 1 and d[acc] == t[acc]:
                    acc += 1
                # Emit t[:take]; never keep verified positions past the last emitted token.
                take = acc + 1
                for j in range(take):
                    if t[j] == self.eos_id or n + j + 1 >= max_tokens:
                        take = j + 1
                        break
                tx.commit([take])
            except BaseException:
                tx.abort()
                raise
            spec["rounds"] += 1
            spec["drafted"] += bs - 1
            spec["accepted"] += acc
            state.length += take  # b and t[: take - 1]
            hidden = feats[:, :take]
            state.pending, state.hidden = [t[take - 1]], hidden
            for j, token in enumerate(t[:take]):
                n += 1
                done = j == take - 1 and (token == self.eos_id or n >= max_tokens)
                yield token, (finish(n) if done else None)
                if done:
                    return
            b = t[take - 1]
            if gated:
                policy.observe(bs, acc, time.perf_counter() - t_round, conf)
            else:
                policy.observe(bs, acc, time.perf_counter() - t_round)
            if spec["rounds"] % 64 == 0:
                mx.clear_cache()

    def _segmenter(self):
        return _Segmenter(self.tokenizer, self.ids)

    def stream(self, messages, images=None, reasoning_effort=None, tools=None, **gen_kwargs) -> Iterator[Event]:
        """Streams Events, split into thinking / answer / tool-call segments."""
        input_ids, pixels, grid = self.encode(messages, images, reasoning_effort, tools)
        seg = self._segmenter()
        for token, stats in self.generate_tokens(input_ids, pixels, grid, **gen_kwargs):
            yield from seg.feed(token)
            if stats is not None:
                yield Event("done", stats=stats)

    def chat(self, messages, **kwargs) -> Reply:
        reply = Reply()
        for ev in self.stream(messages, **kwargs):
            if ev.kind == "done":
                reply.stats = ev.stats
            else:
                setattr(reply, ev.kind, getattr(reply, ev.kind) + ev.text)
        return reply

    # -------------------------------------------------------------- batching
    def generate_batch_tokens(
        self,
        batch_ids,
        max_tokens=2048,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        prefill_step=1024,
        seed=None,
    ):
        """Decode several prompts together. Each prompt is prefilled on its own
        (identical numerics to single-request generation), then the 48 per-visit
        caches are merged into left-padded batch caches and all rows decode in
        lockstep: one weight read serves every row. Finished rows are filtered out
        one step late, which keeps CPU graph building overlapped with GPU work.

        batch_ids: list of (1, L_i) token arrays. Returns (tokens per row, stats)."""
        from mlx_vlm.models.cache import CacheList

        if seed is not None:
            mx.random.seed(seed)
        lm = self.model.language_model
        sample = make_sampler(temperature, top_p, top_k)
        B = len(batch_ids)
        outputs = [[] for _ in range(B)]

        t0 = time.perf_counter()
        caches, first = [], []
        for ids in batch_ids:
            cache = self.model.make_cache(passes=self.passes)  # BF16 KV: batch merging needs plain caches
            logits, _ = self._prefill(self.model.embed(ids), cache, prefill_step, None)
            first.append(sample(logits).reshape(-1))
            mx.eval(first[-1], _cache_arrays(cache))
            caches.append(cache)
        t_prefill = time.perf_counter() - t0

        t1 = time.perf_counter()
        rows = []
        for r, tok in enumerate(mx.concatenate(first).tolist()):
            outputs[r].append(tok)
            if tok != self.eos_id and max_tokens > 1:
                rows.append(r)
        cache = [
            CacheList.merge([caches[r][i] for r in rows]) for i in range(len(caches[0]))
        ] if rows else []
        del caches
        steps = 0
        if rows:
            y = mx.array([[outputs[r][0]] for r in rows], dtype=mx.int32)
            nxt = sample(lm(y, cache=cache)).reshape(-1, 1)
            mx.async_eval(nxt)
            while True:
                following = sample(lm(nxt, cache=cache)).reshape(-1, 1)
                mx.async_eval(following)
                steps += 1
                keep = []
                for i, (r, tok) in enumerate(zip(rows, nxt.reshape(-1).tolist())):
                    outputs[r].append(tok)
                    if tok != self.eos_id and len(outputs[r]) < max_tokens:
                        keep.append(i)
                if not keep:
                    break
                if len(keep) < len(rows):
                    index = mx.array(keep)
                    for c in cache:
                        c.filter(index)
                    following = following[index]
                    rows = [rows[i] for i in keep]
                nxt = following
                if steps % 256 == 0:
                    mx.clear_cache()
        dt = time.perf_counter() - t1
        generated = sum(len(o) for o in outputs)
        stats = dict(
            batch=B,
            prompt_tokens=sum(ids.shape[1] for ids in batch_ids),
            prefill_s=t_prefill,
            gen_tokens=generated,
            gen_s=dt,
            gen_tps=generated / dt if dt > 0 else 0.0,
            steps=steps + 1,
            peak_gb=mx.get_peak_memory() / 1e9,
        )
        return outputs, stats

    def chat_batch(self, conversations, reasoning_effort=None, **gen_kwargs):
        """conversations: list of message lists. Returns (list of Reply, stats)."""
        batch_ids = [self.encode(m, None, reasoning_effort)[0] for m in conversations]
        outputs, stats = self.generate_batch_tokens(batch_ids, **gen_kwargs)
        replies = []
        for tokens in outputs:
            reply, seg = Reply(tokens=tokens), self._segmenter()
            for tok in tokens:
                for ev in seg.feed(tok):
                    setattr(reply, ev.kind, getattr(reply, ev.kind) + ev.text)
            replies.append(reply)
        return replies, stats


class _Segmenter:
    """Turns generated token ids into thinking / text / tool Events."""

    def __init__(self, tokenizer, ids):
        self.tokenizer = tokenizer
        self.modes = {ids[CONTENT_THINKING]: "thinking", ids[CONTENT_TEXT]: "text", ids[CONTENT_TOOL_JSON]: "tool"}
        self.structural = {ids[MESSAGE_MODEL], ids[END_MESSAGE], ids[END_SAMPLING]}
        self.mode, self.segment, self.emitted = None, [], 0

    def feed(self, token):
        if token in self.modes:
            self.mode, self.segment, self.emitted = self.modes[token], [], 0
        elif token in self.structural:
            self.mode, self.segment, self.emitted = None, [], 0
        elif self.mode is not None:
            self.segment.append(token)
            text = self.tokenizer.decode(self.segment)
            if not text.endswith("\ufffd"):  # hold back partial UTF-8 sequences
                if len(text) > self.emitted:
                    yield Event(self.mode, text[self.emitted :])
                self.emitted = len(text)


def user_message(text, images=None):
    if not images:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [*({"type": "image"} for _ in images), {"type": "text", "text": text}]}


class Session:
    """A multi-turn conversation that keeps its KV caches between turns.

    Each turn prefills only the new user message: the chat template renders
    message by message, so render(history + [new]) starts with render(history)
    and the difference is exactly the new message. Everything already processed,
    including the model's own replies (as generated, thinking included), stays in
    the 48 per-visit caches, so turn latency no longer grows with the history.
    """

    def __init__(self, bot, system=None, reasoning_effort=None, tools=None, kv_bits=None):
        self.bot, self.effort, self.tools = bot, reasoning_effort, tools
        self.kv_bits = kv_bits if kv_bits is not None else bot.kv_bits
        self.system = system
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.system}] if self.system else []
        self.state = None
        self._rendered = None

    @property
    def context_tokens(self):
        return 0 if self.state is None else self.state.length + len(self.state.pending)

    def stream(self, content, images=None, **gen_kwargs) -> Iterator[Event]:
        bot = self.bot
        convo = self.messages + [user_message(content, images)]
        text = bot.render(convo, self.effort, self.tools)
        if self.state is not None and not text.startswith(self._rendered):
            self.state = None  # template not append-only for this history: start over
        if self.state is None:
            self.state, suffix = bot.new_state(self.kv_bits), text
        else:
            suffix = text[len(self._rendered) :]
        inputs = bot.processor(images=images or None, text=suffix)
        seg, reply = bot._segmenter(), Reply()
        for token, stats in bot.generate_tokens(
            inputs["input_ids"], inputs.get("pixel_values"), inputs.get("image_grid_thw"),
            state=self.state, **gen_kwargs,
        ):
            reply.tokens.append(token)
            for ev in seg.feed(token):
                setattr(reply, ev.kind, getattr(reply, ev.kind) + ev.text)
                yield ev
            if stats is not None:
                reply.stats = stats
                yield Event("done", stats=stats)
        if reply.tokens and reply.tokens[-1] != bot.eos_id:
            # Truncated reply: close it so the next turn follows a well-formed message.
            self.state.pending += [bot.ids[END_MESSAGE], bot.eos_id]
        assistant = {"role": "assistant", "content": reply.text}
        if reply.thinking:
            assistant["reasoning_content"] = reply.thinking
        self.messages = convo + [assistant]
        self._rendered = bot.render(self.messages, self.effort, self.tools)
        self.last_reply = reply

    def send(self, content, images=None, **gen_kwargs) -> Reply:
        for _ in self.stream(content, images, **gen_kwargs):
            pass
        return self.last_reply

    # ------------------------------------------------------------ persistence
    def save(self, path):
        """Write the caches and conversation to one .safetensors file."""
        import json

        from .caches import QuantSinkKVCache, SinkKVCache, SinkRingKVCache

        st = self.state
        arrays, meta = {}, {"length": st.length, "pending": st.pending, "caches": [], "draft": []}

        def put(name, a):
            if a is not None:
                arrays[name] = a

        for i, c in enumerate(st.cache):
            kv, conv = c[0], c[1]
            entry = {"type": type(kv).__name__, "offset": kv.offset}
            if isinstance(kv, QuantSinkKVCache):
                entry.update(bits=kv.bits, group_size=kv.group_size)
                for j, a in enumerate(kv.qk or []):
                    put(f"c{i}.qk{j}", a[..., : kv.offset, :])
                for j, a in enumerate(kv.qv or []):
                    put(f"c{i}.qv{j}", a[..., : kv.offset, :])
            elif isinstance(kv, SinkRingKVCache):
                entry.update(idx=kv._idx, max_size=kv.max_size)
                if kv.keys is not None:
                    # Until the ring wraps, entries sit chronologically in [0, offset).
                    n = kv.keys.shape[2] if kv.offset >= kv.max_size else kv.offset
                    put(f"c{i}.k", kv.keys[..., :n, :])
                    put(f"c{i}.v", kv.values[..., :n, :])
            elif isinstance(kv, SinkKVCache):
                if kv.keys is not None:
                    put(f"c{i}.k", kv.keys[..., : kv.offset, :])
                    put(f"c{i}.v", kv.values[..., : kv.offset, :])
            else:
                raise TypeError(f"cannot save {type(kv).__name__}")
            entry["conv"] = [j for j in range(4) if conv[j] is not None]
            for j in entry["conv"]:
                put(f"c{i}.conv{j}", conv[j])
            meta["caches"].append(entry)
        for i, kv in enumerate(st.draft_cache or []):
            meta["draft"].append({"offset": kv.offset, "idx": getattr(kv, "_idx", 0), "rotating": hasattr(kv, "max_size")})
            if kv.keys is not None:
                put(f"d{i}.k", kv.keys)
                put(f"d{i}.v", kv.values)
        put("hidden", st.hidden)
        meta.update(messages=self.messages, rendered=self._rendered, effort=self.effort, kv_bits=self.kv_bits)
        mx.save_safetensors(str(path), arrays, metadata={"inkling2_session": json.dumps(meta, ensure_ascii=False)})

    @classmethod
    def load(cls, bot, path, tools=None):
        import json

        from .caches import QuantSinkKVCache, SinkKVCache, SinkRingKVCache

        arrays, metadata = mx.load(str(path), return_metadata=True)
        meta = json.loads(metadata["inkling2_session"])
        self = cls.__new__(cls)
        self.bot, self.effort, self.tools = bot, meta["effort"], tools
        self.kv_bits = meta["kv_bits"]
        self.messages, self._rendered = meta["messages"], meta["rendered"]
        self.system = self.messages[0]["content"] if self.messages and self.messages[0]["role"] == "system" else None
        st = bot.new_state(self.kv_bits or 16, passes=bot.text_config.passes_for(len(meta["caches"])))
        for i, (c, entry) in enumerate(zip(st.cache, meta["caches"])):
            kv, conv = c[0], c[1]
            if type(kv).__name__ != entry["type"]:
                raise ValueError("session was saved with a different cache layout")
            kv.offset = entry["offset"]
            if isinstance(kv, QuantSinkKVCache):
                kv.bits, kv.group_size = entry["bits"], entry["group_size"]
                if f"c{i}.qk0" in arrays:
                    kv.qk = [arrays[f"c{i}.qk{j}"] for j in range(3)]
                    kv.qv = [arrays[f"c{i}.qv{j}"] for j in range(3)]
            elif isinstance(kv, SinkRingKVCache):
                kv._idx = entry["idx"]
                k, v = arrays.get(f"c{i}.k"), arrays.get(f"c{i}.v")
                if k is not None and k.shape[2] < kv.max_size + 1:  # re-pad the ring (slot W = sink)
                    pad = kv.max_size + 1 - k.shape[2]
                    k = mx.concatenate([k, mx.zeros((*k.shape[:2], pad, k.shape[3]), k.dtype)], axis=2)
                    v = mx.concatenate([v, mx.zeros((*v.shape[:2], pad, v.shape[3]), v.dtype)], axis=2)
                kv.keys, kv.values = k, v
            elif isinstance(kv, SinkKVCache) and f"c{i}.k" in arrays:
                kv.keys, kv.values = arrays[f"c{i}.k"], arrays[f"c{i}.v"]
            for j in entry["conv"]:
                conv[j] = arrays[f"c{i}.conv{j}"]
        if meta["draft"] and bot.drafter is not None:
            st.draft_cache = bot.drafter.reset()
            for i, (kv, entry) in enumerate(zip(st.draft_cache, meta["draft"])):
                kv.offset = entry["offset"]
                if entry["rotating"]:
                    kv._idx = entry["idx"]
                if f"d{i}.k" in arrays:
                    kv.keys, kv.values = arrays[f"d{i}.k"], arrays[f"d{i}.v"]
            st.hidden = arrays.get("hidden")
        st.length, st.pending = meta["length"], meta["pending"]
        self.state = st
        return self
