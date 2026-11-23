"""Model worker for the Rust launcher (launcher/): JSON lines on stdin / stdout.

The launcher owns HTTP and the OpenAI wire formats; this process owns the model and
the conversation states, and schedules every request on the one GPU (a reader thread
takes requests and cancellations while generation runs).

Requests (one JSON object per line):
  {"op": "generate", "id": ..., "messages": [...], "tools": [...] | null,
   "effort": ..., "max_tokens": N, "temperature": T, "top_p": P, "top_k": K,
   "seed": S, "stop": [...], "passes": N | null, "continue_from": key | null,
   "store_as": key | null}
  {"op": "cancel", "id": ...}      {"op": "forget", "key": ...}
  {"op": "stats", "id": ...}       {"op": "shutdown"}
Events (one JSON object per line, all but "ready" carry the request id):
  ready, start, delta (kind reasoning | text), tool_call, done, error, stats

Continuous batching. A request is prefilled on its own, then decodes:
  * alone, when it is the only request: the single-request path with DSpark
    speculative decoding and the 8-bit KV cache (fastest for one stream);
  * otherwise in a batch: rows join a running batch as soon as their prefill is done
    and leave it when they finish (inkling2_mini/batching.py), so one weight read
    serves every row (about 44 tok/s in total at 2 rows, 72 at 8, 90 at 16, vs 34 for
    one). A lone request that a new one arrives for is paused at a token boundary and
    moved into the batch; when the batch is down to one row it goes back to decoding
    alone. Prefill runs in chunks between batch steps, so running rows keep streaming
    while a long prompt is read. Rows with different pass counts (variable-loop
    checkpoints) decode in separate batches.

KV reuse. Every finished reply is stored as an entry: the rendered conversation
including the reply, and the GenState that has processed it. A request renders its
messages with the chat template and continues from the entry whose rendering is the
longest prefix of its own (`continue_from` is tried first), so only the new suffix
is prefilled. Entries are cloned on use, so branching conversations and repeated
system prompts keep hitting. On a miss, a long system + tools prefix is prefilled
and snapshotted first, so new conversations with the same prefix reuse it.
OpenAI clients drop `reasoning_content` from echoed assistant messages; it is
re-attached from the stored replies so the rendering still matches the KV.
"""

import argparse
import base64
import io
import json
import os
import queue
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque

_proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)  # anything else printed (loader, libraries) goes to stderr
sys.stdout = sys.stderr
_lock = threading.Lock()


def send(obj):
    line = json.dumps(obj, ensure_ascii=False)
    with _lock:
        _proto.write(line + "\n")
        _proto.flush()


import mlx.core as mx  # noqa: E402

from .batching import Batch, Row, RowSampler, row_bytes  # noqa: E402
from .generate import (  # noqa: E402
    CONTENT_TEXT, CONTENT_THINKING, CONTENT_TOOL_JSON, END_MESSAGE, END_SAMPLING, MESSAGE_MODEL, Inkling2Mini,
    _cache_arrays,
)

IMAGE_TOKEN = "<|content_image|>"


class Entry:
    def __init__(self, key, rendered, state, passes):
        self.key, self.rendered, self.state, self.passes = key, rendered, state, passes
        self.nbytes = state.nbytes()


class StateCache:
    def __init__(self, budget_bytes, max_entries):
        self.entries = OrderedDict()
        self.budget, self.max_entries = budget_bytes, max_entries

    def find(self, text, prefer=None, passes=None):
        """The entry whose rendering is the longest proper prefix of `text` (states of
        another pass count do not match: their KV belongs to other layer visits)."""

        def fits(e):
            return (e is not None and len(e.rendered) < len(text) and text.startswith(e.rendered)
                    and (passes is None or e.passes == passes))

        best = None
        e = self.entries.get(prefer) if prefer else None
        if fits(e):
            best = e
        else:
            for e in self.entries.values():
                if fits(e) and (best is None or len(e.rendered) > len(best.rendered)):
                    best = e
        if best is not None:
            self.entries.move_to_end(best.key)
        return best

    def put(self, entry):
        self.entries.pop(entry.key, None)
        self.entries[entry.key] = entry
        while self.entries and (len(self.entries) > self.max_entries or self.total() > self.budget):
            if len(self.entries) == 1 and entry.key in self.entries:
                if entry.nbytes > self.budget:
                    self.entries.pop(entry.key)
                break
            self.entries.popitem(last=False)
        mx.clear_cache()

    def forget(self, key):
        self.entries.pop(key, None)

    def total(self):
        return sum(e.nbytes for e in self.entries.values())


def _load_image(url):
    from PIL import Image

    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str):
        raise ValueError("image part without a URL")
    if url.startswith("data:"):
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    if url.startswith("file://"):
        url = url[7:]
    if os.path.exists(url):
        return Image.open(url).convert("RGB")
    raise ValueError("images must be data: URLs or local file paths")


def _parse_args(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            raise ValueError("tool call arguments are not valid JSON")
    return arguments if isinstance(arguments, dict) else {}


def normalize(messages, images):
    """OpenAI chat messages -> the chat template's message dicts (images collected)."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "developer":
            role = "system"
        msg = {"role": role}
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append({"type": "text", "text": p})
                    continue
                kind = p.get("type")
                if kind in ("text", "input_text", "output_text"):
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif kind in ("image_url", "input_image", "image"):
                    images.append(_load_image(p.get("image_url") or p.get("url")))
                    parts.append({"type": "image"})
                else:
                    raise ValueError(f"unsupported content part type {kind!r}")
            if role == "assistant" or all(p["type"] == "text" for p in parts) and len(parts) == 1:
                msg["content"] = "".join(p.get("text", "") for p in parts)
            else:
                msg["content"] = parts
        else:
            msg["content"] = content if isinstance(content, str) else ""
        if role == "assistant":
            reasoning = m.get("reasoning_content") or m.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                msg["reasoning_content"] = reasoning
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {"id": c.get("id"), "type": "function",
                     "function": {"name": c["function"]["name"], "arguments": _parse_args(c["function"].get("arguments"))}}
                    for c in calls
                ]
        elif role == "tool":
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                msg["name"] = m["name"]
        out.append(msg)
    return out


def _reply_key(content, tool_calls):
    calls = [(c["function"]["name"], json.dumps(c["function"]["arguments"], sort_keys=True)) for c in tool_calls or []]
    return json.dumps([content or "", calls], ensure_ascii=False)


class Segments:
    """Generated tokens -> reasoning / text deltas and complete tool calls."""

    def __init__(self, bot):
        ids = bot.ids
        self.tok = bot.tokenizer
        self.modes = {ids[CONTENT_THINKING]: "reasoning", ids[CONTENT_TEXT]: "text", ids[CONTENT_TOOL_JSON]: "tool"}
        self.structural = {ids[MESSAGE_MODEL], ids[END_MESSAGE], ids[END_SAMPLING]}
        self.mode, self.buf, self.emitted = None, [], 0
        self.reasoning_tokens = 0

    def feed(self, token):
        if token in self.modes:
            yield from self.close()
            self.mode, self.buf, self.emitted = self.modes[token], [], 0
        elif token in self.structural:
            yield from self.close()
        elif self.mode is not None:
            self.buf.append(token)
            if self.mode == "reasoning":
                self.reasoning_tokens += 1
            if self.mode != "tool":
                text = self.tok.decode(self.buf)
                if not text.endswith("�") and len(text) > self.emitted:
                    yield (self.mode, text[self.emitted:])
                    self.emitted = len(text)

    def close(self):
        if self.mode == "tool" and self.buf:
            raw = self.tok.decode(self.buf)
            try:
                call = json.loads(raw)
                name, args = call.get("name"), call.get("args", call.get("arguments", {}))
            except (json.JSONDecodeError, AttributeError):
                name, args = None, raw
            yield ("tool", (name, args))
        self.mode, self.buf, self.emitted = None, [], 0


class Request:
    """One generate request: setup (render, state lookup, prefill plan), token handling
    (deltas, tool calls, stop sequences) and the final event."""

    def __init__(self, worker, msg):
        b = worker.bot
        self.w, self.rid = worker, msg["id"]
        self.t0 = time.perf_counter()
        images = []
        msgs = normalize(msg.get("messages") or [], images)
        worker._attach_reasoning(msgs)
        self.msgs, self.tools, self.effort = msgs, msg.get("tools") or None, msg.get("effort")
        self.passes = b.check_passes(msg.get("passes") or b.passes)
        self.store_as = msg.get("store_as")
        text = b.render(msgs, self.effort, self.tools)

        # Prefill plan: [(ids, pixel_values, image_grid_thw, snapshot rendering or None)].
        self.segments = []
        entry = worker.cache.find(text, msg.get("continue_from"), self.passes)
        if entry is not None:
            self.state, prefix = entry.state.clone(), entry.rendered
        else:
            self.state, prefix = b.new_state(passes=self.passes), ""
            snap = worker._system_prefix(msgs, self.effort, self.tools, text)
            if snap is not None:
                prefix, ids = snap
                self.segments.append((ids, None, None, prefix))
        self.cached = self.state.length + len(self.state.pending)
        suffix_images = images[prefix.count(IMAGE_TOKEN):]
        inputs = b.processor(images=suffix_images or None, text=text[len(prefix):])
        self.segments.append((inputs["input_ids"], inputs.get("pixel_values"), inputs.get("image_grid_thw"), None))
        self.prompt_tokens = self.cached + sum(int(s[0].shape[1]) for s in self.segments)
        limit = worker.max_context
        if self.prompt_tokens >= limit:
            raise ValueError(f"prompt is {self.prompt_tokens} tokens; the context limit is {limit}")
        self.max_tokens = max(1, min(int(msg.get("max_tokens") or 4096), limit - self.prompt_tokens))
        self.temperature = float(msg.get("temperature", 1.0))
        self.top_p, self.top_k = float(msg.get("top_p", 1.0)), int(msg.get("top_k") or 0)
        self.seed = msg.get("seed")
        self.sampler = RowSampler(self.temperature, self.top_p, self.top_k, self.seed)
        self.row = None  # batching.Row while decoding

        self.stops = [s for s in (msg.get("stop") or []) if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.seg = Segments(b)
        self.reply_text, self.reasoning, self.calls = "", "", []
        self.sent_text, self.stopped = 0, False
        self.n_tokens, self.last = 0, None
        self.prefill_tokens, self.t_prefill, self.t_first = 0, None, None
        self.batched_tokens, self.max_batch, self.spec = 0, 1, None
        send({"type": "start", "id": self.rid, "prompt_tokens": self.prompt_tokens, "cached_tokens": self.cached})

    # --------------------------------------------------------------- tokens
    def emit(self, token):
        """Hands one generated token to the client. Returns None to go on, or why the
        request ends with it: 'eos', 'length', 'stop' or 'cancelled'."""
        rid = self.rid
        self.n_tokens, self.last = self.n_tokens + 1, token
        for kind, val in self.seg.feed(token):
            if kind == "reasoning":
                self.reasoning += val
                send({"type": "delta", "id": rid, "kind": "reasoning", "text": val})
            elif kind == "text":
                self.reply_text += val
                cut = min((i for i in (self.reply_text.find(s) for s in self.stops) if i >= 0), default=-1)
                if cut >= 0:
                    self.reply_text, self.stopped = self.reply_text[:cut], True
                upto = len(self.reply_text) if self.stopped else max(self.sent_text, len(self.reply_text) - self.hold)
                if upto > self.sent_text:
                    send({"type": "delta", "id": rid, "kind": "text", "text": self.reply_text[self.sent_text:upto]})
                    self.sent_text = upto
            else:
                self._tool_call(*val)
        if self.stopped:
            return "stop"
        if rid in self.w.cancelled:
            return "cancelled"
        if token == self.w.bot.eos_id:
            return "eos"
        if self.n_tokens >= self.max_tokens:
            return "length"
        return None

    def _tool_call(self, name, args):
        self.calls.append({"name": name, "arguments": args})
        send({"type": "tool_call", "id": self.rid, "index": len(self.calls) - 1, "name": name, "arguments": args})

    def finish(self, outcome, state=None):
        """Final events. state: the GenState that has processed this reply (stored for
        KV reuse when the reply ended normally)."""
        b, w = self.w.bot, self.w
        for kind, val in self.seg.close():
            if kind == "tool":
                self._tool_call(*val)
        if self.sent_text < len(self.reply_text):
            send({"type": "delta", "id": self.rid, "kind": "text", "text": self.reply_text[self.sent_text:]})
        if outcome == "cancelled":
            finish = "cancelled"
        elif outcome == "stop":
            finish = "stop"
        elif self.calls:
            finish = "tool_calls"
        elif self.last != b.eos_id:
            finish = "length"
        else:
            finish = "stop"

        key = None
        if outcome in ("eos", "length") and state is not None and state.cache is not None:
            if self.last != b.eos_id:
                state.pending += [b.ids[END_MESSAGE], b.eos_id]  # close a truncated reply
            assistant = {"role": "assistant", "content": self.reply_text}
            if self.reasoning:
                assistant["reasoning_content"] = self.reasoning
            good_calls = [c for c in self.calls if c["name"] and isinstance(c["arguments"], dict)]
            if good_calls:
                assistant["tool_calls"] = [{"id": None, "type": "function",
                                            "function": {"name": c["name"], "arguments": c["arguments"]}}
                                           for c in good_calls]
            if self.reasoning:
                w.reasoning[_reply_key(self.reply_text, assistant.get("tool_calls"))] = self.reasoning
                while len(w.reasoning) > 512:
                    w.reasoning.popitem(last=False)
            key = self.store_as or f"conv-{uuid.uuid4().hex[:12]}"
            try:
                rendered = b.render(self.msgs + [assistant], self.effort, self.tools)
                w.cache.put(Entry(key, rendered, state, self.passes))
            except Exception:  # noqa: BLE001 - an unrenderable reply just isn't cached
                key = None
        now = time.perf_counter()
        prefill_s = (self.t_first or now) - (self.t_prefill or now)
        gen_s = now - (self.t_first or now)
        stats = {"prefill_tokens": self.prefill_tokens,
                 "prefill_tps": self.prefill_tokens / prefill_s if prefill_s > 0 else 0.0,
                 "gen_tps": (self.n_tokens - 1) / gen_s if gen_s > 0 and self.n_tokens > 1 else 0.0,
                 "tokens_per_round": (self.spec or {}).get("tokens_per_round"),
                 "seconds": now - self.t0, "queued_s": (self.t_prefill or now) - self.t0,
                 "batched_tokens": self.batched_tokens, "max_batch": self.max_batch, "passes": self.passes}
        send({"type": "done", "id": self.rid, "finish_reason": finish, "text": self.reply_text,
              "reasoning": self.reasoning, "tool_calls": self.calls, "entry": key,
              "usage": {"prompt_tokens": self.prompt_tokens, "cached_tokens": self.cached,
                        "completion_tokens": self.n_tokens, "reasoning_tokens": self.seg.reasoning_tokens},
              "stats": stats})
        w.cancelled.discard(self.rid)


class PrefillJob:
    """A request's prompt, prefilled chunk by chunk (other work runs in between), then
    its first token sampled."""

    def __init__(self, worker, req, chunk):
        self.w, self.req, self.chunk = worker, req, chunk
        self.index, self.it = 0, None
        req.t_prefill = time.perf_counter()

    def _start(self):
        b, st = self.w.bot, self.req.state
        ids, pixels, grid, _ = self.req.segments[self.index]
        h = b.model.embed(ids, pixels, grid)
        if st.pending:
            h = mx.concatenate([b.model.embed(mx.array([st.pending])), h], axis=1)
        self.n = h.shape[1]
        self.capture = b.capture_ids(st)
        self.it = b.prefill_chunks(h, st.cache, self.chunk, self.capture, ctx=st.length)

    def step(self):
        """Prefills one chunk. Returns the first generated token once there is one."""
        if self.it is None:
            self._start()
        try:
            next(self.it)
            return None
        except StopIteration as done:
            logits, fresh = done.value
        req, st = self.req, self.req.state
        st.length += self.n
        st.pending = []
        if self.capture is not None:
            st.hidden = fresh if st.hidden is None else mx.concatenate([st.hidden, fresh], axis=1)
        req.prefill_tokens += self.n
        snapshot = req.segments[self.index][3]
        self.index, self.it = self.index + 1, None
        if snapshot is not None:  # a shared system prefix: keep a copy for other conversations
            mx.eval(_cache_arrays(st.cache), *([] if st.hidden is None else [st.hidden]))
            self.w.cache.put(Entry(f"prefix-{uuid.uuid4().hex[:12]}", snapshot, st.clone(), req.passes))
            return None
        token = req.sampler(logits.reshape(1, -1))
        mx.eval(token, _cache_arrays(st.cache), *([] if st.hidden is None else [st.hidden]))
        req.t_first = time.perf_counter()
        st.pending = [token.item()]
        return st.pending[0]


class Worker:
    def __init__(self, args):
        self.bot = Inkling2Mini(args.model, verbose=True, dense_bits=args.dense_bits, kv_bits=args.kv_bits,
                                speculative=not args.no_spec, passes=args.passes)
        self.cache = StateCache(int(args.cache_gb * 1e9), args.cache_entries)
        self.max_context = args.max_context
        self.snapshot_min = args.snapshot_min_tokens
        self.max_batch = max(1, args.max_batch)
        self.batch_budget = int(args.batch_gb * 1e9)
        self.chunk, self.decode_share = args.prefill_chunk, args.decode_share
        self.reasoning = OrderedDict()  # reply key -> thinking text
        self.cancelled = set()
        self.inbox = queue.Queue()
        self.waiting = deque()  # set up, not prefilling yet
        self.job = None  # the PrefillJob running
        self.solo = None  # the Request decoding alone (single-request caches)
        self.batches = {}  # passes -> Batch
        self.peak_batch = 0

    # ------------------------------------------------------------------ input
    def read_stdin(self):
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("op") == "cancel":
                self.cancelled.add(msg.get("id"))
            else:
                self.inbox.put(msg)
        self.inbox.put({"op": "shutdown"})

    def run(self):
        threading.Thread(target=self.read_stdin, daemon=True).start()
        b = self.bot
        cfg = b.text_config
        send({"type": "ready", "model": b.model_dir.name, "max_context": self.max_context,
              "speculative": b.drafter is not None, "kv_bits": b.kv_bits or 16, "passes": b.passes,
              "varloop": cfg.is_varloop, "max_batch": self.max_batch})
        while self._drain():
            self._tick()

    def _drain(self):
        """Handles queued messages; blocks while there is nothing to run. False on shutdown."""
        block = not self._busy()
        while True:
            try:
                msg = self.inbox.get(block=block)
            except queue.Empty:
                return True
            block = False
            op = msg.get("op")
            if op == "shutdown":
                return False
            if op == "forget":
                self.cache.forget(msg.get("key"))
            elif op == "stats":
                send({"type": "stats", "id": msg.get("id"), "entries": len(self.cache.entries),
                      "cache_gb": self.cache.total() / 1e9, "active_gb": mx.get_active_memory() / 1e9,
                      "decoding": self._decoding(), "waiting": len(self.waiting) + (self.job is not None),
                      "batches": {str(p): len(bt) for p, bt in self.batches.items() if len(bt)},
                      "peak_batch": self.peak_batch})
            elif op == "generate":
                rid = msg.get("id")
                if rid in self.cancelled:
                    self.cancelled.discard(rid)
                    continue
                try:
                    self.waiting.append(Request(self, msg))
                except Exception as e:  # noqa: BLE001 - report to the client, keep serving
                    self._error(rid, e)

    def _busy(self):
        return bool(self.waiting or self.job or self._decoding())

    def _decoding(self):
        return (self.solo is not None) + sum(len(bt) for bt in self.batches.values())

    def _error(self, rid, e):
        import traceback

        traceback.print_exc()
        send({"type": "error", "id": rid, "message": f"{type(e).__name__}: {e}"})
        self.cancelled.discard(rid)

    # -------------------------------------------------------------- schedule
    def _tick(self):
        self._drop_cancelled_waiting()
        n = self._decoding()
        if self.job is None and self.waiting and self._can_admit(self.waiting[0], n):
            self.job = PrefillJob(self, self.waiting.popleft(), self.chunk)
        if self.job is not None:
            t = time.perf_counter()
            self._prefill_step()
            spent = time.perf_counter() - t
            if self.job is not None and self._decoding():
                # Still prefilling: the running rows get decode_share of the chunk's time
                # (at least one step), so they keep streaming while a long prompt is read.
                end = time.perf_counter() + self.decode_share * spent
                while self._decoding():
                    self._step_batches()
                    if time.perf_counter() >= end:
                        break
                return
            if not self._decoding():
                return
        if self._solo_ok():
            self._run_solo()
        else:
            self._step_batches()

    def _solo_ok(self):
        if self._decoding() != 1 or self.job is not None:
            return False
        if self.waiting and self._can_admit(self.waiting[0], 1):
            return False
        if self.solo is None:  # the last row of a batch goes back to decoding alone
            batch = next(bt for bt in self.batches.values() if len(bt))
            row = batch.rows[0]
            token = batch.settle(row)
            outcome = None
            if token is not None:
                row.req.batched_tokens += 1
                outcome = row.req.emit(token)
            if outcome is not None:
                self._leave(batch, [(row.req, outcome)])
                return False
            batch.remove([row], keep_state=[row])
            self.solo = row.req
        return True

    def _can_admit(self, req, n):
        if n == 0:
            return True
        if n >= self.max_batch:
            return False
        cfg = self.bot.text_config
        total = 0
        for passes in {req.passes, *self.batches.keys()}:
            lengths = [r.length for r in self.batches[passes].rows] if passes in self.batches else []
            if self.solo is not None and self.solo.passes == passes:
                lengths.append(self.solo.state.length + 1)
            if passes == req.passes:
                lengths.append(req.prompt_tokens)
            if lengths:
                total += len(lengths) * row_bytes(cfg, max(lengths) + 512, passes)
        return total <= self.batch_budget

    def _drop_cancelled_waiting(self):
        if self.cancelled and self.waiting:
            for req in [r for r in self.waiting if r.rid in self.cancelled]:
                self.waiting.remove(req)
                req.finish("cancelled")

    def _prefill_step(self):
        job, req = self.job, self.job.req
        if req.rid in self.cancelled:
            self.job = None
            req.finish("cancelled")
            return
        try:
            token = job.step()
        except Exception as e:  # noqa: BLE001
            self.job = None
            self._error(req.rid, e)
            mx.clear_cache()
            return
        if token is None:
            return
        self.job = None
        outcome = req.emit(token)
        if outcome is not None:
            req.finish(outcome, req.state)
            return
        req.row = Row(req.sampler, req.state)
        req.row.req = req  # the scheduler maps rows back to their requests
        if self._decoding() == 0 and not self.waiting:
            self.solo = req
        else:
            self._to_batch(req)

    def _batch_for(self, passes):
        bt = self.batches.get(passes)
        if bt is None:
            capture = (self.bot.drafter.capture_ids(self.bot.text_config, passes)
                       if self.bot.drafter is not None else None)
            bt = self.batches[passes] = Batch(self.bot, passes, capture)
        return bt

    def _to_batch(self, req):
        if self.solo is not None and self.solo is not req:
            solo, self.solo = self.solo, None
            self._to_batch(solo)
        bt = self._batch_for(req.passes)
        bt.add(req.row)
        self.peak_batch = max(self.peak_batch, self._decoding())
        for r in bt.rows:
            r.req.max_batch = max(r.req.max_batch, len(bt))

    def _run_solo(self):
        """Decodes the lone request with the single-request path (DSpark, 8-bit KV) until
        it finishes or something else needs the GPU."""
        req = self.solo
        st = req.row.state
        ids = mx.array([st.pending], mx.int32)
        st.pending = []
        seed = None if req.seed is None else (int(req.seed) * 1000003 + req.n_tokens) % (1 << 31)
        gen = self.bot.generate_tokens(
            ids, state=st, max_tokens=req.max_tokens - req.n_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k, seed=seed, should_pause=lambda: not self.inbox.empty(),
        )
        outcome = None
        try:
            for token, stats in gen:
                outcome = req.emit(token)
                if stats is not None:
                    req.spec = stats
                if outcome is not None:
                    break
        except Exception as e:  # noqa: BLE001
            self.solo = None
            self._error(req.rid, e)
            mx.clear_cache()
            return
        finally:
            gen.close()
        if outcome is not None:
            self.solo = None
            req.finish(outcome, st if outcome in ("eos", "length") else None)
            mx.clear_cache()
        # else paused: st is consistent (the last token pending); the next tick decides
        # whether the request goes on alone or joins a batch

    def _step_batches(self):
        if self.solo is not None:  # something else needs the GPU: the lone request joins a batch
            solo, self.solo = self.solo, None
            self._to_batch(solo)
        for passes, bt in list(self.batches.items()):
            if not len(bt):
                continue
            try:
                emitted = bt.step()
            except Exception as e:  # noqa: BLE001 - the whole batch fails
                rows, bt.rows, bt.cache = bt.rows, [], None
                for r in rows:
                    self._error(r.req.rid, e)
                mx.clear_cache()
                continue
            ended = []
            for row, token in emitted:
                row.req.batched_tokens += 1
                outcome = row.req.emit(token)
                if outcome is not None:
                    ended.append((row.req, outcome))
            seen = {id(r) for r, _ in ended}
            for r in bt.rows:  # cancelled rows that emitted nothing this step
                if r.req.rid in self.cancelled and id(r.req) not in seen:
                    ended.append((r.req, "cancelled"))
            if ended:
                self._leave(bt, ended)

    def _leave(self, bt, ended):
        keep = [req.row for req, o in ended if o in ("eos", "length")]
        bt.remove([req.row for req, _ in ended], keep_state=keep)
        for req, outcome in ended:
            req.finish(outcome, req.row.state if outcome in ("eos", "length") else None)

    # ------------------------------------------------------------- KV reuse
    def _attach_reasoning(self, msgs):
        for m in msgs:
            if m["role"] == "assistant" and "reasoning_content" not in m:
                r = self.reasoning.get(_reply_key(m.get("content"), m.get("tool_calls")))
                if r:
                    m["reasoning_content"] = r

    def _system_prefix(self, msgs, effort, tools, text):
        """On a miss: the system messages + tool declarations as a prefix worth
        snapshotting (long enough, a proper prefix of the request). Returns
        (rendered prefix, ids) or None; the prefill job stores the snapshot."""
        k = 0
        while k < len(msgs) and msgs[k]["role"] == "system":
            k += 1
        if k == 0:  # the template cannot render tools without a message
            return None
        prefix = self.bot.render(msgs[:k], effort, tools)
        if not text.startswith(prefix) or len(prefix) >= len(text):
            return None
        ids = self.bot.processor(text=prefix)["input_ids"]
        if ids.shape[1] < self.snapshot_min:
            return None
        return prefix, ids


def main():
    ap = argparse.ArgumentParser(description="Inkling 2 Mini model worker (JSON lines on stdin/stdout)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dense-bits", type=int, default=6)
    ap.add_argument("--kv-bits", type=int, default=8)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--passes", type=int, default=None,
                    help="loop passes (default: the checkpoint's; 1-5 for variable-loop checkpoints)")
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--cache-gb", type=float, default=4.0, help="memory for stored conversation states")
    ap.add_argument("--cache-entries", type=int, default=16)
    ap.add_argument("--snapshot-min-tokens", type=int, default=512)
    ap.add_argument("--max-batch", type=int, default=16, help="requests decoding at once (1: one at a time)")
    ap.add_argument("--batch-gb", type=float, default=6.0, help="memory for the running batch's KV caches")
    ap.add_argument("--prefill-chunk", type=int, default=1024, help="tokens per prefill chunk")
    ap.add_argument("--decode-share", type=float, default=0.25,
                    help="while a prompt is prefilled, decode steps for the running requests take this "
                         "fraction of each chunk's time (at least one step between chunks)")
    args = ap.parse_args()
    try:
        Worker(args).run()
    except Exception as e:  # noqa: BLE001
        send({"type": "fatal", "message": f"{type(e).__name__}: {e}"})
        raise


if __name__ == "__main__":
    main()
