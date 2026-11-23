"""Model worker for the Rust launcher (launcher/): JSON lines on stdin / stdout.

The launcher owns HTTP and the OpenAI wire formats; this process owns the model and
the conversation states. Requests run one at a time (one GPU); a reader thread
handles cancellation while a generation runs.

Requests (one JSON object per line):
  {"op": "generate", "id": ..., "messages": [...], "tools": [...] | null,
   "effort": ..., "max_tokens": N, "temperature": T, "top_p": P, "top_k": K,
   "seed": S, "stop": [...], "continue_from": key | null, "store_as": key | null}
  {"op": "cancel", "id": ...}      {"op": "forget", "key": ...}
  {"op": "stats", "id": ...}       {"op": "shutdown"}
Events (one JSON object per line, all but "ready" carry the request id):
  ready, start, delta (kind reasoning | text), tool_call, done, error, stats

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
from collections import OrderedDict

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

from .generate import (  # noqa: E402
    CONTENT_TEXT, CONTENT_THINKING, CONTENT_TOOL_JSON, END_MESSAGE, END_SAMPLING, MESSAGE_MODEL, Inkling2Mini,
)

IMAGE_TOKEN = "<|content_image|>"


class Entry:
    def __init__(self, key, rendered, state):
        self.key, self.rendered, self.state = key, rendered, state
        self.nbytes = state.nbytes()


class StateCache:
    def __init__(self, budget_bytes, max_entries):
        self.entries = OrderedDict()
        self.budget, self.max_entries = budget_bytes, max_entries

    def find(self, text, prefer=None):
        best = None
        e = self.entries.get(prefer) if prefer else None
        if e is not None and len(e.rendered) < len(text) and text.startswith(e.rendered):
            best = e
        else:
            for e in self.entries.values():
                if (len(e.rendered) < len(text) and text.startswith(e.rendered)
                        and (best is None or len(e.rendered) > len(best.rendered))):
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


class Worker:
    def __init__(self, args):
        self.bot = Inkling2Mini(args.model, verbose=True, dense_bits=args.dense_bits, kv_bits=args.kv_bits,
                                speculative=not args.no_spec)
        self.cache = StateCache(int(args.cache_gb * 1e9), args.cache_entries)
        self.max_context = args.max_context
        self.snapshot_min = args.snapshot_min_tokens
        self.reasoning = OrderedDict()  # reply key -> thinking text
        self.cancelled = set()
        self.requests = queue.Queue()

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
                self.requests.put(msg)
        self.requests.put({"op": "shutdown"})

    def run(self):
        threading.Thread(target=self.read_stdin, daemon=True).start()
        b = self.bot
        send({"type": "ready", "model": b.model_dir.name, "max_context": self.max_context,
              "speculative": b.drafter is not None, "kv_bits": b.kv_bits or 16})
        while True:
            msg = self.requests.get()
            op = msg.get("op")
            if op == "shutdown":
                return
            if op == "forget":
                self.cache.forget(msg.get("key"))
            elif op == "stats":
                send({"type": "stats", "id": msg.get("id"), "entries": len(self.cache.entries),
                      "cache_gb": self.cache.total() / 1e9, "active_gb": mx.get_active_memory() / 1e9})
            elif op == "generate":
                rid = msg.get("id")
                if rid in self.cancelled:
                    self.cancelled.discard(rid)
                    continue
                try:
                    self.generate(msg)
                except Exception as e:  # noqa: BLE001 - report to the client, keep serving
                    import traceback

                    traceback.print_exc()
                    send({"type": "error", "id": rid, "message": f"{type(e).__name__}: {e}"})
                finally:
                    self.cancelled.discard(rid)
                    mx.clear_cache()

    # ------------------------------------------------------------- generation
    def _attach_reasoning(self, msgs):
        for m in msgs:
            if m["role"] == "assistant" and "reasoning_content" not in m:
                r = self.reasoning.get(_reply_key(m.get("content"), m.get("tool_calls")))
                if r:
                    m["reasoning_content"] = r

    def generate(self, req):
        b, rid = self.bot, req["id"]
        images = []
        msgs = normalize(req.get("messages") or [], images)
        self._attach_reasoning(msgs)
        tools, effort = req.get("tools") or None, req.get("effort")
        text = b.render(msgs, effort, tools)
        t0 = time.perf_counter()

        entry = self.cache.find(text, req.get("continue_from"))
        if entry is not None:
            state, prefix = entry.state.clone(), entry.rendered
        else:
            state, prefix = self._snapshot_system_prefix(msgs, effort, tools, text)
        cached = state.length + len(state.pending)
        suffix = text[len(prefix):]
        suffix_images = images[prefix.count(IMAGE_TOKEN):]
        inputs = b.processor(images=suffix_images or None, text=suffix)
        ids = inputs["input_ids"]
        prompt_tokens = cached + ids.shape[1]
        if prompt_tokens >= self.max_context:
            raise ValueError(f"prompt is {prompt_tokens} tokens; the context limit is {self.max_context}")
        max_tokens = max(1, min(int(req.get("max_tokens") or 4096), self.max_context - prompt_tokens))
        send({"type": "start", "id": rid, "prompt_tokens": prompt_tokens, "cached_tokens": cached})

        stops = [s for s in (req.get("stop") or []) if s]
        hold = max((len(s) for s in stops), default=1) - 1
        seg = Segments(b)
        reply_text, reasoning, calls = "", "", []
        sent_text = 0
        finish, n_tokens, stats, last = "stop", 0, {}, None
        gen = b.generate_tokens(
            ids, inputs.get("pixel_values"), inputs.get("image_grid_thw"), state=state,
            max_tokens=max_tokens, temperature=float(req.get("temperature", 1.0)),
            top_p=float(req.get("top_p", 1.0)), top_k=int(req.get("top_k") or 0), seed=req.get("seed"),
        )
        stopped = False
        try:
            for token, st in gen:
                n_tokens, last = n_tokens + 1, token
                for kind, val in seg.feed(token):
                    if kind == "reasoning":
                        reasoning += val
                        send({"type": "delta", "id": rid, "kind": "reasoning", "text": val})
                    elif kind == "text":
                        reply_text += val
                        cut = min((i for i in (reply_text.find(s) for s in stops) if i >= 0), default=-1)
                        if cut >= 0:
                            reply_text, stopped = reply_text[:cut], True
                        upto = len(reply_text) if stopped else max(sent_text, len(reply_text) - hold)
                        if upto > sent_text:
                            send({"type": "delta", "id": rid, "kind": "text", "text": reply_text[sent_text:upto]})
                            sent_text = upto
                    else:
                        name, args = val
                        calls.append({"name": name, "arguments": args})
                        send({"type": "tool_call", "id": rid, "index": len(calls) - 1, "name": name, "arguments": args})
                if st is not None:
                    stats = st
                if stopped or rid in self.cancelled:
                    break
        finally:
            gen.close()
        for kind, val in seg.close():
            if kind == "tool":
                calls.append({"name": val[0], "arguments": val[1]})
                send({"type": "tool_call", "id": rid, "index": len(calls) - 1, "name": val[0], "arguments": val[1]})
        if sent_text < len(reply_text):
            send({"type": "delta", "id": rid, "kind": "text", "text": reply_text[sent_text:]})
        cancelled = rid in self.cancelled
        if cancelled:
            finish = "cancelled"
        elif stopped:
            finish = "stop"
        elif calls:
            finish = "tool_calls"
        elif last != b.eos_id:
            finish = "length"

        key = None
        if not cancelled and not stopped and stats:
            if last != b.eos_id:
                state.pending += [b.ids[END_MESSAGE], b.eos_id]  # close a truncated reply
            assistant = {"role": "assistant", "content": reply_text}
            if reasoning:
                assistant["reasoning_content"] = reasoning
            good_calls = [c for c in calls if c["name"] and isinstance(c["arguments"], dict)]
            if good_calls:
                assistant["tool_calls"] = [{"id": None, "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                           for c in good_calls]
            if reasoning:
                self.reasoning[_reply_key(reply_text, assistant.get("tool_calls"))] = reasoning
                while len(self.reasoning) > 512:
                    self.reasoning.popitem(last=False)
            key = req.get("store_as") or f"conv-{uuid.uuid4().hex[:12]}"
            try:
                rendered = b.render(msgs + [assistant], effort, tools)
                self.cache.put(Entry(key, rendered, state))
            except Exception:  # noqa: BLE001 - an unrenderable reply just isn't cached
                key = None
        dt = time.perf_counter() - t0
        send({"type": "done", "id": rid, "finish_reason": finish, "text": reply_text, "reasoning": reasoning,
              "tool_calls": calls, "entry": key,
              "usage": {"prompt_tokens": prompt_tokens, "cached_tokens": cached, "completion_tokens": n_tokens,
                        "reasoning_tokens": seg.reasoning_tokens},
              "stats": {"prefill_tokens": int(ids.shape[1]), "prefill_tps": stats.get("prefill_tps", 0.0),
                        "gen_tps": stats.get("gen_tps", 0.0), "tokens_per_round": stats.get("tokens_per_round"),
                        "seconds": dt}})

    def _snapshot_system_prefix(self, msgs, effort, tools, text):
        """On a miss, prefill the system messages + tool declarations as their own
        entry when they are long, so other conversations with the same prefix reuse
        them. Returns (state, rendered prefix already in it)."""
        state = self.bot.new_state()
        k = 0
        while k < len(msgs) and msgs[k]["role"] == "system":
            k += 1
        if k == 0:  # the template cannot render tools without a message
            return state, ""
        prefix = self.bot.render(msgs[:k], effort, tools)
        if not text.startswith(prefix) or len(prefix) >= len(text):
            return state, ""
        ids = self.bot.processor(text=prefix)["input_ids"]
        if ids.shape[1] < self.snapshot_min:
            return state, ""
        self.bot.prefill(ids, state)
        self.cache.put(Entry(f"prefix-{uuid.uuid4().hex[:12]}", prefix, state.clone()))
        return state, prefix


def main():
    ap = argparse.ArgumentParser(description="Inkling 2 Mini model worker (JSON lines on stdin/stdout)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dense-bits", type=int, default=6)
    ap.add_argument("--kv-bits", type=int, default=8)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--cache-gb", type=float, default=4.0, help="memory for stored conversation states")
    ap.add_argument("--cache-entries", type=int, default=16)
    ap.add_argument("--snapshot-min-tokens", type=int, default=512)
    args = ap.parse_args()
    try:
        Worker(args).run()
    except Exception as e:  # noqa: BLE001
        send({"type": "fatal", "message": f"{type(e).__name__}: {e}"})
        raise


if __name__ == "__main__":
    main()
