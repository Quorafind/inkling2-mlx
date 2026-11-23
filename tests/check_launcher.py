"""End-to-end check of the Rust launcher (launcher/, binary `inkling2`): starts the
server, then exercises the OpenAI-compatible endpoints with the standard library.

  - /v1/models, errors (bad JSON 400, unknown previous_response_id 404)
  - chat completions: non-streaming, streaming (reasoning_content / content deltas,
    usage chunk, [DONE]), multi-turn KV reuse, tool calls (both modes)
  - responses: non-streaming, streaming event sequence, previous_response_id,
    function_call -> function_call_output round trip, GET / DELETE
  - a client that disconnects mid-stream frees the model for the next request
  - time to first token on turn 2 of a long-document conversation: cold prefill vs
    chat completions resending the history vs responses with previous_response_id

    python tests/check_launcher.py [PORT]
"""

import http.client
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
BIN = ROOT / "launcher" / "target" / "release" / "inkling2"
server = subprocess.Popen([str(BIN), "serve", "--port", str(PORT)], cwd=ROOT)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)


def conn():
    return http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)


def call(method, path, body=None):
    c = conn()
    c.request(method, path, body=None if body is None else (body if isinstance(body, (bytes, str)) else json.dumps(body)),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    try:
        return r.status, json.loads(data)
    except json.JSONDecodeError:
        return r.status, data


def stream(path, body, stop_after=None):
    """POST with stream=true; returns (events [(event, data)], seconds to first delta)."""
    c = conn()
    t0 = time.perf_counter()
    c.request("POST", path, body=json.dumps({**body, "stream": True}), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    events, ev_name, first = [], None, None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode().rstrip("\r\n")
        if line.startswith("event:"):
            ev_name = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                events.append((ev_name, "[DONE]"))
                break
            obj = json.loads(data)
            events.append((ev_name, obj))
            delta = (obj.get("type", "").endswith(".delta")
                     or any(ch.get("delta", {}).get(k) for ch in obj.get("choices", []) for k in ("content", "reasoning_content")))
            if delta and first is None:
                first = time.perf_counter() - t0
            if stop_after and len(events) >= stop_after:
                c.close()  # client goes away mid-stream
                return events, first
            if obj.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                break
        ev_name = None if line == "" else ev_name
    return events, first


for _ in range(600):
    try:
        if call("GET", "/health")[1].get("status") == "ok":
            break
    except OSError:
        pass
    time.sleep(1)
else:
    print("server did not start")
    server.kill()
    sys.exit(1)

try:
    s, m = call("GET", "/v1/models")
    check("GET /v1/models", s == 200 and m["data"][0]["id"] == "inkling-2-mini")
    s, e = call("POST", "/v1/chat/completions", b"{not json")
    check("bad JSON -> 400 with an error object", s == 400 and "error" in e)
    s, e = call("POST", "/v1/responses", {"input": "hi", "previous_response_id": "resp_missing"})
    check("unknown previous_response_id -> 404", s == 404 and e["error"]["param"] == "previous_response_id")

    # ---------------------------------------------------------------- chat
    msgs = [{"role": "user", "content": "用一句话介绍一下杭州。"}]
    s, r = call("POST", "/v1/chat/completions", {"model": "x", "messages": msgs, "temperature": 0, "max_tokens": 200})
    msg = r["choices"][0]["message"]
    check("chat: non-streaming", s == 200 and msg["content"] and r["usage"]["completion_tokens"] > 0,
          f"{msg['content'][:40]!r}, reasoning {len(msg.get('reasoning_content') or '')} chars, finish {r['choices'][0]['finish_reason']}")
    msgs2 = msgs + [{"role": "assistant", "content": msg["content"]}, {"role": "user", "content": "再说说西湖。"}]
    ev, _ = stream("/v1/chat/completions", {"messages": msgs2, "temperature": 0, "max_tokens": 200,
                                            "stream_options": {"include_usage": True}})
    chunks = [d for _, d in ev if isinstance(d, dict)]
    content = "".join(ch["delta"].get("content") or "" for d in chunks for ch in d.get("choices", []))
    usage = next((d["usage"] for d in chunks if d.get("usage")), {})
    check("chat: streaming deltas + usage chunk + [DONE]", content and usage and ev[-1][1] == "[DONE]",
          f"{content[:40]!r}")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    check("chat: turn 2 reuses turn 1's KV", cached > r["usage"]["prompt_tokens"],
          f"{cached} of {usage.get('prompt_tokens')} prompt tokens cached")

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    ask = [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]
    s, r = call("POST", "/v1/chat/completions", {"messages": ask, "tools": tools, "temperature": 0,
                                                 "reasoning_effort": "none"})
    tc = (r["choices"][0]["message"].get("tool_calls") or [{}])[0]
    check("chat: tool call", r["choices"][0]["finish_reason"] == "tool_calls" and tc.get("function", {}).get("name") == "get_weather"
          and json.loads(tc["function"]["arguments"]).get("city"), json.dumps(tc.get("function")))
    follow = ask + [r["choices"][0]["message"], {"role": "tool", "tool_call_id": tc["id"], "content": "18°C, light rain"}]
    ev, _ = stream("/v1/chat/completions", {"messages": follow, "tools": tools, "temperature": 0,
                                            "reasoning_effort": "none", "stream_options": {"include_usage": True}})
    text = "".join(ch["delta"].get("content") or "" for _, d in ev if isinstance(d, dict) for ch in d.get("choices", []))
    usage = next((d["usage"] for _, d in ev if isinstance(d, dict) and d.get("usage")), {})
    check("chat: tool result -> answer (streamed)", "18" in text,
          f"{text[:60]!r}, {usage.get('prompt_tokens_details', {}).get('cached_tokens')} cached")

    # ----------------------------------------------------------- responses
    s, r = call("POST", "/v1/responses", {"input": "Name three primary colors.", "temperature": 0,
                                          "reasoning": {"effort": "low"}, "max_output_tokens": 300})
    kinds = [o["type"] for o in r.get("output", [])]
    text = "".join(c["text"] for o in r["output"] if o["type"] == "message" for c in o["content"])
    check("responses: non-streaming", s == 200 and r["status"] == "completed" and "message" in kinds and text,
          f"items {kinds}, {text[:40]!r}")
    first_id = r["id"]
    ev, _ = stream("/v1/responses", {"input": "And three secondary colors?", "previous_response_id": first_id,
                                     "temperature": 0, "reasoning": {"effort": "low"}, "max_output_tokens": 300})
    names = [n for n, _ in ev]
    done = ev[-1][1]
    deltas = "".join(d["delta"] for n, d in ev if n == "response.output_text.delta")
    final = "".join(c["text"] for o in done["response"]["output"] if o["type"] == "message" for c in o["content"])
    seq_ok = [d["sequence_number"] for _, d in ev] == list(range(len(ev)))
    check("responses: streaming event sequence", names[0] == "response.created" and names[-1] == "response.completed"
          and "response.output_item.added" in names and deltas == final and seq_ok, f"{len(ev)} events")
    u = done["response"]["usage"]
    check("responses: previous_response_id reuses the KV", u["input_tokens_details"]["cached_tokens"] >= r["usage"]["total_tokens"],
          f"{u['input_tokens_details']['cached_tokens']} of {u['input_tokens']} input tokens cached")

    rtools = [{"type": "function", "name": "get_weather", "description": "Current weather for a city",
               "parameters": tools[0]["function"]["parameters"]}]
    s, r = call("POST", "/v1/responses", {"input": "What's the weather in Paris? Use the tool.", "temperature": 0,
                                          "reasoning": {"effort": "none"}, "tools": rtools})
    fc = next((o for o in r["output"] if o["type"] == "function_call"), None)
    check("responses: function_call item", fc is not None and fc["name"] == "get_weather", json.dumps(fc))
    if fc:
        s, r2 = call("POST", "/v1/responses", {"previous_response_id": r["id"], "temperature": 0, "reasoning": {"effort": "none"},
                                               "input": [{"type": "function_call_output", "call_id": fc["call_id"], "output": "18°C, light rain"}],
                                               "tools": rtools})
        text = "".join(c["text"] for o in r2["output"] if o["type"] == "message" for c in o["content"])
        check("responses: function_call_output -> answer, KV reused", "18" in text and r2["usage"]["input_tokens_details"]["cached_tokens"] > 0,
              f"{text[:60]!r}, {r2['usage']['input_tokens_details']['cached_tokens']} cached")
    s, g = call("GET", f"/v1/responses/{first_id}")
    check("GET /v1/responses/{id}", s == 200 and g["id"] == first_id)
    s, d = call("DELETE", f"/v1/responses/{first_id}")
    s2, _ = call("GET", f"/v1/responses/{first_id}")
    check("DELETE /v1/responses/{id}", s == 200 and d.get("deleted") and s2 == 404)

    # ------------------------------------------------- disconnect cancels
    stream("/v1/chat/completions", {"messages": [{"role": "user", "content": "Write a very long story about the sea."}],
                                    "max_tokens": 3000, "reasoning_effort": "none"}, stop_after=5)
    t = time.perf_counter()
    s, r = call("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say OK."}],
                                                 "max_tokens": 10, "reasoning_effort": "none", "temperature": 0})
    dt = time.perf_counter() - t
    check("client disconnect frees the model", s == 200 and dt < 5, f"next request done in {dt:.1f} s")

    # ------------------------------------------- turn-2 latency, long document
    doc = (ROOT / "README.md").read_text()[:24000]
    q1 = f"Here is a document:\n\n{doc}\n\nSummarize it in two sentences."
    base = {"temperature": 0, "max_tokens": 80, "reasoning_effort": "none"}

    def turn2_chat(first_user):
        s, r = call("POST", "/v1/chat/completions", {**base, "messages": [{"role": "user", "content": first_user}]})
        hist = [{"role": "user", "content": first_user}, r["choices"][0]["message"], {"role": "user", "content": "What is the KV cache default?"}]
        ev, ttft = stream("/v1/chat/completions", {**base, "messages": hist, "stream_options": {"include_usage": True}})
        u = next(d["usage"] for _, d in ev if isinstance(d, dict) and d.get("usage"))
        return ttft, u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"]

    cold_ttft = None
    hist_cold = [{"role": "user", "content": q1 + " "}, {"role": "assistant", "content": "It describes an MLX runtime."},
                 {"role": "user", "content": "What is the KV cache default?"}]
    ev, cold_ttft = stream("/v1/chat/completions", {**base, "messages": hist_cold, "stream_options": {"include_usage": True}})
    u_cold = next(d["usage"] for _, d in ev if isinstance(d, dict) and d.get("usage"))
    chat_ttft, chat_prompt, chat_cached = turn2_chat(q1)
    s, r = call("POST", "/v1/responses", {"input": q1 + "  ", "temperature": 0, "max_output_tokens": 80, "reasoning": {"effort": "none"}})
    ev, resp_ttft = stream("/v1/responses", {"input": "What is the KV cache default?", "previous_response_id": r["id"],
                                             "temperature": 0, "max_output_tokens": 80, "reasoning": {"effort": "none"}})
    ru = ev[-1][1]["response"]["usage"]
    print(f"\nturn 2 of a {u_cold['prompt_tokens']}-token conversation, time to first token:")
    print(f"  cold (no reuse)                         {cold_ttft:6.2f} s  ({u_cold['prompt_tokens_details']['cached_tokens']} cached)")
    print(f"  chat completions, history resent        {chat_ttft:6.2f} s  ({chat_cached} of {chat_prompt} cached)")
    print(f"  responses, previous_response_id         {resp_ttft:6.2f} s  ({ru['input_tokens_details']['cached_tokens']} of {ru['input_tokens']} cached)")
    check("turn 2 with reuse is faster than cold", chat_ttft < cold_ttft / 3 and resp_ttft < cold_ttft / 3)
finally:
    server.send_signal(2)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
