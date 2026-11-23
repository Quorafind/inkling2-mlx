"""The launcher's model worker (inkling2_mini/server_worker.py), driven over its
JSON-lines protocol like the Rust launcher does.

1. Turn 2 resends the whole history with the assistant message as an OpenAI client
   echoes it (no reasoning_content): the worker must reuse turn 1's state
   (cached_tokens > 0) and prefill only the new message.
2. The same turn-2 request twice: each clones the stored state, so greedy output is
   identical token for token (cloning leaves the stored state untouched).
3. A long system prompt is snapshotted on a miss and reused by another conversation.
4. Tool calls come back parsed; stop sequences cut the text; cancel stops a generation.

    python tests/check_server_worker.py MODEL_DIR
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
proc = subprocess.Popen(
    [sys.executable, "-m", "inkling2_mini.server_worker", "--model", sys.argv[1]],
    cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
)
events = {}
cond = threading.Condition()


def reader():
    for line in proc.stdout:
        ev = json.loads(line)
        with cond:
            events.setdefault(ev.get("id"), []).append(ev)
            cond.notify_all()


threading.Thread(target=reader, daemon=True).start()


def send(obj):
    proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def wait(rid, kind="done", timeout=600):
    end = time.time() + timeout
    with cond:
        while not any(e["type"] in (kind, "error", "fatal") for e in events.get(rid, [])):
            cond.wait(max(0.1, end - time.time()))
            if time.time() > end:
                raise TimeoutError(rid)
    return events[rid]


def run(rid, **req):
    t = time.perf_counter()
    send({"op": "generate", "id": rid, "temperature": 0.0, "max_tokens": 120, **req})
    evs = wait(rid)
    err = [e for e in evs if e["type"] in ("error", "fatal")]
    if err:
        raise RuntimeError(err[0]["message"])
    start = next(e for e in evs if e["type"] == "start")
    done = evs[-1]
    first = next((e for e in evs if e["type"] in ("delta", "tool_call")), None)
    return start, done, time.perf_counter() - t


wait(None, "ready")
ok = True

# 1-2. Multi-turn reuse with an echoed assistant message, twice.
q1 = [{"role": "user", "content": "用两句话介绍一下杭州。"}]
s1, d1, t1 = run("t1", messages=q1, effort="low")
history = q1 + [{"role": "assistant", "content": d1["text"]}]  # no reasoning_content, like OpenAI clients
q2 = history + [{"role": "user", "content": "再用一句话说说西湖。"}]
s2, d2, t2 = run("t2", messages=q2, effort="low")
s3, d3, t3 = run("t3", messages=q2, effort="low")
print(f"turn 1: {s1['prompt_tokens']} prompt tokens, {d1['usage']['completion_tokens']} generated "
      f"({d1['usage']['reasoning_tokens']} reasoning), {t1:.2f} s")
print(f"turn 2: {s2['prompt_tokens']} prompt tokens, {s2['cached_tokens']} reused, "
      f"{d2['stats']['prefill_tokens']} prefilled, {t2:.2f} s")
reuse = s2["cached_tokens"] > s1["prompt_tokens"]
same = d2["text"] == d3["text"] and d2["reasoning"] == d3["reasoning"]
print(f"  reuse across turns (reasoning re-attached): {'OK' if reuse else 'FAIL'}")
print(f"  two clones of the stored state give identical output: {'OK' if same else 'FAIL'}")
ok &= reuse and same

# 3. System-prompt snapshot shared by two conversations.
system = "You are a meticulous assistant for a hardware lab. " + " ".join(
    f"Rule {i}: always state units, cite the instrument, and keep answers short." for i in range(60))
sa, da, ta = run("sa", messages=[{"role": "system", "content": system}, {"role": "user", "content": "What is 2+2?"}],
                 effort="none", max_tokens=20)
sb, db, tb = run("sb", messages=[{"role": "system", "content": system}, {"role": "user", "content": "Name a color."}],
                 effort="none", max_tokens=20)
snap = sb["cached_tokens"] > 500
print(f"system prompt snapshot: conversation B reused {sb['cached_tokens']} of {sb['prompt_tokens']} tokens "
      f"({tb:.2f} s vs {ta:.2f} s): {'OK' if snap else 'FAIL'}")
ok &= snap

# 4. Tool call, stop sequence, cancel.
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
st, dt, _ = run("tool", messages=[{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
                tools=tools, effort="none")
call = dt["tool_calls"][0] if dt["tool_calls"] else None
tool_ok = call is not None and call["name"] == "get_weather" and isinstance(call["arguments"], dict) \
    and dt["finish_reason"] == "tool_calls"
print(f"tool call: {call} finish={dt['finish_reason']}: {'OK' if tool_ok else 'FAIL'}")
ok &= tool_ok

_, ds, _ = run("stop", messages=[{"role": "user", "content": "List three fruits, one per line."}],
               effort="none", stop=["\n"])
stop_ok = "\n" not in ds["text"] and ds["finish_reason"] == "stop"
print(f"stop sequence: {ds['text']!r} finish={ds['finish_reason']}: {'OK' if stop_ok else 'FAIL'}")
ok &= stop_ok

send({"op": "generate", "id": "long", "temperature": 0.0, "max_tokens": 2000, "effort": "none",
      "messages": [{"role": "user", "content": "Write a very long story about the sea."}]})
wait("long", "delta")
t = time.perf_counter()
send({"op": "cancel", "id": "long"})
dl = wait("long")[-1]
cancel_ok = dl["finish_reason"] == "cancelled" and time.perf_counter() - t < 2.0
print(f"cancel: finish={dl['finish_reason']} after {time.perf_counter() - t:.2f} s: {'OK' if cancel_ok else 'FAIL'}")
ok &= cancel_ok

send({"op": "stats", "id": "stats"})
print("cache:", {k: v for k, v in wait("stats", "stats")[-1].items() if k not in ("type", "id")})
send({"op": "shutdown"})
proc.wait(timeout=30)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
