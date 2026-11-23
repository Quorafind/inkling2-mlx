"""Continuous batching in the model worker (inkling2_mini/server_worker.py), driven over
its JSON-lines protocol like the Rust launcher does.

1. Six requests at once (questions with known answers, plus long essays that keep the
   batch busy): all answer correctly, several decode in one batch.
2. A follow-up turn on a conversation that finished inside the batch: the stored state
   (extracted from the batch caches) must hold the reply, so a question about it is
   answered from the KV (and cached_tokens covers it).
3. A long request decoding alone gets company mid-stream: it is paused, joins a batch,
   goes back to decoding alone when the other finishes, and its follow-up turn still
   reads its own list back correctly.
4. Cancel and a stop sequence inside a running batch.

    python tests/check_server_batching.py MODEL_DIR
"""

import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
proc = subprocess.Popen(
    [sys.executable, "-m", "inkling2_mini.server_worker", "--model", sys.argv[1], *sys.argv[2:]],
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


def wait(rid, kind="done", count=1, timeout=900):
    end = time.time() + timeout
    with cond:
        while sum(e["type"] in (kind, "error", "fatal") for e in events.get(rid, [])) < count:
            cond.wait(max(0.1, end - time.time()))
            if time.time() > end:
                raise TimeoutError(rid)
    evs = events[rid]
    err = [e for e in evs if e["type"] in ("error", "fatal")]
    if err:
        raise RuntimeError(f"{rid}: {err[0]['message']}")
    return evs


def generate(rid, messages, **kw):
    send({"op": "generate", "id": rid, "messages": messages, "temperature": 0.0, "max_tokens": 200,
          "effort": "none", **kw})


def done(rid):
    evs = wait(rid)
    return next(e for e in evs if e["type"] == "start"), evs[-1]


def user(text):
    return [{"role": "user", "content": text}]


wait(None, "ready")
ok = True


def check(name, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"  {'OK  ' if good else 'FAIL'} {name}{': ' + detail if detail else ''}")


# 1-2. Six at once.
print("six requests at once")
qa = {
    "q1": ("What is the capital of France? Answer with one word.", "paris"),
    "q2": ("What is 17 + 25? Answer with just the number.", "42"),
    "q3": ("Which planet is the largest in the solar system? One word.", "jupiter"),
    "q4": ("Write a numbered list of 8 fruits, one per line, nothing else.", None),  # thinking first
}
essays = {"e1": "Write a long, detailed essay about lighthouses.", "e2": "Write a long, detailed essay about glaciers."}
t0 = time.perf_counter()
for rid, (q, _) in qa.items():
    generate(rid, user(q), max_tokens=400)
for rid, q in essays.items():
    generate(rid, user(q), max_tokens=300)
results = {rid: done(rid) for rid in [*qa, *essays]}
wall = time.perf_counter() - t0
tokens = sum(d["usage"]["completion_tokens"] for _, d in results.values())
for rid, (q, want) in qa.items():
    d = results[rid][1]
    if want is not None:
        check(f"{rid} {q[:40]!r}", want in d["text"].lower(), f"{d['text'][:60]!r} (batch {d['stats']['max_batch']})")
lines = [ln for ln in results["q4"][1]["text"].splitlines() if re.match(r"\s*\d+[.)]", ln)]
check("q4 list", len(lines) >= 8, f"{len(lines)} numbered lines (batch {results['q4'][1]['stats']['max_batch']})")
peak = max(d["stats"]["max_batch"] for _, d in results.values())
check("several requests decoded in one batch", peak >= 4, f"largest batch {peak}")
print(f"  {tokens} tokens in {wall:.1f} s: {tokens / wall:.1f} tok/s in total")

# 2. Follow-up on a reply that finished inside the batch.
fruit = [ln.split(".", 1)[-1].split(")", 1)[-1].strip().lower() for ln in lines]
hist = user(qa["q4"][0]) + [{"role": "assistant", "content": results["q4"][1]["text"]}]
generate("q4b", hist + user("What was fruit number 3 in your list? Answer with just the name."), max_tokens=400)
s, d = done("q4b")
answer = d["text"].strip().lower()
check("follow-up reads the batched reply back", len(fruit) >= 3 and fruit[2].rstrip(".") in answer,
      f"list item 3 {fruit[2] if len(fruit) > 2 else None!r}, answer {answer!r}")
check("follow-up reused the stored state", s["cached_tokens"] >= results["q4"][0]["prompt_tokens"]
      + results["q4"][1]["usage"]["completion_tokens"] - 2,
      f"{s['cached_tokens']} of {s['prompt_tokens']} prompt tokens")

# 3. Company mid-stream.
print("a lone request gets company mid-stream")
generate("long", user("Write a numbered list of 40 different animals, one per line, nothing else."), max_tokens=1500)
wait("long", "delta", count=40)
generate("short", user("Name three primary colors, comma separated."), max_tokens=400)
_, ds = done("short")
_, dl = done("long")
st = dl["stats"]
check("the short request joined a batch", ds["stats"]["max_batch"] == 2, f"batch {ds['stats']['max_batch']}")
check("the long request decoded in the batch and alone",
      0 < st["batched_tokens"] < dl["usage"]["completion_tokens"],
      f"{st['batched_tokens']} of {dl['usage']['completion_tokens']} tokens batched")
animals = [ln.split(".", 1)[-1].strip().lower() for ln in dl["text"].splitlines() if re.match(r"\s*\d+\.", ln)]
hist = user("Write a numbered list of 40 different animals, one per line, nothing else.") + \
    [{"role": "assistant", "content": dl["text"]}]
for n in (5, 30):
    generate(f"long{n}", hist + user(f"What was animal number {n} in your list? Answer with just the name."),
             max_tokens=400)
    s, d = done(f"long{n}")
    want = animals[n - 1].rstrip(".") if len(animals) >= n else None
    check(f"follow-up: animal {n}", want is not None and want in d["text"].lower(),
          f"list {want!r}, answer {d['text'].strip()!r}, {s['cached_tokens']} cached")

# 4. Cancel and stop inside a batch.
print("cancel and stop inside a batch")
generate("c1", user("Write a very long story about the sea."), max_tokens=600)
generate("c2", user("List ten countries, one per line, nothing else."), max_tokens=500)
generate("c3", user("List five vegetables, one per line."), max_tokens=500, stop=["\n"])
wait("c1", "delta", count=20)
wait("c2", "delta", count=5)
t = time.perf_counter()
send({"op": "cancel", "id": "c1"})
_, d1 = done("c1")
dt = time.perf_counter() - t
_, d2 = done("c2")
_, d3 = done("c3")
check("cancel", d1["finish_reason"] == "cancelled" and dt < 2.0, f"after {dt:.2f} s")
check("the other rows go on", d2["finish_reason"] == "stop" and len(d2["text"].splitlines()) >= 10,
      f"{d2['usage']['completion_tokens']} tokens, batch {d2['stats']['max_batch']}")
check("stop sequence", "\n" not in d3["text"] and d3["finish_reason"] == "stop", repr(d3["text"]))

send({"op": "stats", "id": "stats"})
print("worker:", {k: v for k, v in wait("stats", "stats")[-1].items() if k not in ("type", "id")})
send({"op": "shutdown"})
proc.wait(timeout=30)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
