"""Throughput of the model worker with N concurrent requests (continuous batching).

N essay requests are sent at once (greedy, 300 tokens each). Reported: total generated
tokens over the wall time from the first request to the last reply, the median time to
first token, and the steady-state rate while all N rows decode together.

    python tests/bench_server_batch.py MODEL_DIR [N ...] [-- worker args]
"""

import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
args = sys.argv[2:]
extra = args[args.index("--") + 1:] if "--" in args else []
sizes = [int(a) for a in (args[: args.index("--")] if "--" in args else args)] or [1, 2, 4, 8]
proc = subprocess.Popen([sys.executable, "-m", "inkling2_mini.server_worker", "--model", sys.argv[1],
                         "--max-batch", str(max(sizes)), *extra],
                        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
events, cond = {}, threading.Condition()


def reader():
    for line in proc.stdout:
        ev = json.loads(line)
        ev["t"] = time.perf_counter()
        with cond:
            events.setdefault(ev.get("id"), []).append(ev)
            cond.notify_all()


threading.Thread(target=reader, daemon=True).start()


def send(o):
    proc.stdin.write(json.dumps(o) + "\n")
    proc.stdin.flush()


def wait(rid, kind="done"):
    with cond:
        while not any(e["type"] in (kind, "error") for e in events.get(rid, [])):
            cond.wait(1)
    return events[rid]


TOPICS = ["lighthouses", "volcanoes", "the printing press", "honeybees", "black holes", "the Silk Road",
          "coral reefs", "the steam engine", "glaciers", "octopuses", "the Roman aqueducts", "tea",
          "comets", "the violin", "deserts", "bridges"]
wait(None, "ready")
for n in sizes:
    ids = [f"b{n}-{i}" for i in range(n)]
    t0 = time.perf_counter()
    for i, rid in enumerate(ids):
        send({"op": "generate", "id": rid, "temperature": 0.0, "max_tokens": 300, "effort": "none",
              "messages": [{"role": "user", "content": f"Write a detailed 400-word essay about {TOPICS[i % 16]}."}]})
    evs = {rid: wait(rid) for rid in ids}
    wall = time.perf_counter() - t0
    done = {rid: e[-1] for rid, e in evs.items()}
    tokens = sum(d["usage"]["completion_tokens"] for d in done.values())
    first = [next(e["t"] for e in evs[rid] if e["type"] in ("delta", "tool_call")) - t0 for rid in ids]
    # steady state: from the last first token to the first reply
    t_all = max(first) + t0
    t_end = min(e[-1]["t"] for e in evs.values())
    steady = sum(sum(1 for e in evs[rid] if e["type"] == "delta" and t_all <= e["t"] <= t_end) for rid in ids)
    rate = steady / (t_end - t_all) if t_end > t_all else 0.0
    gen = statistics.median(d["stats"]["gen_tps"] for d in done.values())
    print(f"N={n:2d}: {tokens:5d} tokens in {wall:5.1f} s -> {tokens / wall:5.1f} tok/s overall; "
          f"all {n} decoding: ~{rate:5.1f} deltas/s; median TTFT {statistics.median(first):4.1f} s; "
          f"per request {gen:4.1f} tok/s; largest batch {max(d['stats']['max_batch'] for d in done.values())}")
send({"op": "shutdown"})
proc.wait(timeout=30)
