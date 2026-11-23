"""Trace DSpark rounds at a fixed block of 8: per-position draft confidence, accepted count,
draft time and verify time; plus a verify-cost curve V(L) and the plain step time.

    python tests/trace_spec.py MODEL_DIR OUT.json   (input for tests/sim_spec_policy.py)
"""

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402

import inkling2_mini.speculative as S  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1], verbose=True)
drafter = bot.drafter
lm = bot.model.language_model

rec = {"probs": [], "margin": [], "ent": []}


def rec_sampler(logits):
    lf = logits.astype(mx.float32)
    lp = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
    p = mx.exp(lp)
    top2 = mx.topk(lp, 2, axis=-1)  # ascending order
    rec["probs"].append(mx.max(p, axis=-1))
    rec["margin"].append(top2[..., 1] - top2[..., 0])
    rec["ent"].append(-mx.sum(p * lp, axis=-1))
    return mx.argmax(lf, axis=-1)


S.greedy = rec_sampler

rounds = []
cur = {}
orig_draft = drafter.draft_block


def timed_draft(*a, **k):
    rec["probs"].clear(); rec["margin"].clear(); rec["ent"].clear()
    t = time.perf_counter()
    out = orig_draft(*a, **k)
    mx.eval(out, *rec["probs"], *rec["margin"], *rec["ent"])
    cur["draft_s"] = time.perf_counter() - t
    cur["probs"] = [float(x.item()) for x in rec["probs"]]
    cur["margin"] = [float(x.item()) for x in rec["margin"]]
    cur["ent"] = [float(x.item()) for x in rec["ent"]]
    cur["t_end_draft"] = time.perf_counter()
    return out


drafter.draft_block = timed_draft
orig_observe = S.BlockPolicy.observe


def observe(self, bs, accepted, seconds):
    rounds.append(dict(run=cur["run"], bs=bs, acc=accepted, round_s=seconds, draft_s=cur["draft_s"],
                       probs=cur["probs"], margin=cur["margin"], ent=cur["ent"]))
    return orig_observe(self, bs, accepted, seconds)


S.BlockPolicy.observe = observe

prompts = [
    ("story", "Write a 200-word story about a lighthouse keeper.", "none"),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and tests.", "none"),
    ("zh", "写一段 300 字左右的秋天散文。", "none"),
    ("zh_long", "以“归乡”为题，写一篇 800 字左右的短篇小说，人物要有对话。", "none"),
    ("zh_think", "写一段 200 字左右的散文，主题是城市里的雨夜。", "low"),
]
runs = []
for name, text, effort in prompts:
    ids, _, _ = bot.encode([{"role": "user", "content": text}], reasoning_effort=effort)
    for temp in (0.0, 1.0):
        if name == "code" and temp == 1.0:
            continue
        cur["run"] = f"{name}@T{temp:g}"
        n0 = len(rounds)
        stats = None
        for _, st in bot.generate_tokens(ids, max_tokens=400, temperature=temp, seed=0, block_size=8):
            stats = st or stats
        rs = rounds[n0:]
        tpr = sum(r["acc"] + 1 for r in rs) / len(rs)
        print(f"{cur['run']:16s} rounds {len(rs):3d}  tok/round {tpr:4.2f}  {stats['gen_tps']:5.1f} tok/s  "
              f"draft {1e3 * sum(r['draft_s'] for r in rs) / len(rs):5.1f} ms  "
              f"round {1e3 * sum(r['round_s'] for r in rs) / len(rs):5.1f} ms", flush=True)
        runs.append(dict(run=cur["run"], gen_tps=stats["gen_tps"], tokens=stats["gen_tokens"]))

# Plain step time on a warm cache (pipelined, as in the plain loop).
ids, _, _ = bot.encode([{"role": "user", "content": prompts[0][1]}], reasoning_effort="none")
best = 0
for _ in range(2):
    st = None
    for _, s in bot.generate_tokens(ids, max_tokens=200, temperature=0.0, speculative=False):
        st = s or st
    best = max(best, st["gen_tps"])
plain_ms = 1e3 / best
print(f"plain {best:.1f} tok/s = {plain_ms:.1f} ms/token")

# Verify cost curve V(L): target forward over L tokens with capture + logits + sampling.
from mlx_vlm.speculative.cache_state import start_speculative_cache  # noqa: E402

state = bot.new_state()
for _ in bot.generate_tokens(ids, max_tokens=64, temperature=0.0, state=state, speculative=True):
    pass
cache = state.cache
capture = list(drafter.config.target_layer_ids)
text = mx.array(bot.tokenizer.encode(" The keeper walked down to the shore every morning and watched the waves roll in."))
verify_ms = {}
for L in range(1, 9):
    times = []
    for rep in range(6):
        tx = start_speculative_cache(cache, L)
        t = time.perf_counter()
        out, caps = lm.model(text[None, :L], cache=cache, capture=capture)
        tgt = mx.argmax(lm.logits(out), axis=-1)
        feats = mx.concatenate(caps, axis=-1)
        mx.eval(tgt, feats)
        times.append(time.perf_counter() - t)
        tx.abort()
    verify_ms[L] = 1e3 * min(times[1:])
    print(f"V({L}) = {verify_ms[L]:6.1f} ms", flush=True)

json.dump(dict(rounds=rounds, runs=runs, plain_ms=plain_ms, verify_ms=verify_ms), open(sys.argv[2], "w"))
