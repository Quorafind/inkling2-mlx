"""Screened DSpark drafting (speculative.DSparkDrafter.markov_screened) vs the
full-vocabulary Markov loop, on the same draft features every round of real
generations (prose and code, T = 0 and 1).

Reports per-position agreement of the drafted tokens, the confidence difference, how
often the full loop's token falls outside the shortlist, and the time of each Markov
loop after the shared draft layers + vocabulary head. The draft only proposes tokens,
so disagreement can lower acceptance but never changes the output.

    python tests/check_screened_draft.py MODEL_DIR [CANDIDATES [K]]   (default 4096 16)
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402
from inkling2_mini.speculative import chunk_topk  # noqa: E402

bot = Inkling2Mini(sys.argv[1], verbose=False)
dr = bot.drafter
SCREEN = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
SCREEN_K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
rec = []


def compare(last_bonus, hidden, cache, block_size, screen=None):
    feats = dr.draft_features(last_bonus, hidden, cache, block_size)
    mx.eval(*feats)
    t0 = time.perf_counter()
    full = dr.markov_full(*feats, return_logits=True)
    mx.eval(full)
    t1 = time.perf_counter()
    scr = dr.markov_screened(*feats, screen=SCREEN, k=SCREEN_K)
    mx.eval(scr)
    t2 = time.perf_counter()
    static = feats[0] + dr.markov_head.fold_q.bias.astype(feats[0].dtype)
    _, ids = chunk_topk(static, dr.config.unpadded_vocab_size, SCREEN // SCREEN_K, SCREEN_K)
    ft = full[0].reshape(-1).tolist()
    stoks = scr[0].reshape(-1).tolist()
    in_list = [ft[i] in set(ids[i].tolist()) for i in range(len(ft))]
    # Full-vocabulary BF16 logit gap between the full loop's token and the screened one
    # (0 = a BF16 tie). Only meaningful while the drafted prefixes agree.
    rows = full[2]
    gap = [float(rows[i, ft[i]] - rows[i, stoks[i]]) for i in range(len(ft))]
    rec.append(dict(full_t=ft, scr_t=stoks, full_c=full[1].reshape(-1).tolist(),
                    scr_c=scr[1].reshape(-1).tolist(), in_list=in_list, gap=gap,
                    t_full=t1 - t0, t_scr=t2 - t1))
    return scr


dr.draft_with_confidence = compare
prompts = [
    ("story", "Write a 200-word story about a lighthouse keeper."),
    ("zh", "写一段 300 字左右的秋天散文。"),
    ("code", "Implement a binary search tree in Python with insert, delete and in-order traversal."),
    ("code2", "Write a Rust function that parses a CSV line into a Vec<String>, handling quoted fields."),
]
for name, text in prompts:
    ids, _, _ = bot.encode([{"role": "user", "content": text}], reasoning_effort="none")
    for temp in (0.0, 1.0):
        for _ in bot.generate_tokens(ids, max_tokens=300, temperature=temp, seed=0):
            pass

ft = np.array([r["full_t"] for r in rec])
st = np.array([r["scr_t"] for r in rec])
fc = np.array([r["full_c"] for r in rec])
sc = np.array([r["scr_c"] for r in rec])
inl = np.array([r["in_list"] for r in rec])
# Position i only matters if positions < i agree (later drafts condition on earlier ones).
prefix_ok = np.cumprod(np.concatenate([np.ones((len(rec), 1), bool), ft[:, :-1] == st[:, :-1]], axis=1), axis=1).astype(bool)
agree = (ft == st)[prefix_ok].mean()
first = (ft[:, 0] == st[:, 0]).mean()
same = prefix_ok & (ft == st)
dc = np.abs(fc - sc)[same]
gap = np.array([r["gap"] for r in rec])
differ = prefix_ok & (ft != st)
ties = differ & (gap <= 0.0)
# Agreement up to BF16 ties of the reference loop: those are arbitrary in the reference.
agree_t = (same | ties)[prefix_ok].mean()
first_t = (same | ties)[:, 0].mean()
print(f"{len(rec)} rounds")
print(f"draft token agreement: position 1 {first:.2%}; all positions with an agreeing prefix {agree:.2%}")
print(f"  of the {differ.sum()} differing positions, {ties.sum()} are BF16 ties in the reference loop "
      f"(full-vocab logit gap 0); real gaps: {np.sort(gap[differ & ~ties])[::-1][:8].round(3).tolist()}")
print(f"agreement up to reference ties: position 1 {first_t:.2%}; all {agree_t:.2%}")
print("per position (agreeing prefix): " + " ".join(
    f"{i + 1}:{(same | ties)[prefix_ok[:, i], i].mean():.0%}" for i in range(ft.shape[1])))
print(f"full loop's token in the shortlist: {inl[prefix_ok].mean():.2%}")
print(f"confidence |full - screened| where tokens agree: mean {dc.mean():.2e}, p99 {np.quantile(dc, 0.99):.2e}, max {dc.max():.2e}")
tf = np.median([r["t_full"] for r in rec]) * 1e3
ts = np.median([r["t_scr"] for r in rec]) * 1e3
print(f"Markov loop (median): full {tf:.2f} ms, screened {ts:.2f} ms ({tf - ts:+.2f} ms per round saved)")
ok = first_t >= 0.99 and agree_t >= 0.98
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
