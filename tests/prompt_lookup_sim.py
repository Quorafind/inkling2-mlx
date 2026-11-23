"""How much would prompt-lookup (n-gram) drafting accept on a compaction task?
Runs a real compaction (greedy, DSpark on for comparison), then simulates drafting
from the context offline.  python tests/prompt_lookup_sim.py MODEL_DIR"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1])
doc = open(Path(__file__).resolve().parents[1] / "README.md").read()
prompt = doc + "\n\n---\nCompact the document above into a detailed summary for a colleague who will continue this work. Keep every number, file name, flag and command exactly."
ids, _, _ = bot.encode([{"role": "user", "content": prompt}], reasoning_effort="none")
toks, stats = [], None
for t, st in bot.generate_tokens(ids, max_tokens=700, temperature=0):
    toks.append(t)
    stats = st or stats
ctx = ids[0].tolist()
print(f"context {len(ctx)} tokens, summary {len(toks)} tokens, DSpark {stats.get('tokens_per_round', 0):.2f} tok/round, "
      f"{stats['gen_tps']:.1f} tok/s")


def simulate(k=8, nmax=4, nmin=2):
    seq = ctx + toks
    base = len(ctx)
    i, rounds, drafted_rounds, accepted_total = 0, 0, 0, 0
    while i < len(toks):
        pos = base + i
        best = None
        for n in range(nmax, nmin - 1, -1):  # longest suffix match, most recent occurrence
            if pos < n:
                continue
            key = seq[pos - n:pos]
            for j in range(pos - n - 1, -1, -1):
                if seq[j:j + n] == key:
                    best = j + n
                    break
            if best is not None:
                break
        acc = 0
        if best is not None:
            drafted_rounds += 1
            prop = seq[best:best + k]
            while acc < len(prop) and i + acc < len(toks) and prop[acc] == toks[i + acc]:
                acc += 1
        accepted_total += acc
        i += acc + 1
        rounds += 1
    return len(toks) / rounds, accepted_total / len(toks), drafted_rounds / rounds


for k in (4, 8, 16):
    tpr, cov, dr = simulate(k)
    print(f"prompt lookup, k={k:2d}: {tpr:.2f} tok/round, {cov:.0%} of tokens come from accepted drafts, "
          f"a match exists in {dr:.0%} of rounds")
