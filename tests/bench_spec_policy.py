"""End-to-end decode speed: plain vs DSpark with the geometric BlockPolicy vs DSpark with
the confidence-gated policy (generate.SPEC_POLICY). One model load; modes alternate per
rep and the best of the reps is kept. 300-token generations, T = 0 and T = 1 (seeded).

    python tests/bench_spec_policy.py MODEL_DIR [reps]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inkling2_mini.generate as G  # noqa: E402

reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
bot = G.Inkling2Mini(sys.argv[1], verbose=False)
prompts = [
    ("story", "Write a 200-word story about a lighthouse keeper.", (0.0, 1.0)),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and tests.", (0.0,)),
    ("code2", "Implement a binary search tree in Python with insert, delete and in-order traversal.", (0.0,)),
    ("zh", "写一段 300 字左右的秋天散文。", (0.0, 1.0)),
    ("zh_long", "以“归乡”为题，写一篇 800 字左右的短篇小说，人物要有对话。", (0.0, 1.0)),
]
modes = ("plain", "geometric", "confidence")
best, tpr = {}, {}
for _ in range(reps):
    for name, text, temps in prompts:
        ids, _, _ = bot.encode([{"role": "user", "content": text}], reasoning_effort="none")
        for temp in temps:
            for mode in modes:
                G.SPEC_POLICY = mode if mode != "plain" else G.SPEC_POLICY
                st = None
                for _, s in bot.generate_tokens(ids, max_tokens=300, temperature=temp, seed=0,
                                                speculative=mode != "plain"):
                    st = s or st
                k = (name, temp, mode)
                if st["gen_tps"] > best.get(k, 0):
                    best[k] = st["gen_tps"]
                    tpr[k] = st.get("tokens_per_round", 0)
G.SPEC_POLICY = "confidence"
print(f"{'prompt':12s} {'plain':>7s} {'geometric':>16s} {'confidence':>16s}   conf vs geo   conf vs plain")
for name, _, temps in prompts:
    for temp in temps:
        p, g, c = (best[(name, temp, m)] for m in modes)
        print(f"{name + f'@T{temp:g}':12s} {p:7.1f} {g:7.1f} ({tpr[(name, temp, 'geometric')]:.2f}/rd) "
              f"{c:7.1f} ({tpr[(name, temp, 'confidence')]:.2f}/rd)   {c / g - 1:+6.1%}       {c / p - 1:+6.1%}")
