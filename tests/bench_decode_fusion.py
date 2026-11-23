"""End-to-end decode speed with the decode glue fusions (decode_kernels) on and off,
plain and DSpark, greedy, 300 tokens. Runs alternate on/off to share system noise.

    python tests/bench_decode_fusion.py MODEL_DIR [reps]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini import decode_kernels as DK  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
bot = Inkling2Mini(sys.argv[1])
prompts = [
    ("story", "Write a 200-word story about a lighthouse keeper.", "none"),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and tests.", "none"),
    ("zh", "写一段 300 字左右的秋天散文。", "none"),
]
for name, text, effort in prompts:
    ids, _, _ = bot.encode([{"role": "user", "content": text}], reasoning_effort=effort)
    best = {}
    for _ in range(reps):
        for spec in (False, True):
            for on in (False, True):
                DK.ENABLED = on
                stats = None
                for _, st in bot.generate_tokens(ids, max_tokens=300, temperature=0.0, speculative=spec):
                    stats = st or stats
                k = (spec, on)
                best[k] = max(best.get(k, 0), stats["gen_tps"])
    DK.ENABLED = True
    for spec in (False, True):
        a, b = best[(spec, False)], best[(spec, True)]
        print(f"{name:6s} {'DSpark' if spec else 'plain ':6s}  fusions off {a:5.1f}  on {b:5.1f} tok/s  ({b / a - 1:+.1%})")
