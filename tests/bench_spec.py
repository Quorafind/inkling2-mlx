"""Decode speed: plain vs adaptive DSpark vs fixed block 8 (greedy and sampled).

    python tests/bench_spec.py MODEL_DIR [--dense-bits N]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bits = int(sys.argv[sys.argv.index("--dense-bits") + 1]) if "--dense-bits" in sys.argv else 6
bot = Inkling2Mini(sys.argv[1], dense_bits=bits)
prompts = [
    ("story", "Write a 200-word story about a lighthouse keeper.", "none"),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and tests.", "none"),
    ("zh-think", "用三句话解释一下什么是 looped transformer。", "low"),
    ("list", "List the 20 largest countries by area, one per line as 'name - area km2'.", "none"),
]
configs = [("plain", dict(speculative=False)), ("adaptive", {}), ("block-8", dict(block_size=8))]
print(f"dense_bits={bits}")
for temp in (0.0, 1.0):
    for name, text, effort in prompts:
        ids, _, _ = bot.encode([{"role": "user", "content": text}], reasoning_effort=effort)
        row = []
        for label, kw in configs:
            stats = None
            for _, st in bot.generate_tokens(ids, max_tokens=300, temperature=temp, seed=0, **kw):
                stats = st or stats
            extra = f" ({stats['tokens_per_round']:.1f}/rd)" if "tokens_per_round" in stats else ""
            row.append(f"{label} {stats['gen_tps']:5.1f}{extra}")
        print(f"T={temp:.0f} {name:9s} " + " | ".join(row))
