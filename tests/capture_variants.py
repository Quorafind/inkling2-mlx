"""Which target capture points give the DSpark draft its best acceptance?
    python tests/capture_variants.py MODEL_DIR
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1])
prompts = [
    ("story", "Write a 150-word story about a lighthouse keeper.", "none"),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.", "none"),
    ("zh", "用三句话解释一下什么是 looped transformer。", "low"),
]
variants = {
    "inputs of visits 5,23,41 (blog)": [5, 23, 41, 48],
    "outputs of visits 5,23,41": [6, 24, 42, 48],
    "1-indexed visits": [4, 22, 40, 48],
}
for label, ids_ in variants.items():
    bot.drafter.config.target_layer_ids = ids_
    rows = []
    for name, text, effort in prompts:
        ids, _, _ = bot.encode([{"role": "user", "content": text}], reasoning_effort=effort)
        stats = None
        for _, st in bot.generate_tokens(ids, max_tokens=200, temperature=0, block_size=8):
            stats = st or stats
        rows.append(f"{name} {stats['tokens_per_round']:.2f}")
    print(f"{label:34s} tok/round: " + "  ".join(rows))
