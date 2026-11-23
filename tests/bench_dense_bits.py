"""End-to-end decode speed and resident memory at --dense-bits 8 vs 6 (or the bits given),
plain and DSpark, greedy, 300 tokens. Both models don't fit at once, so each rep loads them
in turn (8, 6, 8, 6, ...) and the best of the reps is kept.

    python tests/bench_dense_bits.py MODEL_DIR [reps] [bits ...]
"""

import gc
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
bits_list = [int(b) for b in sys.argv[3:]] or [8, 6]
prompts = [
    ("story", "Write a 200-word story about a lighthouse keeper."),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and tests."),
    ("zh", "写一段 300 字左右的秋天散文。"),
]
best, mem = {}, {}
for _ in range(reps):
    for bits in bits_list:
        bot = Inkling2Mini(sys.argv[1], dense_bits=bits, verbose=False)
        mem[bits] = mx.get_active_memory() / 1e9
        for name, text in prompts:
            ids, _, _ = bot.encode([{"role": "user", "content": text}], reasoning_effort="none")
            for spec in (False, True):
                stats = None
                for _, st in bot.generate_tokens(ids, max_tokens=300, temperature=0.0, speculative=spec):
                    stats = st or stats
                k = (name, spec, bits)
                best[k] = max(best.get(k, 0), stats["gen_tps"])
        del bot
        gc.collect()
        mx.clear_cache()
print("resident after load: " + ", ".join(f"dense {b}: {mem[b]:.2f} GB" for b in bits_list))
base = bits_list[0]
for name, _ in prompts:
    for spec in (False, True):
        row = "  ".join(f"{b}-bit {best[(name, spec, b)]:5.1f}" for b in bits_list)
        gain = "  ".join(f"({best[(name, spec, b)] / best[(name, spec, base)] - 1:+.1%})" for b in bits_list[1:])
        print(f"{name:6s} {'DSpark' if spec else 'plain ':6s}  {row} tok/s  {gain}")
