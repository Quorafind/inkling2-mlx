"""Aggregate decode throughput vs batch size (plain batched decoding).

    python tests/bench_batch.py MODEL_DIR [batch sizes...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

TOPICS = ["lighthouses", "volcanoes", "the printing press", "honeybees", "black holes", "the Silk Road",
          "coral reefs", "the steam engine", "glaciers", "octopuses", "the Roman aqueducts", "tea",
          "comets", "the violin", "deserts", "bridges"]
bot = Inkling2Mini(sys.argv[1], speculative=False)
sizes = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8, 16]
for B in sizes:
    convs = [[{"role": "user", "content": f"Write a detailed 300-word essay about {t}."}] for t in TOPICS[:B]]
    replies, st = bot.chat_batch(convs, reasoning_effort="none", max_tokens=256, temperature=0)
    print(f"batch {B:2d}: {st['gen_tokens']:5d} tokens in {st['gen_s']:5.1f} s  -> {st['gen_tps']:6.1f} tok/s total, "
          f"{st['gen_tps'] / B:5.1f} tok/s per request   (prefill {st['prefill_s']:.1f} s, peak {st['peak_gb']:.1f} GB)")
r = replies[-1]
print("sample:", (r.text or "[thinking] " + r.thinking)[:300].replace("\n", " "))
