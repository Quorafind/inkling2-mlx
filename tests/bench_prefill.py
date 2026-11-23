"""Long-context prefill throughput and peak memory.

    python tests/bench_prefill.py MODEL_DIR CHUNK CONTEXT [CONTEXT ...] [--kv-bits 8] [--dense-bits 6]
"""
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

kv_bits = None
if "--kv-bits" in sys.argv:
    i = sys.argv.index("--kv-bits")
    kv_bits = int(sys.argv[i + 1])
    del sys.argv[i : i + 2]
dense_bits = 6
if "--dense-bits" in sys.argv:
    i = sys.argv.index("--dense-bits")
    dense_bits = int(sys.argv[i + 1])
    del sys.argv[i : i + 2]
bot = Inkling2Mini(sys.argv[1], speculative=False, kv_bits=kv_bits, dense_bits=dense_bits)
lm = bot.model.language_model
chunk = int(sys.argv[2])
text = bot.tokenizer.encode(open(Path(__file__).resolve().parents[1] / "README.md").read())
for n in [int(x) for x in sys.argv[3:]]:
    ids = mx.array([(text * (n // len(text) + 1))[:n]])
    cache = bot.model.make_cache(kv_bits=kv_bits)
    mx.clear_cache()
    mx.reset_peak_memory()
    t = time.perf_counter()
    logits, _ = bot._prefill(lm.model.embed(ids), cache, chunk, None)
    mx.eval(logits)
    dt = time.perf_counter() - t
    tok = mx.array([[1000]])
    for _ in range(3):
        mx.eval(lm(tok, cache=cache))
    t = time.perf_counter()
    for _ in range(10):
        mx.eval(lm(tok, cache=cache))
    dec = (time.perf_counter() - t) / 10 * 1e3
    print(f"dense {dense_bits:2d} kv_bits {kv_bits or 16:2d} chunk {chunk:5d} context {n:6d}: prefill {dt:6.1f} s = {n / dt:5.0f} tok/s, "
          f"peak {mx.get_peak_memory() / 1e9:5.1f} GB, decode {dec:5.1f} ms/token", flush=True)
    del cache, logits
