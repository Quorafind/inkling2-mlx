"""Speed at 1 to 5 loop passes (variable-loop checkpoints choose the pass count at
inference time). Speed depends only on the architecture, so Inkling 2 Mini's weights
measure it too (its text at other pass counts is not meaningful: it is trained for 3).

Per pass count: prefill of a ~4k-token prompt, then plain decoding (no DSpark) of 128
tokens, ms per token as the median of 3 runs.

    python tests/bench_passes.py MODEL_DIR [passes ...]
"""

import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1], verbose=False)
passes = [int(a) for a in sys.argv[2:]] or [1, 2, 3, 4, 5]
text = (ROOT / "README.md").read_text()[:12000]
ids = bot.encode([{"role": "user", "content": text + "\n\nSummarize."}], reasoning_effort="none")[0]
cfg = bot.text_config
print(f"prompt {ids.shape[1]} tokens")
for p in passes:
    pre, dec = [], []
    for _ in range(3):
        st = bot.new_state(passes=p)
        t0 = time.perf_counter()
        first = None
        n = 0
        for tok, stats in bot.generate_tokens(ids, state=st, max_tokens=129, temperature=0, speculative=False):
            n += 1
            if first is None:
                first = time.perf_counter()
        end = time.perf_counter()
        pre.append(ids.shape[1] / (first - t0))
        dec.append((end - first) / (n - 1) * 1e3)
        mx.clear_cache()
    ms = statistics.median(dec)
    print(f"{p} pass{'es' if p > 1 else '  '} ({cfg.visits(p)} visits): prefill {statistics.median(pre):5.0f} tok/s, "
          f"decode {ms:5.1f} ms/token = {1e3 / ms:5.1f} tok/s")
