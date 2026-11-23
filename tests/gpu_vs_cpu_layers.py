"""Per-visit hidden-state differences between the GPU and CPU paths on real weights.

    python tests/gpu_vs_cpu_layers.py MODEL_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1], speculative=False)
lm = bot.model.language_model
ids, _, _ = bot.encode([{"role": "user", "content": "List three primary colors, then name one fruit for each."}],
                       reasoning_effort="none")
visits = list(range(lm.config.num_cache_layers + 1))


def run(device):
    with mx.stream(device):
        out, caps = lm.model(ids, capture=visits)
        logits = lm.logits(out)
        mx.eval(caps, logits)
    return [np.asarray(c.astype(mx.float32))[0] for c in caps], np.asarray(logits)[0]


g, gl = run(mx.gpu)
c, cl = run(mx.cpu)
for v in visits:
    d = np.abs(g[v] - c[v])
    rel = np.linalg.norm(g[v] - c[v], axis=-1) / (np.linalg.norm(c[v], axis=-1) + 1e-9)
    worst = int(rel.argmax())
    print(f"visit {v:2d}: median rel {np.median(rel):.2e}  max rel {rel.max():.2e} @pos {worst}  max|d| {d.max():.2e}")
top_g, top_c = gl.argmax(-1), cl.argmax(-1)
print("logits top1 agreement per position:", "".join("." if a == b else "X" for a, b in zip(top_g, top_c)))
