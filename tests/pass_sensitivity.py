"""How close are fewer loop passes to the trained 3? (feasibility of pass-level
self-speculation / early exit). Teacher-forced over the model's own text.

    python tests/pass_sensitivity.py MODEL_DIR
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1], speculative=False)
lm = bot.model.language_model
cfg = lm.model.config
prompts = ["Explain how a hash map works, with a short Python example.",
           "Write a short story about a lighthouse keeper.",
           "用中文解释一下为什么天空是蓝色的，并举一个生活中的例子。"]
rows = {1: [], 2: [], 3: []}
for p in prompts:
    ids, _, _ = bot.encode([{"role": "user", "content": p}], reasoning_effort="none")
    toks = [t for t, _ in bot.generate_tokens(ids, max_tokens=300, temperature=0, speculative=False)]
    seq = mx.concatenate([ids, mx.array([toks])], axis=1)
    start = ids.shape[1] - 1
    for passes in (3, 2, 1):
        cfg.loop_config["num_passes"] = passes
        out = lm(seq, last_only=False)[0, start:-1].astype(mx.float32)
        rows[passes].append(np.asarray(out - mx.logsumexp(out, axis=-1, keepdims=True)))
    cfg.loop_config["num_passes"] = 3
ref = np.concatenate(rows[3])
for passes in (2, 1):
    q = np.concatenate(rows[passes])
    kl = (np.exp(ref) * (ref - q)).sum(-1)
    agree = (ref.argmax(-1) == q.argmax(-1)).mean()
    print(f"{passes} pass(es) vs 3: top-1 agreement {agree:.3f}, mean KL {kl.mean():.3f}, median KL {np.median(kl):.3f} "
          f"({len(kl)} positions)")
