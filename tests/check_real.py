"""Consistency checks on the real checkpoint (no reference implementation fits in 32 GB):

  * GPU (fast Metal kernels, neural-accelerator matmuls) vs CPU (generic paths) logits
  * one-shot prefill vs chunked prefill + token-by-token decode on the GPU

    python tests/check_real.py MODEL_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1])
lm = bot.model.language_model
messages = [{"role": "user", "content": "List three primary colors, then name one fruit for each."}]
ids, _, _ = bot.encode(messages, reasoning_effort="none")
ids = mx.concatenate([ids, mx.array([[bot.ids["<|message_model|>"], bot.ids["<|content_text|>"]]])], axis=1)
n = ids.shape[1]
print("prompt tokens:", n)


def run(device, chunks):
    with mx.stream(device):
        cache = bot.model.make_cache()
        outs, start = [], 0
        for size in chunks:
            outs.append(lm(ids[:, start : start + size], cache=cache, last_only=False))
            mx.eval(outs[-1])
            start += size
        return np.asarray(mx.concatenate(outs, axis=1).astype(mx.float32))


def report(name, a, b):
    agree = (a.argmax(-1) == b.argmax(-1)).mean()
    pa = np.exp(a - a.max(-1, keepdims=True)); pa /= pa.sum(-1, keepdims=True)
    pb = np.exp(b - b.max(-1, keepdims=True)); pb /= pb.sum(-1, keepdims=True)
    kl = (pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(-1)
    print(f"{name:34s} top1-agree={agree:.3f} max|dlogit|={np.abs(a - b).max():.3f} "
          f"mean KL={kl.mean():.2e} max KL={kl.max():.2e}")


gpu = run(mx.gpu, [n])
cpu = run(mx.cpu, [n])
split = n - 6
inc = run(mx.gpu, [split // 2, split - split // 2] + [1] * 6)
report("GPU vs CPU (one-shot)", gpu, cpu)
report("GPU one-shot vs prefill+decode", gpu, inc)
nxt = bot.tokenizer.decode(gpu[0, -1].argmax())
print("next token after the prompt:", repr(nxt))
