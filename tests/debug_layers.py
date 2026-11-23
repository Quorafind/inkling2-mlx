"""Layer-by-layer comparison between transformers and MLX on the tiny model.

    python tests/debug_layers.py TINY_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import Inkling2LoopForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini import model as M  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

tiny = Path(sys.argv[1])
ids = np.load(tiny / "ref_ids.npy")
ref = np.load(tiny / "ref_logits.npy")

tm = Inkling2LoopForConditionalGeneration.from_pretrained(tiny, dtype=torch.float32).eval()
captured = []
for i, layer in enumerate(tm.model.language_model.layers):
    layer.register_forward_hook(lambda mod, args, out, i=i: captured.append((i, out.detach().float().numpy())))
with torch.no_grad():
    logits = tm(input_ids=torch.tensor(ids)).logits.float().numpy()
print("torch reload vs saved reference:", np.abs(logits - ref).max())

mine = []
orig_dec, orig_loop = M.DecoderLayer.__call__, M.LoopDecoderLayer.__call__


depth = [0]


def wrap(fn):
    def call(self, x, blocks, cache=None, *a, **k):
        depth[0] += 1
        y = fn(self, x, blocks, cache, *a, **k)
        depth[0] -= 1
        if depth[0] == 0:  # LoopDecoderLayer delegates to DecoderLayer on pass 1
            mine.append(np.asarray(y.astype(mx.float32)))
        return y
    return call


model = load(tiny, verbose=False)
lm = model.language_model
M.LoopDecoderLayer.__call__ = wrap(orig_loop)
for layer in lm.model.layers:
    if type(layer) is M.DecoderLayer:
        type(layer).__call__ = wrap(orig_dec)
        break
out = lm(mx.array(ids), last_only=False)
print("mlx logits vs ref:", float(np.abs(np.asarray(out) - ref).max()))

emb_t = tm.model.language_model.embed_tokens(torch.tensor(ids)).detach().numpy()
emb_m = np.asarray(lm.model.embed(mx.array(ids)))
print(f"embed diff {np.abs(emb_t - emb_m).max():.2e}")
for n, ((i, t), m) in enumerate(zip(captured, mine)):
    print(f"app {n:2d} layer {i}: max|diff|={np.abs(t - m).max():.3e}  |t|={np.abs(t).max():.2f}")
