"""Do the decode / prep kernels move the GPU path away from FP32 ground truth?

Truth: CPU, every non-expert weight in FP32 (experts stay native NVFP4 / MXFP8). GPU BF16
variants (fusions on, off, and each fusion alone off) prefill the first P positions and
then decode the last D one token at a time, so prefill-side kernels (attention prep,
router) and decode-side ones (ring attention, single-token experts, conv state) are
both covered. Fixed text, so runs are comparable.

    python tests/kernel_precision.py MODEL_DIR
"""

import gc
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx_vlm.models.inkling2.language as IL  # noqa: E402
from inkling2_mini import decode_kernels as DK  # noqa: E402
from inkling2_mini import fused as F  # noqa: E402
from inkling2_mini import model as M  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

TEXT = (
    "The lighthouse stood at the end of a narrow spit of rock, and every evening the keeper climbed "
    "its one hundred and twelve steps to light the lamp. He kept a logbook of the weather, the ships "
    "that passed, and the birds that rested on the gallery rail. 秋天的时候，海面上的雾来得特别早，"
    "他常常要在下午四点就点亮灯塔。In the winter of that year a storm broke the main lens, and for three "
    "nights he kept a paraffin lamp burning by hand, sleeping in twenty-minute stretches between "
    "refills. When the supply boat finally arrived, the captain asked him why he had not simply "
    "signalled for help. He answered that the ships needed the light more than he needed the sleep, "
    "and that the log would show the lamp had never once gone dark."
)
model_dir = Path(sys.argv[1])
bot = Inkling2Mini(model_dir, speculative=False, dense_bits=16)
ids = bot.tokenizer.encode(TEXT)
D = 24
P = len(ids) - D
lm = bot.model.language_model
orig_route = M.GroupedSparseMoE._route


class _Off:
    active = staticmethod(lambda: False)


def variant(name):
    DK.ENABLED = name != "all off"
    M.GroupedSparseMoE._route = orig_route if name != "router off" else (
        lambda self, x, t: IL._route(x, self.gate_weight, self.e_score_correction_bias, self.top_k, self.route_scale))
    F.DK = _Off if name == "attn prep + ring off" else DK
    cache = bot.model.make_cache()
    pre = lm(mx.array([ids[:P]]), cache=cache, last_only=False)[0].astype(mx.float32)
    out = [np.asarray(pre)]
    for t in ids[P:-1]:
        out.append(np.asarray(lm(mx.array([[t]]), cache=cache)[0].astype(mx.float32)))
    return np.concatenate(out)


names = ["all off", "all on", "router off", "attn prep + ring off"]
gpu = {n: variant(n) for n in names}
del bot, lm
gc.collect()
mx.clear_cache()
with mx.stream(mx.cpu):
    truth_model = load(model_dir, dtype=mx.float32)
    truth = truth_model.language_model(mx.array([ids[:-1]]), last_only=False)[0]
    mx.eval(truth)
truth = np.asarray(truth)


def kl(p_logits, q_logits):
    p = np.exp(p_logits - p_logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    q = np.exp(q_logits - q_logits.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
    return (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)


print(f"{len(ids) - 1} positions: {P - 1 + 1} from prefill, {D - 1} from single-token decode (dense BF16, BF16 KV)")
for n in names:
    k = kl(truth, gpu[n])
    a = truth.argmax(-1) == gpu[n].argmax(-1)
    print(f"  {n:22s} prefill: mean KL {k[:P].mean():.2e} median {np.median(k[:P]):.2e} top1 {a[:P].mean():.3f} | "
          f"decode: mean KL {k[P:].mean():.2e} median {np.median(k[P:]):.2e} top1 {a[P:].mean():.3f}")
