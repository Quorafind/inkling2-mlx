"""Which --dense-bits stays at the model's own noise? GPU runs at dense 16 / 8 / 6 bits vs FP32
ground truth.

Truth: CPU, every non-expert weight in FP32 (experts stay native NVFP4 / MXFP8). Each GPU
variant (all fusions on, 8-bit KV as in chat.py) prefills the first P positions of a fixed
English + Chinese text and decodes the last D one token at a time. Dense 16 is the BF16
noise floor the quantized variants are judged against.

    python tests/dense_precision.py MODEL_DIR [bits ...]
"""

import gc
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

TEXT = (
    "The lighthouse stood at the end of a narrow spit of rock, and every evening the keeper climbed "
    "its one hundred and twelve steps to light the lamp. He kept a logbook of the weather, the ships "
    "that passed, and the birds that rested on the gallery rail. In the winter of that year a storm "
    "broke the main lens, and for three nights he kept a paraffin lamp burning by hand, sleeping in "
    "twenty-minute stretches between refills. When the supply boat finally arrived, the captain asked "
    "him why he had not simply signalled for help. He answered that the ships needed the light more "
    "than he needed the sleep, and that the log would show the lamp had never once gone dark.\n\n"
    "秋天的时候，海面上的雾来得特别早，他常常要在下午四点就点亮灯塔。岛上只有他一个人，补给船每两周来一次，"
    "带来面粉、煤油、几封信和一捆旧报纸。他读报纸很慢，一份报纸要读上好几天，连广告都不放过。"
    "有一年冬天，船迟了整整一个月。他把最后一袋面粉分成三十份，每天只烤一个小小的饼，"
    "剩下的时间就坐在灯室里看海。后来他在日志里写道：那一个月，海是灰色的，天也是灰色的，"
    "只有灯是黄色的。船终于来的那天早上，他远远看见桅杆上的旗子，竟然一下子不知道该做什么，"
    "只是站在码头上，把手里的帽子攥得紧紧的。船长跳上岸，第一句话是问灯有没有灭过。"
    "他摇摇头，说没有，一个晚上也没有。船长没再说什么，只是拍了拍他的肩膀，"
    "让水手们先把煤油搬上去，再搬面粉。"
)
model_dir = Path(sys.argv[1])
bits_list = [int(b) for b in sys.argv[2:]] or [16, 8, 6]
D = 32


def run(bits):
    bot = Inkling2Mini(model_dir, speculative=False, dense_bits=bits, verbose=False)
    ids = bot.tokenizer.encode(TEXT)
    lm = bot.model.language_model
    cache = bot.model.make_cache(kv_bits=bot.kv_bits)  # 8-bit full-attention KV, as in chat.py
    P = len(ids) - D
    out = [np.asarray(lm(mx.array([ids[:P]]), cache=cache, last_only=False)[0].astype(mx.float32))]
    for t in ids[P:-1]:
        out.append(np.asarray(lm(mx.array([[t]]), cache=cache)[0].astype(mx.float32)))
    gb = mx.get_active_memory() / 1e9
    del bot, lm, cache
    gc.collect()
    mx.clear_cache()
    return ids, np.concatenate(out), gb


gpu = {}
for b in bits_list:
    ids, gpu[b], gb = run(b)
    print(f"dense {b:2d}: active memory after the run {gb:.2f} GB", flush=True)
P = len(ids) - D
with mx.stream(mx.cpu):
    truth_model = load(model_dir, dtype=mx.float32, verbose=False)
    truth = truth_model.language_model(mx.array([ids[:-1]]), last_only=False)[0]
    mx.eval(truth)
truth = np.asarray(truth)


def kl(p_logits, q_logits):
    p = np.exp(p_logits - p_logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    q = np.exp(q_logits - q_logits.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
    return (p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1)


print(f"{len(ids) - 1} positions vs CPU-FP32 truth: {P} from prefill, {D - 1} from single-token decode")
for b in bits_list:
    k = kl(truth, gpu[b])
    a = truth.argmax(-1) == gpu[b].argmax(-1)
    print(f"  dense {b:2d}  all: mean KL {k.mean():.2e} median {np.median(k):.2e} p99 {np.quantile(k, 0.99):.2e} "
          f"top1 {a.mean():.3f} | decode: mean KL {k[P:].mean():.2e} top1 {a[P:].mean():.3f}")
for b in bits_list:
    if b != 16 and 16 in gpu:
        k = kl(gpu[16], gpu[b])
        print(f"  dense {b:2d} vs dense 16: mean KL {k.mean():.2e} median {np.median(k):.2e} "
              f"top1 {(gpu[16].argmax(-1) == gpu[b].argmax(-1)).mean():.3f}")
