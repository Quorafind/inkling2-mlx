"""Where does a single-token decode step spend its time?

    python tests/profile_decode.py MODEL_DIR
"""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini  # noqa: E402
from mlx_vlm.models.inkling2 import language as L  # noqa: E402

bot = Inkling2Mini(sys.argv[1])
lm = bot.model.language_model
ids, _, _ = bot.encode([{"role": "user", "content": "Tell me about the history of lighthouses."}], reasoning_effort="none")
cache = bot.model.make_cache()
mx.eval(lm(ids, cache=cache))
tok = mx.array([[1000]])


def step(n=20):
    build, total = 0.0, 0.0
    for _ in range(n):
        t0 = time.perf_counter()
        out = lm(tok, cache=cache)
        t1 = time.perf_counter()
        mx.eval(out)
        t2 = time.perf_counter()
        build += t1 - t0
        total += t2 - t0
    return build / n * 1e3, total / n * 1e3


step(3)
b, t = step()
print(f"full step:        graph build {b:5.1f} ms   build+run {t:5.1f} ms")

# Without experts (MoE returns zeros of the right shape): attention + glue only.
orig_moe = L.SparseMoE.__call__
L.SparseMoE.__call__ = lambda self, x: mx.zeros_like(x)
b, t = step()
print(f"no experts:       graph build {b:5.1f} ms   build+run {t:5.1f} ms")
L.SparseMoE.__call__ = orig_moe

# Without the vocabulary projection.
orig_logits = type(lm).logits
type(lm).logits = lambda self, h: h
b, t = step()
print(f"no lm_head:       graph build {b:5.1f} ms   build+run {t:5.1f} ms")
type(lm).logits = orig_logits

# Pure weight-streaming floor for the same bytes: one GEMV per weight group.
dense = []
for layer in lm.model.layers:
    a = layer.self_attn
    dense += [a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight, a.gate_proj.weight, a.r_proj.weight]
reps = {i: (3 if 6 <= i < 18 else 1) for i in range(24)}
ws = [w for i, layer in enumerate(lm.model.layers) for w in dense[i * 6 : i * 6 + 6] for _ in range(reps[i])]
x = mx.ones((1, 4096), mx.bfloat16)
mx.eval(ws)
def floor():
    outs = [x[:, : w.shape[1]] @ w.T for w in ws] + [x[:, :2304] @ lm.lm_head.weight.T]
    mx.eval(outs)
floor()
t0 = time.perf_counter()
for _ in range(10):
    floor()
dt = (time.perf_counter() - t0) / 10
gb = (sum(w.nbytes for w in ws) + lm.lm_head.weight.nbytes) / 1e9
print(f"dense+lm_head GEMV floor: {dt * 1e3:5.1f} ms for {gb:.2f} GB ({gb / dt:.0f} GB/s)")
