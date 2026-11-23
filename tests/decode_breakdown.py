"""Where does a plain decode step go, against its byte floor?

1. Achievable read bandwidth: a vectorized Metal read kernel and mx.sum over 2 GB.
2. The steady-state plain decode step (pipelined, as in generate) and its bytes per token.
3. Ablations: routed experts skipped, lm_head replaced by a tiny head. What each removal saves,
   against its byte floor, says how much of the step is bytes and how much is fixed cost.
   Run-to-run noise is about 1 ms.

    python tests/decode_breakdown.py MODEL_DIR DENSE_BITS
"""

import sys
import time

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import inkling2_mini.model as M  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bits = int(sys.argv[2])

# ---- 1. bandwidth probe
kern = mx.fast.metal_kernel(
    name="bw_probe", input_names=["inp", "n4"], output_names=["out"],
    source="""
    uint gid = thread_position_in_grid.x, n = threads_per_grid.x;
    const device float4* p = (const device float4*)inp;
    float4 acc = 0;
    for (uint i = gid; i < (uint)n4[0]; i += n) acc += p[i];
    out[gid] = acc.x + acc.y + acc.z + acc.w;""")
x = mx.random.uniform(shape=(512 * 1024 * 1024,))  # 2 GB of float32
mx.eval(x)
n4 = mx.array([x.size // 4], mx.int32)
best = {}
for tg_count in (2048, 8192, 32768):
    g = tg_count * 256
    f = lambda: kern(inputs=[x, n4], grid=(g, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(g,)], output_dtypes=[mx.float32])[0]
    for _ in range(3):
        mx.eval(f())
    t = time.perf_counter()
    for _ in range(10):
        mx.eval(f())
    best[tg_count] = x.nbytes / ((time.perf_counter() - t) / 10) / 1e9
t = time.perf_counter()
for _ in range(10):
    mx.eval(x.sum())
best["mx.sum"] = x.nbytes / ((time.perf_counter() - t) / 10) / 1e9
print("bandwidth GB/s:", {k: round(v) for k, v in best.items()}, flush=True)
del x
mx.clear_cache()
peak_bw = max(best.values())

# ---- 2/3. decode step
bot = Inkling2Mini(sys.argv[1], verbose=False, dense_bits=bits, speculative=False)
ids = bot.encode([{"role": "user", "content": "Write a 200-word story about a lighthouse keeper."}], reasoning_effort="none")[0]


def step_ms(n=160):
    st = None
    for _, s in bot.generate_tokens(ids, max_tokens=n, temperature=0.0, speculative=False):
        st = s or st
    return 1e3 / st["gen_tps"]


step_ms(16)
full = min(step_ms() for _ in range(2))
orig_moe = M.GroupedSparseMoE.__call__
M.GroupedSparseMoE.__call__ = lambda self, x: mx.zeros_like(x)
no_exp = min(step_ms() for _ in range(2))
M.GroupedSparseMoE.__call__ = orig_moe
lm = bot.model.language_model
orig_head = lm.lm_head
V = lm.config.vocab_size
tiny = nn.Linear(lm.config.hidden_size, 8, bias=False)
mx.eval(tiny.parameters())
lm.lm_head = lambda h: mx.pad(tiny(h), [(0, 0)] * (h.ndim - 1) + [(0, V - 8)])
no_head = min(step_ms() for _ in range(2))
lm.lm_head = orig_head

from mlx.utils import tree_flatten  # noqa: E402
dense = expert = 0
for i, layer in enumerate(lm.model.layers):
    d = e = 0
    for k, v in tree_flatten(layer.parameters()):
        if "switch_mlp" in k:
            e += v.nbytes
        else:
            d += v.nbytes
    visits = 3 if 6 <= i < 18 else 1
    dense += d * visits
    expert += e / layer.mlp.switch_mlp.gate_up.weight.shape[0] * layer.mlp.top_k * visits
head = sum(v.nbytes for _, v in tree_flatten(orig_head.parameters()))
tot = dense + expert + head
fl = lambda b: b / (peak_bw * 1e9) * 1e3
print(f"dense {bits}: step {full:.1f} ms ({1e3 / full:.1f} tok/s); bytes/token {tot / 1e9:.2f} GB -> floor {fl(tot):.1f} ms at {peak_bw:.0f} GB/s", flush=True)
print(f"  experts: {expert / 1e9:.2f} GB, floor {fl(expert):.1f} ms, measured removal saves {full - no_exp:.1f} ms", flush=True)
print(f"  lm_head: {head / 1e9:.2f} GB, floor {fl(head):.1f} ms, measured removal saves {full - no_head:.1f} ms", flush=True)
rest = no_exp - (full - no_head)  # step without experts and without head
print(f"  dense+glue (no experts, no head): ~{rest:.1f} ms for {dense / 1e9:.2f} GB (floor {fl(dense):.1f} ms) -> fixed overhead ~{rest - fl(dense):.1f} ms", flush=True)
