"""Single-token MoE decode, isolated: the router, the NVFP4 expert kernels and the MXFP8
fallback, each timed over the 48 layer visits of one token and set against its bytes; then
the whole MoE + short conv + residual tail, unfused (route, experts, conv: 6 kernels on the
NVFP4 layers, more on the MXFP8 ones) against moe_kernels.moe_decode_conv (3 kernels).

    python tests/bench_moe_decode.py MODEL_DIR
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402

from inkling2_mini import decode_kernels, moe_kernels, prefill_kernels  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1], verbose=False, speculative=False)
lm = bot.model.language_model
layers = lm.model.layers
VISITS = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
H = lm.config.hidden_size
mx.random.seed(0)
x0 = (mx.random.normal((1, 1, H)) * 0.5).astype(mx.bfloat16)


def expert_bytes(sw, top_k=8):
    n = sum(v.nbytes for v in (sw.gate_up.weight, sw.gate_up.scales, sw.down_proj.weight, sw.down_proj.scales))
    return n / sw.gate_up.weight.shape[0] * top_k


def timeit(fn, reps=30):
    for _ in range(5):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def chain(visits, body):
    def run():
        x = x0
        for i in visits:
            x = body(layers[i].mlp, x)
        return x
    return run


nv = [i for i in VISITS if layers[i].mlp._moe_decode_ok()]
mx8 = [i for i in VISITS if not layers[i].mlp._moe_decode_ok()]
print(f"visits: {len(nv)} with the fused NVFP4 kernels, {len(mx8)} on the gather_qmm fallback", flush=True)


def route_only(m, x):
    idx, w = m._route(x, 1)
    return x + (w.sum() * 1e-6).astype(x.dtype)  # keep a dependency, as the real chain has


def fixed_routes(m, x):
    idx, w = m._route(x0, 1)  # same experts each call: only the expert kernels are timed
    return idx, w


routes = {i: fixed_routes(layers[i].mlp, x0) for i in set(VISITS)}
mx.eval(routes)


def experts_only(m, x, i=None):
    idx, w = routes[i]
    return decode_kernels.moe_decode(x, m.switch_mlp, idx.reshape(-1), w.reshape(-1), m.inter, *m._caps)


def full(m, x):
    return m(x)


t_route = timeit(chain(VISITS, route_only))
b_route = sum(layers[i].mlp.gate_weight.nbytes * 2 for i in VISITS)  # the FP32 transposed copy
print(f"router, 48 visits: {t_route:5.2f} ms  ({b_route / 1e6:.0f} MB)", flush=True)

run_nv = lambda: _chain_nv()  # noqa: E731


def _chain_nv():
    x = x0
    for i in nv:
        x = experts_only(layers[i].mlp, x, i)
    return x


t_nv = timeit(_chain_nv)
b_nv = sum(expert_bytes(layers[i].mlp.switch_mlp) for i in nv)
print(f"NVFP4 experts (2 kernels), {len(nv)} visits: {t_nv:5.2f} ms for {b_nv / 1e9:.2f} GB -> {b_nv / t_nv / 1e6:.0f} GB/s", flush=True)

t_mx = timeit(chain(mx8, full))
b_mx = sum(expert_bytes(layers[i].mlp.switch_mlp) for i in mx8)
print(f"MXFP8 experts (router + fallback), {len(mx8)} visits: {t_mx:5.2f} ms for {b_mx / 1e9:.2f} GB -> {b_mx / t_mx / 1e6:.0f} GB/s", flush=True)

t_full = timeit(chain(VISITS, full))
b_all = b_nv + b_mx + b_route
print(f"full MoE, 48 visits: {t_full:5.2f} ms for {b_all / 1e9:.2f} GB -> {b_all / t_full / 1e6:.0f} GB/s", flush=True)


state0 = mx.zeros((1, 3, H), mx.float32)


def tail_unfused(i, x):
    lay = layers[i]
    out = prefill_kernels.short_conv(lay.mlp(x), state0, lay.mlp_sconv.conv.weight, residual=x.astype(mx.float32),
                                     out_dtype=x.dtype)
    return out


def tail_fused(i, x):
    lay = layers[i]
    out, _ = moe_kernels.moe_decode_conv(x, lay.mlp, state0, lay.mlp_sconv.conv.weight, x.astype(mx.float32), x.dtype)
    return out


def chain_i(visits, body):
    def run():
        x = x0
        for i in visits:
            x = body(i, x)
        return x
    return run


moe_kernels.ENABLED = False
t_old = timeit(chain_i(VISITS, tail_unfused))
moe_kernels.ENABLED = True
t_new = timeit(chain_i(VISITS, tail_fused))
print(f"MoE + conv + residual, 48 visits: unfused {t_old:5.2f} ms, fused {t_new:5.2f} ms for {b_all / 1e9:.2f} GB "
      f"({b_all / t_new / 1e6:.0f} GB/s)", flush=True)
