"""Bytes read per decoded token (from the loaded parameters) and this machine's achievable
bandwidth, giving the single-request decode floor.

    python tests/byte_floor.py MODEL_DIR [dense_bits, default 6]
"""
import sys, time, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx.utils import tree_flatten
from inkling2_mini.loader import load
model = load(__import__("pathlib").Path(sys.argv[1]), verbose=False,
             dense_bits=int(sys.argv[2]) if len(sys.argv) > 2 else 6)
lm = model.language_model
cfg = lm.config
layers = lm.model.layers
visits = {i: 1 for i in range(len(layers))}
for i in range(6, 18): visits[i] = 3
dense = expert = 0
for i, layer in enumerate(layers):
    d = e = 0
    for k, v in tree_flatten(layer.parameters()):
        if "switch_mlp" in k: e += v.nbytes
        else: d += v.nbytes
    E = layer.mlp.switch_mlp.gate_up.weight.shape[0]
    dense += d * visits[i]; expert += e / E * layer.mlp.top_k * visits[i]
head = sum(v.nbytes for _, v in tree_flatten(lm.lm_head.parameters()))
tot = dense + expert + head
print(f"bytes per decoded token: dense {dense/1e9:.2f} GB (loop layers x3), routed experts {expert/1e9:.2f} GB, lm_head {head/1e9:.2f} GB -> {tot/1e9:.2f} GB")
# achievable bandwidth: 8-bit quantized GEMV over a large weight (lm_head-like), one row
w = mx.random.normal((131072, 2304)).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=8); del w
x = mx.random.normal((1, 2304)).astype(mx.bfloat16); mx.eval(wq, s, b, x)
nb = wq.nbytes + s.nbytes + b.nbytes
f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=8)
for _ in range(3): mx.eval(f())
t = time.perf_counter(); n = 30
for _ in range(n): mx.eval(f())
dt = (time.perf_counter() - t) / n
print(f"8-bit qmv over {nb/1e9:.2f} GB: {nb/dt/1e9:.0f} GB/s")
y = mx.zeros((nb // 2,), mx.bfloat16) + 1; mx.eval(y)
t = time.perf_counter()
for _ in range(10): mx.eval(y.sum())
print(f"plain sum over {y.nbytes/1e9:.2f} GB: {y.nbytes/((time.perf_counter()-t)/10)/1e9:.0f} GB/s")
for bw in (120, 130, 140):
    print(f"  floor at {bw} GB/s: {tot/bw/1e6:.1f} ms/token = {bw*1e9/tot:.1f} tok/s")
