"""Fused message-gate kernels vs the reference MessageGate math (CPU, FP32), on real
loop-layer weights. python tests/test_message_gates.py MODEL_DIR"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini import prefill_kernels as P  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

model = load(Path(sys.argv[1]), verbose=False)
ok = True
for li in (7, 11):
    layer = model.language_model.model.layers[li]
    for N in (1, 1024):
        rng = np.random.default_rng(N + li)
        QH = 1 if layer.is_sliding else 3
        h = mx.array(rng.normal(0, 3, (N, QH, 2304)).astype(np.float32))
        msgs = mx.array(rng.normal(0, 1, (N, 4, 2304)).astype(np.float32))
        gates = [layer.q_message_gate, layer.k_message_gate, layer.v_message_gate]
        norms = [layer.input_layernorm] * 3 if layer.is_sliding else [layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm]
        with mx.stream(mx.cpu):
            ref = mx.stack([norms[g](gates[g](h[:, 0 if QH == 1 else g], msgs[:, g])).astype(mx.bfloat16) for g in range(3)])
            ref_mlp = layer.post_attention_layernorm(layer.mlp_message_gate(h[:, 0], msgs[:, 3])).astype(mx.bfloat16)
            mx.eval(ref, ref_mlp)
        down, fg = layer._gates3
        w = layer.input_layernorm.weight if layer.is_sliding else layer._qkv_weight()
        got = P.message_gates(h, msgs, 0, 3, down, fg, w)
        d1, f1 = layer._gate_mlp
        got_mlp = P.message_gates(h[:, :1], msgs, 3, 1, d1, f1, layer.post_attention_layernorm.weight)[0]
        for name, g_, r_ in (("q/k/v gates", got, ref), ("mlp gate", got_mlp, ref_mlp)):
            g_, r_ = np.asarray(g_.astype(mx.float32)), np.asarray(r_.astype(mx.float32))
            err = np.abs(g_ - r_).max() / np.abs(r_).max()
            good = err < 2e-2
            ok &= good
            print(f"layer {li} N={N:4d} {name:12s}: rel.err {err:.1e}, equal {(g_ == r_).mean():.3f} {'ok' if good else 'MISMATCH'}")
print("MESSAGE GATES OK" if ok else "MESSAGE GATES MISMATCH")

# Same GEMMs on both sides: the old compiled GPU path vs the fused kernels.
from inkling2_mini.fused import gates as old_gates  # noqa: E402

for li in (7, 11):
    layer = model.language_model.model.layers[li]
    N = 1024
    rng = np.random.default_rng(li)
    QH = 1 if layer.is_sliding else 3
    h = mx.array(rng.normal(0, 3, (N, QH, 2304)).astype(np.float32))
    msgs = mx.array(rng.normal(0, 1, (N, 4, 2304)).astype(np.float32))
    h3 = mx.broadcast_to(h.transpose(1, 0, 2), (3, N, 2304)) if QH == 1 else h.transpose(1, 0, 2)
    m3 = msgs[:, :3].transpose(1, 0, 2)
    down, fg = layer._gates3
    g3 = old_gates(h3, m3, down, fg)
    if layer.is_sliding:
        old = mx.fast.rms_norm(g3, layer.input_layernorm.weight, 1e-6).astype(mx.bfloat16)
        w = layer.input_layernorm.weight
    else:
        old = (mx.fast.rms_norm(g3, None, 1e-6) * layer._qkv_weight()[:, None, :].astype(mx.float32)).astype(mx.bfloat16)
        w = layer._qkv_weight()
    new = P.message_gates(h, msgs, 0, 3, down, fg, w)
    a, b = np.asarray(old.astype(mx.float32)), np.asarray(new.astype(mx.float32))
    print(f"GPU old path vs fused kernels, layer {li}: rel.err {np.abs(a - b).max() / np.abs(a).max():.1e}, equal {(a == b).mean():.4f}")
