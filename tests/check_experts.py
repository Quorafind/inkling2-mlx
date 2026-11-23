"""Check that the natively-loaded NVFP4 / MXFP8 experts decode to exactly the
values the transformers reference dequantizer produces.

The reference side reads raw bytes straight from the safetensors file with numpy
(independent of MLX) and follows `Inkling2LoopDequantizeExperts`:
  NVFP4: two E2M1 codes per byte (even element in the low nibble), an E4M3 scale
         per 16 values, times the per-expert factor amax / outer_scale_denominator
  MXFP8: E4M3 values with a 2 ** (scale - 127) scale per 32 values
then de-interleaves the fused w13 rows into gate (even rows) and up (odd rows).

    python tests/check_experts.py MODEL_DIR [layer ...]
"""

import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.loader import load  # noqa: E402

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3(bits):
    bits = bits.astype(np.int32)
    s = np.where(bits & 0x80, -1.0, 1.0)
    e = (bits >> 3) & 0xF
    m = bits & 0x7
    val = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * 2.0 ** (e - 7))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return (s * val).astype(np.float32)


class Raw:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get(self, name, expert=None):
        info = self.header[name]
        a, b = info["data_offsets"]
        shape = info["shape"]
        raw = self.mm[self.base + a : self.base + b]
        dt = {"F32": np.float32, "BF16": np.uint16}.get(info["dtype"], np.uint8)
        arr = raw.view(dt).reshape(shape)
        return arr if expert is None else np.array(arr[expert])


def reference(raw, layer, proj, e):
    p = f"model.llm.layers.{layer}.mlp.experts.{proj}_weight"
    w = raw.get(p, e)
    scale = raw.get(p + "_scale", e)
    if p + "_amax" in raw.header:  # NVFP4
        vals = np.stack([E2M1[w & 0xF], E2M1[w >> 4]], -1).reshape(w.shape[0], -1)
        vals = vals.reshape(*scale.shape, 16) * e4m3(scale)[..., None]
        factor = raw.get(p + "_amax")[e] / raw.get(p + "_outer_scale_denominator")[0]
        vals = vals.reshape(w.shape[0], -1) * factor
    else:  # MXFP8
        vals = e4m3(w).reshape(*scale.shape, 32) * np.exp2(scale.astype(np.float32) - 127)[..., None]
        vals = vals.reshape(w.shape[0], -1)
    if proj == "w13":
        return {"gate_proj": vals[0::2], "up_proj": vals[1::2]}
    return {"down_proj": vals}


def mine(module, e):
    w = mx.dequantize(
        module.weight[e], module.scales[e], None,
        group_size=module.group_size, bits=module.bits, mode=module.mode,
    ).astype(mx.float32)
    if "weight_scale_2" in module:
        w = w * module["weight_scale_2"][e].astype(mx.float32)
    return np.asarray(w)


model_dir = Path(sys.argv[1])
layers = [int(x) for x in sys.argv[2:]] or [0, 9, 19, 20, 23]
model = load(model_dir)
raw = Raw(next(model_dir.glob("model-*.safetensors")))
ok = True
for layer in layers:
    sw = model.language_model.model.layers[layer].mlp.switch_mlp
    for e in (0, 63, 127):
        for proj in ("w13", "w2"):
            for name, ref in reference(raw, layer, proj, e).items():
                if name == "down_proj":
                    module, got = sw.down_proj, mine(sw.down_proj, e)
                else:  # gate / up live in one fused [gate; up] matrix
                    module, full = sw.gate_up, mine(sw.gate_up, e)
                    half = full.shape[0] // 2
                    got = full[:half] if name == "gate_proj" else full[half:]
                err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
                good = got.shape == ref.shape and err < 1e-6
                ok &= good
                if not good or e == 0:
                    print(f"layer {layer:2d} {module.mode:5s} expert {e:3d} {name:9s} "
                          f"shape {got.shape} rel.err {err:.1e} {'ok' if good else 'MISMATCH'}")
print("EXPERTS OK" if ok else "EXPERT MISMATCH")
