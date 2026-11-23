"""Load the original Inkling 2 Mini checkpoint into the MLX model.

The checkpoint ships experts in their trained precisions and we keep them that
way: layers listed under `nvfp4_nearest` in hf_quant_config.json become MLX
`nvfp4` gather-matmuls with a per-expert global scale (amax / denominator),
layers listed under `mxfp8_nearest` become `mxfp8` gather-matmuls. Nothing is
dequantized or re-quantized. Everything else is BF16 / FP32 as stored.
"""

import json
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_vlm.models.inkling2.inkling2 import Model as _Inkling2Model
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear
from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear

from .model import Model, ModelConfig

EXPERT_MODES = {
    "nvfp4_nearest": dict(group_size=16, bits=4, mode="nvfp4"),
    "mxfp8_nearest": dict(group_size=32, bits=8, mode="mxfp8"),
}
_GATES = {"attn": "q_message_gate", "k": "k_message_gate", "v": "v_message_gate", "mlp": "mlp_message_gate"}


def expert_modes(model_dir: Path) -> dict:
    """layer index -> MLX quantization settings for that layer's experts."""
    meta = json.loads((model_dir / "hf_quant_config.json").read_text())
    if meta.get("quant_method") != "inkling2":
        raise ValueError(f"unexpected quant_method {meta.get('quant_method')!r}")
    modes = {}
    for group in meta["model"]["parameter_groups"]:
        settings = EXPERT_MODES.get(group.get("weight_transform"))
        if settings is None:
            if group.get("weight_transform") != "cast":
                raise ValueError(f"unsupported weight transform {group['weight_transform']}")
            continue
        for name in group["tensors"]:
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            if modes.setdefault(layer, settings) != settings:
                raise ValueError(f"layer {layer} mixes expert precisions")
    return modes


def _quantize_experts(model: Model, modes: dict):
    cfg = model.config.text_config
    hidden, inter, experts = cfg.hidden_size, cfg.intermediate_size, cfg.n_routed_experts
    for i, layer in enumerate(model.language_model.model.layers):
        settings = modes[i]
        mlp = layer.mlp.switch_mlp
        for name, (din, dout) in (
            ("gate_up", (hidden, 2 * inter)),
            ("down_proj", (inter, hidden)),
        ):
            q = QuantizedSwitchLinear(din, dout, experts, bias=False, **settings)
            if settings["mode"] == "nvfp4":
                q = ScaledQuantizedSwitchLinear(q, mx.ones((experts,)), "weight_scale_2")
            setattr(mlp, name, q)


def remap(weights: dict) -> dict:
    """Original checkpoint names -> this MLX model's parameter names."""
    w = _Inkling2Model.sanitize(None, weights)  # shared Inkling 2 layout
    out = {}
    queries = kv_queries = None
    for key, value in w.items():
        if key == "language_model.model.middle_cycle_cross_pass_queries":
            queries = value
            continue
        if key == "language_model.model.middle_cycle_cross_pass_kv_queries":
            kv_queries = value
            continue
        m = re.match(r"(language_model\.model\.layers\.\d+\.)controllers\.(\w+)\.(\w+)$", key)
        if m:
            prefix, which, leaf = m.groups()
            if leaf in ("forget_weight", "gain_weight"):
                value = value.astype(mx.float32)
            key = f"{prefix}{_GATES[which]}.{leaf}"
        out[key] = value
    # gate / up were split from one w13 tensor; keep them as one [gate; up] matrix.
    # The NVFP4 per-expert global scale is shared (one amax per expert for w13).
    for key in [k for k in out if k.endswith(".switch_mlp.gate_proj.weight")]:
        base = key[: -len("gate_proj.weight")]
        for leaf in ("weight", "scales", "biases"):
            g, u = base + "gate_proj." + leaf, base + "up_proj." + leaf
            if g in out:
                out[base + "gate_up." + leaf] = mx.concatenate([out.pop(g), out.pop(u)], axis=-2)
        g, u = base + "gate_proj.weight_scale_2", base + "up_proj.weight_scale_2"
        if g in out:
            out[base + "gate_up.weight_scale_2"] = out.pop(g)
            out.pop(u)
    # [q, mlp] pairs and [k, v] pairs per loop layer -> (loop_layers, [q, k, v, mlp], hidden)
    q, mlp = queries[0::2], queries[1::2]
    k, v = kv_queries[0::2], kv_queries[1::2]
    out["language_model.model.message_query"] = mx.stack([q, k, v, mlp], axis=1).astype(mx.float32)
    return out


# BF16 projections that dominate per-token weight traffic (48 visits each), plus the
# vocabulary head. sink_proj is included so it can share the fused projection matrix;
# the small dynamic-conv tap generators stay BF16.
DENSE_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "r_proj", "sink_proj")


def quantize_dense(model, bits, group_size=64):
    """Optional, lossy: affine-quantize attention projections and lm_head.

    Not part of the checkpoint's trained precision. 8 bits is close to lossless and
    removes ~1.7 GB of BF16 reads per decoded token; 6 bits (the chat default) removes
    another ~0.5 GB at the model's own BF16 noise.
    """

    def predicate(path, module):
        if not isinstance(module, nn.Linear):
            return False
        leaf = path.split(".")[-1]
        return path == "language_model.lm_head" or (
            ".self_attn." in path and leaf in DENSE_PROJECTIONS
        )

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    mx.eval(model.parameters())


def load(model_dir, verbose=True, dtype=None, dense_bits=None, fuse=True):
    """dtype: optionally cast float weights (except FP32-by-design tensors); tests only.
    dense_bits: optionally quantize attention projections + lm_head (see quantize_dense).
    fuse: concatenate attention projections and stack message gates (fused.py)."""
    model_dir = Path(model_dir)
    t0 = time.perf_counter()
    config = ModelConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
    model = Model(config)
    # Unquantized (float) expert checkpoints are only used by the parity tests.
    if (model_dir / "hf_quant_config.json").exists():
        _quantize_experts(model, expert_modes(model_dir))

    weights = {}
    for f in sorted(model_dir.glob("model*.safetensors")):
        weights.update(mx.load(str(f)))
    weights = remap(weights)
    if dtype is not None:
        weights = {
            k: v.astype(dtype) if v.dtype in (mx.float32, mx.bfloat16, mx.float16) and v.dtype != dtype
            and not k.endswith(("query", "gate_weight", "e_score_correction_bias", "forget_weight", "gain_weight"))
            else v
            for k, v in weights.items()
        }
    model.load_weights(list(weights.items()), strict=True)
    del weights

    # Materialize layer by layer to bound the peak while gate/up are split.
    lm = model.language_model
    mx.eval(lm.model.embed_tokens.parameters(), lm.lm_head.parameters())
    for layer in lm.model.layers:
        mx.eval(layer.parameters())
    mx.eval(model.parameters())
    if dense_bits and dense_bits < 16:
        quantize_dense(model, dense_bits)
    if fuse:
        model.language_model.model.fuse()
        mx.clear_cache()
    model.eval()
    if verbose:
        gb = sum(v.nbytes for _, v in tree_flatten(model.parameters())) / 1e9
        extra = f", attention+lm_head at {dense_bits}-bit" if dense_bits and dense_bits < 16 else ""
        print(f"[inkling2-mini] loaded {gb:.2f} GB in {time.perf_counter() - t0:.1f}s{extra}")
    return model
