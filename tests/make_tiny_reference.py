"""Build a tiny random Inkling2Loop with the transformers reference implementation,
save it in the original checkpoint layout (save_pretrained reverses the conversion
mapping), and record reference logits for the MLX parity test.

    python tests/make_tiny_reference.py OUT_DIR
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Inkling2LoopConfig, Inkling2LoopForConditionalGeneration

out = Path(sys.argv[1])
torch.manual_seed(0)

FULL = {1, 3, 7}  # pre-loop, in-loop and post-loop full-attention layers
N_LAYERS = 8
text = dict(
    vocab_size=512,
    unpadded_vocab_size=500,
    hidden_size=128,
    num_hidden_layers=N_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=32,
    sliding_window_size=8,
    d_rel=8,
    rel_extent=16,
    log_scaling_n_floor=4,
    layer_types=["hybrid" if i in FULL else "hybrid_sliding" for i in range(N_LAYERS)],
    n_layers_per_attn_res_block=2,
    moe_intermediate_size=64,
    n_routed_experts=8,
    num_experts_per_tok=2,
    loop_layer_index_start=2,
    loop_layer_index_end=6,
    loop_num_passes=3,
    loop_gate_rank=16,
)
vision = dict(
    model_type="inkling2_vision",
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    projection_hidden_dim=128,
    posemb_grid_size=4,
    text_hidden_size=128,
)
IMAGE_TOKEN = 501  # inside the tiny vocab
config = Inkling2LoopConfig(text_config=text, vision_config=vision, image_token_id=IMAGE_TOKEN)
model = Inkling2LoopForConditionalGeneration(config).float().eval()

# Random everything, at scales where every path (gates, messages, queries, softcaps) matters.
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            std = 1.0 / math.sqrt(p.shape[-1])
            if "query" in name or "proj" == name.split(".")[-1]:
                std = 1.0
            p.normal_(0.0, std)
        elif name.endswith("bias") and "e_score" not in name:
            p.normal_(0.0, 0.1)
        else:
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
    for name, b in model.named_buffers():
        if "e_score_correction_bias" in name:
            b.normal_(0.0, 0.1)

ids = torch.randint(0, 500, (1, 24))
with torch.no_grad():
    logits = model(input_ids=ids).logits.float().numpy()
print("reference logits", logits.shape, float(np.abs(logits).mean()))


def set_passes(n):
    for m in [model, *model.modules()]:
        cfg = getattr(m, "config", None)
        for c in (cfg, getattr(cfg, "text_config", None)):
            if c is not None and hasattr(c, "loop_num_passes"):
                c.loop_num_passes = n


# Other pass counts (variable-loop checkpoints choose them at inference time).
pass_logits = {}
with torch.no_grad():
    for n in (1, 2, 4, 5):
        set_passes(n)
        pass_logits[n] = model(input_ids=ids).logits.float().numpy()
    set_passes(3)
    assert np.allclose(model(input_ids=ids).logits.float().numpy(), logits)
print("reference logits at 1, 2, 4, 5 passes:",
      ", ".join(f"{n}: {float(np.abs(v - logits).max()):.2f} from 3" for n, v in pass_logits.items()))

# Multimodal: one 100x130 image -> 3x4 grid of 40px patches -> 12 image tokens.
from mlx_vlm.models.inkling2.processing_inkling2 import Inkling2ImageProcessor  # noqa: E402

rng = np.random.default_rng(0)
image = rng.integers(0, 256, (100, 130, 3), dtype=np.uint8)
proc = Inkling2ImageProcessor()(image)
pixels = np.asarray(proc["pixel_values"]).astype(np.float32)
grid = np.asarray(proc["image_grid_thw"]).astype(np.int64)
n_img = int(grid.prod())
mm_ids = torch.cat([ids[:, :5], torch.full((1, n_img), IMAGE_TOKEN), ids[:, 5:12]], dim=1)
with torch.no_grad():
    vision_feats = model.model.vision_tower(
        pixel_values=torch.tensor(pixels), image_grid_thw=torch.tensor(grid)
    ).pooler_output.float().numpy()
    mm_logits = model(
        input_ids=mm_ids, pixel_values=torch.tensor(pixels), image_grid_thw=torch.tensor(grid)
    ).logits.float().numpy()
print("reference vision", vision_feats.shape, "mm logits", mm_logits.shape)

out.mkdir(parents=True, exist_ok=True)
np.save(out / "ref_pixels.npy", pixels)
np.save(out / "ref_grid.npy", grid)
np.save(out / "ref_mm_ids.npy", mm_ids.numpy())
np.save(out / "ref_vision.npy", vision_feats)
np.save(out / "ref_mm_logits.npy", mm_logits)
model.save_pretrained(out, safe_serialization=True)
np.save(out / "ref_ids.npy", ids.numpy())
np.save(out / "ref_logits.npy", logits)
for n, v in pass_logits.items():
    np.save(out / f"ref_logits_p{n}.npy", v)

# Rewrite config.json in the original release format that the MLX loader reads.
release = dict(
    model_type="inkling2_loop",
    eos_token_id=200006,
    image_token_id=IMAGE_TOKEN,
    text_config={
        **{k: v for k, v in text.items() if not k.startswith("loop_") and k != "moe_intermediate_size"},
        "intermediate_size": text["moe_intermediate_size"],
        "loop_config": dict(core_start=2, core_end=6, num_passes=3, gate_rank=16, is_varloop=False),
    },
    vision_config={**vision, "native_patch_size": 20, "adapter_merge_size": 2},
)
(out / "config.json").write_text(json.dumps(release, indent=2))
