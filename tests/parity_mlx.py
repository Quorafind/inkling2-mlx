"""Compare the MLX implementation with the transformers reference logits.

    python tests/parity_mlx.py TINY_DIR      (made by make_tiny_reference.py)

Strict check: FP32 on the CPU stream (the M-series GPU's FP32 matmuls run on the
neural accelerators at ~1e-3 relative precision, which is too loose for parity).
Informational: BF16 on the GPU (random MoE routers make this chaotic).
Checks:
  1. one-shot forward over the whole sequence
  2. chunked prefill + token-by-token decode through the KV / conv caches
     (sequence 24 > sliding window 8, so the rotating caches wrap)
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.loader import load  # noqa: E402

tiny = Path(sys.argv[1])
ids = np.load(tiny / "ref_ids.npy")
ref = np.load(tiny / "ref_logits.npy")


def compare(name, got, want, tol):
    got = np.asarray(got.astype(mx.float32))
    err = np.abs(got - want).max()
    rel = err / np.abs(want).max()
    agree = (got.argmax(-1) == want.argmax(-1)).mean()
    ok = tol is None or rel < tol
    print(f"  {'PASS' if tol and ok else 'FAIL' if tol else 'info'} {name:28s} max|diff|={err:.2e} rel={rel:.2e} argmax-agree={agree:.3f}")
    return ok


ok = True
for dtype, device, tol in ((mx.float32, mx.cpu, 1e-4), (mx.bfloat16, mx.gpu, None)):
    print(f"dtype={dtype} device={device}")
    mx.set_default_device(device)
    model = load(tiny, verbose=False, dtype=dtype)
    lm = model.language_model
    x = mx.array(ids)

    full = lm(x, last_only=False)
    ok &= compare("one-shot forward", full, ref, tol)

    cache = model.make_cache()
    split = 13
    steps = [lm(x[:, :5], cache=cache, last_only=False), lm(x[:, 5:split], cache=cache, last_only=False)]
    for t in range(split, ids.shape[1]):
        steps.append(lm(x[:, t : t + 1], cache=cache, last_only=False))
    inc = mx.concatenate(steps, axis=1)
    ok &= compare("chunked prefill + decode", inc, ref, tol)

    pixels = mx.array(np.load(tiny / "ref_pixels.npy")).astype(dtype)
    grid = mx.array(np.load(tiny / "ref_grid.npy"))
    vision = model.vision_tower(pixels, image_grid_thw=grid)
    ok &= compare("vision tower features", vision, np.load(tiny / "ref_vision.npy"), tol)
    mm_ids = mx.array(np.load(tiny / "ref_mm_ids.npy"))
    h = model.embed(mm_ids, pixels, grid)
    mm = lm(input_embeddings=h, last_only=False)
    ok &= compare("image + text forward", mm, np.load(tiny / "ref_mm_logits.npy"), tol)

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
