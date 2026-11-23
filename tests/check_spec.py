"""Correctness of speculative verification and cache rollback.

  oracle drafts  (the plain greedy continuation): every draft must be accepted
  garbage drafts (token 0): nothing accepted, every round rolls back 7 positions
Both must reproduce plain greedy decoding, up to genuine near-ties. At the first
divergence we print the plain-decode top-1 / top-2 logit margin.

    python tests/check_spec.py MODEL_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini import fused  # noqa: E402
from inkling2_mini.generate import Inkling2Mini  # noqa: E402

# Decode and verification then use the same (chronological) attention layout, so any
# difference is a rollback bug, not reduction order. The in-place ring decode layout
# is checked separately by tests/check_ring.py and the tiny parity test.
fused.RING_DECODE = False

N = 120
bot = Inkling2Mini(sys.argv[1])
lm = bot.model.language_model
msgs = [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively."}]
ids, _, _ = bot.encode(msgs, reasoning_effort="none")

# Plain greedy, recording margins.
cache = bot.model.make_cache()
logits = lm(ids, cache=cache)
plain, margins = [], []
for _ in range(N):
    top = mx.sort(logits.reshape(-1))[-2:]
    margins.append(float(top[1] - top[0]))
    tok = int(mx.argmax(logits.reshape(-1)))
    plain.append(tok)
    logits = lm(mx.array([[tok]]), cache=cache)

# One-shot teacher forcing over the plain sequence: how often does argmax agree?
full = lm(mx.concatenate([ids, mx.array([plain])], axis=1), last_only=False)
forced = mx.argmax(full[0, ids.shape[1] - 1 : -1], axis=-1).tolist()
agree = sum(a == b for a, b in zip(forced, plain))
print(f"one-shot prefill vs step decode argmax agreement: {agree}/{N}")
for i, (a, b) in enumerate(zip(forced, plain)):
    if a != b:
        print(f"  first mismatch at {i}: plain margin {margins[i]:.3f}")
        break

drafter = bot.drafter
real_draft = drafter.draft_block


def run(kind):
    pos = {"n": 0}

    def fake(b, hidden, cache, bs, sampler, dtype):
        real = real_draft(b, hidden, cache, bs, sampler, dtype)  # keep the draft KV consistent
        if kind == "oracle":
            k = pos["n"]
            row = (plain[k + 1 : k + bs] + [0] * bs)[: bs - 1]
            return mx.array([row], dtype)
        return mx.zeros_like(real)

    drafter.draft_block = fake
    out = []
    for tok, _ in bot.generate_tokens(ids, max_tokens=N, temperature=0):
        out.append(tok)
        pos["n"] = len(out) - 1
    drafter.draft_block = real_draft
    same = next((i for i, (a, b) in enumerate(zip(out, plain)) if a != b), len(out))
    msg = f"{kind:8s} drafts: identical prefix {same}/{N}"
    if same < N:
        msg += f"  (plain margin there {margins[same]:.3f})"
    print(msg)


run("oracle")
run("garbage")
