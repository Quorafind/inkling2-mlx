"""Confidence-gated DSpark loop (speculative.ConfidencePolicy): bookkeeping and output.

1. Draft-feature bookkeeping. After every generation, the draft's global-attention
   cache must have ingested exactly the target tokens whose features are not still
   pending: draft offset + pending features == processed target tokens. A gated round
   that falls back to plain decoding after drafting must not feed the same features
   twice, and none may be skipped. (A bookkeeping bug would only lower acceptance,
   never change the output, so it has to be checked directly.)
2. Greedy output vs plain greedy decoding: identical up to genuine near-ties (verify
   and single-token decode use different kernels). At the first divergence, the plain
   model's top-1 / top-2 logit margin over the common prefix is printed.
3. Two turns in one GenState (multi-turn reuse with gated rounds in both).

    python tests/check_gated_spec.py MODEL_DIR
"""

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini, user_message  # noqa: E402

bot = Inkling2Mini(sys.argv[1], verbose=False)
lm = bot.model.language_model
GLOBAL = [i for i, layer in enumerate(bot.drafter.layers) if layer.self_attn.window is None][0]


def check_features(state, label):
    pending = 0 if state.hidden is None else state.hidden.shape[1]
    offset = state.draft_cache[GLOBAL].offset
    ok = offset + pending == state.length
    print(f"  {label}: draft offset {offset} + pending {pending} vs target length {state.length}: "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def margin_at(ids, prefix):
    seq = mx.concatenate([ids.reshape(1, -1), mx.array([prefix], mx.int32)], axis=1)
    logits = lm(seq, cache=bot.model.make_cache()).astype(mx.float32).reshape(-1)
    top = mx.topk(logits, 2)
    return float(mx.abs(top[1] - top[0]).item())


prompts = [
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and tests."),
    ("zh", "写一段 300 字左右的秋天散文。"),
    ("story", "Write a 200-word story about a lighthouse keeper."),
]
all_ok = True
for name, text in prompts:
    ids, _, _ = bot.encode([{"role": "user", "content": text}], reasoning_effort="none")
    plain = [t for t, _ in bot.generate_tokens(ids, max_tokens=240, temperature=0.0, speculative=False)]
    state = bot.new_state()
    stats = None
    spec = []
    for t, st in bot.generate_tokens(ids, max_tokens=240, temperature=0.0, state=state):
        spec.append(t)
        stats = st or stats
    print(f"{name}: {stats['gen_tps']:.1f} tok/s, {stats.get('tokens_per_round', 0):.2f} tok/round, "
          f"{stats.get('plain_fallbacks', 0)} plain fallbacks")
    all_ok &= check_features(state, "features")
    j = next((i for i, (a, b) in enumerate(zip(plain, spec)) if a != b), None)
    if j is None and len(plain) == len(spec):
        print("  output: identical to plain greedy decoding")
    else:
        j = j if j is not None else min(len(plain), len(spec))
        m = margin_at(ids, plain[:j])
        tie = m < 0.5
        print(f"  output: first divergence at token {j} of {len(plain)}, plain top-2 margin {m:.3f} "
              f"({'near-tie' if tie else 'NOT a near-tie'})")
        all_ok &= tie

# Two turns on one state; the second turn starts from the first turn's pending features.
state = bot.new_state()
msgs = [user_message("用三句话介绍一下杭州。", [])]
ids, _, _ = bot.encode(msgs, reasoning_effort="none")
reply = [t for t, _ in bot.generate_tokens(ids, max_tokens=120, temperature=0.0, state=state)]
all_ok &= check_features(state, "turn 1")
ids2 = mx.array([bot.tokenizer.encode("\n再写一段关于西湖的。")])
reply2 = [t for t, _ in bot.generate_tokens(ids2, max_tokens=120, temperature=0.0, state=state)]
all_ok &= check_features(state, "turn 2")

print("PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
