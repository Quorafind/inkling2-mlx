"""Multi-turn KV reuse (GenState / Session) and session save / load.

1. Tiny reference model, CPU FP32 (strict): two greedy turns continuing one state
   produce exactly the tokens of a fresh run over the concatenated token stream.
2. Real weights: turn 2 through the reused caches vs a one-shot prefill of the
   same token stream (next-token KL at BF16 noise level), and the prefill cost.
3. Real weights: save after turn 1, load, run turn 2 greedily: identical to the
   in-memory session.

    python tests/check_session.py TINY_DIR MODEL_DIR
"""

import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling2_mini.generate import Inkling2Mini, Session  # noqa: E402
from inkling2_mini.loader import load  # noqa: E402

tiny, model_dir = Path(sys.argv[1]), Path(sys.argv[2])

# 1. strict state continuation on the tiny model
with mx.stream(mx.cpu):
    t = Inkling2Mini.__new__(Inkling2Mini)
    t.model, t.drafter, t.eos_id, t.kv_bits = load(tiny, verbose=False), None, -1, None
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 500, (1, 11)), rng.integers(0, 500, (1, 6))
    state = t.new_state()
    turn1 = [tok for tok, _ in t.generate_tokens(mx.array(a), max_tokens=9, temperature=0, speculative=False, state=state)]
    turn2 = [tok for tok, _ in t.generate_tokens(mx.array(b), max_tokens=12, temperature=0, speculative=False, state=state)]
    stream = mx.array(np.concatenate([a, [turn1], b], axis=1))
    fresh = [tok for tok, _ in t.generate_tokens(stream, max_tokens=12, temperature=0, speculative=False)]
    ok1 = turn2 == fresh and state.length + len(state.pending) == stream.shape[1] + len(turn2)
    print(f"tiny, CPU FP32: continued state == fresh prefill: {turn2 == fresh}  "
          f"(context {state.length}+{len(state.pending)} pending) {'ok' if ok1 else 'MISMATCH'}")

# 2. / 3. real model
bot = Inkling2Mini(model_dir)
lm = bot.model.language_model
q1 = "Here is a list: apples, pears, plums, figs, dates. Name the third item and explain briefly what it is."
q2 = "Now name the last item and say which of the five is your favourite."
for spec in (False, True):
    s = Session(bot, reasoning_effort="none")
    r1 = s.send(q1, max_tokens=200, temperature=0, speculative=spec)
    before = s.context_tokens
    t0 = time.perf_counter()
    r2 = s.send(q2, max_tokens=60, temperature=0, speculative=spec)
    print(f"real, speculative={spec}: turn 1 {len(r1.tokens)} tok; turn 2 prefilled {r2.stats['prompt_tokens']} new tokens "
          f"(context {before}) in {r2.stats['prefill_s']:.2f} s -> {r2.text[:80]!r}")

# KL of turn 2's first token: reused caches vs one-shot prefill of the same stream.
s = Session(bot, reasoning_effort="none")
r1 = s.send(q1, max_tokens=200, temperature=0, speculative=False)
first_ids = bot.encode([{"role": "user", "content": q1}], reasoning_effort="none")[0][0].tolist()
convo_text = bot.render(s.messages + [{"role": "user", "content": q2}], "none")
suffix_ids = bot.processor(text=convo_text[len(s._rendered):])["input_ids"][0].tolist()
stream = first_ids + r1.tokens + s.state.pending[1:] + suffix_ids if r1.tokens[-1] != bot.eos_id else first_ids + r1.tokens + suffix_ids
cache = bot.model.make_cache()
one_shot, _ = bot._prefill(lm.model.embed(mx.array([stream])), cache, 512, None)
state = s.state
reuse, _ = bot._prefill(lm.model.embed(mx.array([state.pending + suffix_ids])), state.cache, 512, None, ctx=state.length)
p = np.asarray(mx.softmax(one_shot.reshape(-1).astype(mx.float32)))
q = np.asarray(mx.softmax(reuse.reshape(-1).astype(mx.float32)))
kl = float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())
print(f"real: turn-2 first-token KL(one-shot || reused) {kl:.1e}, argmax equal {p.argmax() == q.argmax()} "
      f"(stream {len(stream)} tokens; the reused run prefills {len(state.pending) + len(suffix_ids)})")

# Save / load round trip.
s1 = Session(bot, reasoning_effort="none")
s1.send(q1, max_tokens=120, temperature=0)
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "session.safetensors"
    s1.save(path)
    size = path.stat().st_size / 1e6
    s2 = Session.load(bot, path)
    a2 = s2.send(q2, max_tokens=50, temperature=0)
a1 = s1.send(q2, max_tokens=50, temperature=0)
print(f"real: save/load ({size:.0f} MB file): turn 2 identical after reload: {a1.tokens == a2.tokens}")
