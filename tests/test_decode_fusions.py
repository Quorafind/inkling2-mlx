"""The single-token decode fusions of 2026-11-23 against the paths they replace.

1. moe_kernels.moe_decode_conv (router GEMV -> gate/up with top-k -> down + route sum +
   short conv + residual) matches the verify path (moe_kernels.route + gather_qmm +
   combine_routes + short_conv) up to a few last-bit roundings, on NVFP4 and MXFP8 layers.
2. depth_kernels.depth_mix matches kernels.attention_residual (FP32 summation order only):
   1 and 3 queries, mixed and normed outputs, 1 and 4 tokens, and the fused post-attention
   conv + residual for one token.
3. The decode attention with the fused epilogue (relative bias, sinks, output gate) matches
   the unfused one over 24 teacher-forced decode steps, at short context and past the
   sliding window (the ring wraps).
4. Greedy DSpark output from two clones of one state is identical (the verify lengths no
   longer depend on wall-clock timing).

    python tests/test_decode_fusions.py MODEL_DIR
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402

from inkling2_mini import decode_attention as q8  # noqa: E402
from inkling2_mini import depth_kernels as DK  # noqa: E402
from inkling2_mini import kernels as K  # noqa: E402
from inkling2_mini import moe_kernels as MK  # noqa: E402
from inkling2_mini import prefill_kernels as PK  # noqa: E402
from inkling2_mini.generate import GenState, Inkling2Mini  # noqa: E402

bot = Inkling2Mini(sys.argv[1], verbose=False, speculative=True)
lm = bot.model.language_model
layers = lm.model.layers
H = lm.config.hidden_size
ok = True


def check(name, good, detail):
    global ok
    ok &= bool(good)
    print(f"  {name}: {detail} {'ok' if good else 'FAIL'}", flush=True)


# 1. experts
print("fused MoE tail vs route + gather_qmm + combine_routes + short_conv (the verify path)")
worst = (0, 0)
for seed in range(3):
    mx.random.seed(seed)
    x = (mx.random.normal((1, 1, H)) * 0.5).astype(mx.bfloat16)
    res = mx.random.normal((1, 1, H)).astype(mx.float32)
    st = mx.random.normal((1, 3, H)).astype(mx.float32)
    for i in range(len(layers)):
        m, cw = layers[i].mlp, layers[i].mlp_sconv.conv.weight
        sw = m.switch_mlp
        idx, w = MK.route(x, m.gate_weight, m.e_score_correction_bias, m.top_k, m.route_scale)
        xe = mx.expand_dims(x, (-2, -3))
        gu = sw.gate_up(xe, idx)
        out = sw.down_proj(sw.activation(gu[..., m.inter:], gu[..., :m.inter]), idx)
        moe = PK.combine_routes(out.reshape(-1, H), mx.arange(m.top_k, dtype=mx.uint32), w.reshape(1, -1))
        ref = PK.short_conv(moe.reshape(1, 1, H), st, cw, residual=res, out_dtype=mx.bfloat16, with_state=True)
        got = MK.moe_decode_conv(x, m, st, cw, res, mx.bfloat16)
        a, r = got[1].reshape(3, H)[2], ref[1].reshape(3, H)[2]  # the MoE output, kept in the conv state
        n = (a != r).sum().item()
        ab = a.astype(mx.bfloat16).view(mx.int16).astype(mx.int32)
        rb = r.astype(mx.bfloat16).view(mx.int16).astype(mx.int32)
        ulps = mx.abs(ab - rb).max().item()  # same sign here: a distance in BF16 steps
        worst = (max(worst[0], n), max(worst[1], ulps))
# the dot products and the route sum follow MLX's order; what is left are a few last-bit
# rounding differences, one or two BF16 steps on a handful of values (the previous
# single-token kernel had the same ones)
check("24 layers x 3 inputs", worst[0] <= 16 and worst[1] <= 2,
      f"at most {worst[0]} of {H} values differ in a layer, by at most {worst[1]} BF16 steps:")

# 2. depth mix
print("depth mix vs attention_residual")
w3 = (mx.random.normal((3, H)) * 0.1 + 1).astype(mx.bfloat16)
for T in (1, 4):
    for S in (2, 9):
        states = [mx.random.normal((1, T, H)).astype(mx.float32) * (1 + i) for i in range(S)]
        for Q in (1, 3):
            q = (mx.random.normal((Q, H)) * 0.05).astype(mx.float32)
            w = w3[:Q]
            mixed_ref, normed_ref = K.attention_residual(states, q, w if Q > 1 else w[0])
            DK.begin(T)
            got = DK.depth_mix(states[:-1], q, last=states[-1], norm_weight=w if Q > 1 else w[0], write_mixed=True)
            DK.end()
            e1 = (mx.abs(got["mixed"] - mixed_ref).max() / mx.abs(mixed_ref).max()).item()
            e2 = (mx.abs(got["normed"].astype(mx.float32) - normed_ref.astype(mx.float32)).max()).item()
            check(f"T={T} S={S} Q={Q}", e1 < 1e-5 and e2 <= 2 ** -6, f"mixed rel {e1:.1e}, normed max abs {e2:.1e}:")
attn = (mx.random.normal((1, 1, H)) * 0.5).astype(mx.bfloat16)
cw = layers[3].attn_sconv.conv.weight
blocks = [mx.random.normal((1, 1, H)).astype(mx.float32) for _ in range(4)]
q1 = (mx.random.normal((H,)) * 0.05).astype(mx.float32)
for base in (None, (mx.random.normal((1, 1, H))).astype(mx.bfloat16)):
    r_ref, s_ref = PK.short_conv(attn, st, cw, residual=base, out_dtype=mx.float32, with_state=True)
    _, n_ref = K.attention_residual([*blocks, r_ref], q1, w3[0])
    DK.begin(1)
    got = DK.depth_mix(blocks, q1, norm_weight=w3[0], conv=(attn, st, cw, base))
    DK.end()
    e = (mx.abs(got["residual"] - r_ref).max()).item()
    s_same = (got["state"] == s_ref).all().item()
    e2 = (mx.abs(got["normed"].astype(mx.float32).reshape(-1) - n_ref.astype(mx.float32).reshape(-1)).max()).item()
    check(f"fused conv, residual {'added' if base is not None else 'none'}", e == 0 and s_same and e2 <= 2 ** -6,
          f"residual max abs {e:.1e}, state identical {s_same}, normed max abs {e2:.1e}:")

# 3. attention epilogue, teacher-forced
print("fused attention epilogue vs unfused, 24 teacher-forced decode steps")
short = bot.encode([{"role": "user", "content": "写一段 300 字左右的秋天散文。"}], reasoning_effort="none")[0]
doc = (Path(__file__).resolve().parents[1] / "README.md").read_text()[:6000]
long = bot.encode([{"role": "user", "content": doc + "\n\nSummarize."}], reasoning_effort="none")[0]


def decode(ids, force=None, n=24):
    cache = bot.model.make_cache(kv_bits=bot.kv_bits)
    logits = lm(ids, cache=cache)
    toks, outs = [], []
    for i in range(n):
        lg = logits[0, -1].astype(mx.float32)
        t = int(mx.argmax(lg).item()) if force is None else force[i]
        outs.append(lg)
        toks.append(t)
        logits = lm(mx.array([[t]]), cache=cache)
    outs.append(logits[0, -1].astype(mx.float32))
    return toks, mx.stack(outs)


for name, ids in (("short", short), ("long", long)):
    q8.FUSED_EPILOGUE = False
    toks, ref = decode(ids)
    q8.FUSED_EPILOGUE = True
    _, new = decode(ids, toks)
    lp = ref - mx.logsumexp(ref, -1, keepdims=True)
    lq = new - mx.logsumexp(new, -1, keepdims=True)
    kl = (mx.exp(lp) * (lp - lq)).sum(-1)[1:]
    agree = (mx.argmax(ref, -1) == mx.argmax(new, -1))[1:].mean().item()
    # the bias is rounded to BF16 as in the unfused path, but its FP32 sum runs in another
    # order, so a value near a rounding boundary moves by one BF16 step (0.03 at |bias| ~ 4);
    # over 1,900 keys x 48 visits that gives a few 1e-3 of KL, well under 6-bit dense (8e-3)
    check(f"{name} ({ids.shape[1]} prompt tokens)", kl.mean().item() < 5e-3 and agree == 1.0,
          f"mean KL {kl.mean().item():.1e}, top-1 agree {agree * 100:.0f} %:")

# 4. reproducible DSpark output
print("greedy DSpark from two clones of one state")
q1 = bot.encode([{"role": "user", "content": "用两句话介绍一下杭州。"}], reasoning_effort="low")[0]
follow = mx.array([bot.tokenizer.encode("再用一句话说说西湖。")])
state = GenState(cache=bot.model.make_cache(kv_bits=bot.kv_bits))
for _ in bot.generate_tokens(q1, state=state, max_tokens=120, temperature=0.0):
    pass
outs = [[t for t, _ in bot.generate_tokens(follow, state=state.clone(), max_tokens=160, temperature=0.0)]
        for _ in range(2)]
check("two clones", outs[0] == outs[1], f"{len(outs[0])} tokens identical:")

print("DECODE FUSIONS OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
