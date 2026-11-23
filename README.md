# Inkling 2 Mini on Apple silicon (MLX)

A hand-written MLX runtime for Thinking Machines' **Inkling 2 Mini**
(`thinkingmachines/Inkling-2-Mini`, model type `inkling2_loop`, released 2026-11-20).

- **Native experts.** The NVFP4 / MXFP8 experts run as native MLX quantized matmuls, exactly
  as shipped.
- **6-bit dense layers by default.** Attention projections and lm_head run at affine 6 bits.
  That is about 7 % faster than 8 bits, with an error about 1.5× the model's own BF16 noise
  (item 9 under Performance). `--dense-bits 8` is below that noise, and `--dense-bits 16`
  keeps the checkpoint's BF16.
- **Speculative decoding.** The bundled **DSpark** draft drives speculative decoding. Each
  round's verify length comes from the draft's own token confidence (item 10), so code runs
  about 20 % faster than plain decoding and prose runs at plain speed instead of losing 5–9 %.
- **Batching and images.** Several requests can decode in one batch, and image input works.
- **KV reuse across turns.** Conversations keep their KV caches between turns, can be saved
  to disk, and can quantize the full-attention KV caches for very long contexts.

At the time of writing, mlx-vlm 0.8 supports Inkling 2 Small (`inkling2`) but not the
looped Mini (`inkling2_loop`). This project reuses mlx-vlm's per-layer Inkling 2 blocks and
its DSpark draft network, and adds:

- the loop, the message gates, the weight loader and the 48-entry cache layout;
- visit-indexed feature capture, the speculative loop and its adaptive policy;
- fused decode kernels, sink-slot and quantized caches, batched decoding, and multi-turn sessions.

## Quick start

You need an Apple silicon Mac with 32 GB of unified memory or more (the model stays
resident at 18.9 GB), Python 3.12 and, for the server, a Rust toolchain (`cargo`).

```bash
git clone https://github.com/Quorafind/inkling2-mlx && cd inkling2-mlx
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt

# Weights, about 20 GB, into models/Inkling-2-Mini (either source works)
hf download thinkingmachines/Inkling-2-Mini --local-dir models/Inkling-2-Mini
# modelscope download --model thinkingmachines/Inkling-2-Mini --local_dir models/Inkling-2-Mini

./inkling2          # builds the launcher on first use, serves http://127.0.0.1:8000/v1
```

Any OpenAI-compatible client can then use it:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=local
curl $OPENAI_BASE_URL/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Hi"}]}'
```

`./inkling2 chat` opens the terminal chat instead. `launcher/README.md` lists the
endpoints and options.

## Compared with Strata, and why this design

[Strata](https://github.com/Niko1221/Strata) also runs Inkling 2 Mini locally. It is a C++
engine for several models and GPU vendors, with tiered VRAM / RAM / SSD expert placement and
a desktop app. This project only targets one model on Apple silicon.
`docs/strata-comparison.md` (in Chinese) has the full side-by-side comparison, with sources.

Measured on one Mac mini M6 (32 GB) with the same official safetensors and the same token
ids:

| | Strata 0.1.66 (with a local fix) | This project, BF16 dense | This project, 6-bit dense (default) |
| --- | --- | --- | --- |
| Plain decode | 21.7 tok/s | 20.5 tok/s | 30.1 tok/s |
| DSpark on code | 22–26 tok/s | 20–29 tok/s | 39–40 tok/s |
| Prefill 8k / 32k | 718 / 627 tok/s | 967 / 867 tok/s | 879 / 805 tok/s |
| Mean KL vs CPU-FP32 truth | 9.3e-4 | 4.3e-3 | 8.2e-3 |

- **Strata's official-safetensors path on Metal prints garbage as released.** Its Metal
  kernels read small BF16 tensors (norm weights, scales, short-conv taps) as FP32.
  `docs/strata-metal-safetensors-fix.diff` fixes it locally, and the numbers above use that
  fix. Strata's own `.strata` packs, which its Mac measurements use, are a separate path and
  were not tested here.
- **Strata is more precise** because it keeps activations in FP32. Perplexity on the test
  text is 5.11 for Strata and 5.08–5.13 for this project, against 5.110 for the truth.

Why this project is built the way it is:

- **Experts stay at their trained NVFP4 / MXFP8 precision.** On unified memory every expert
  is already resident, so there is nothing to tier. Lower-bit expert packs would read fewer
  bytes but change the model.
- **Only the dense part is quantized post hoc, to 6 bits by default.** Decode is bound by
  memory bandwidth, so bytes per token set the speed: 3.8 GB at 6 bits against 6.4 GB in
  BF16. The precision cost is measured, and `--dense-bits 8 / 16` trade it back.
- **MLX instead of a new engine.** Apple's GEMM kernels give the fastest prefill here, which
  matters most for long documents. Custom Metal kernels cover the decode glue.
- **DSpark verifies only when it pays.** On this MoE with a 3-pass loop, each extra verified
  token costs about 30 % of a step. The verify length comes from the draft's own confidence,
  and the runtime falls back to plain decoding. Prose runs at plain-decoding speed (−2 % to
  +3 %).
- **Stateful serving.** Prefill dominates multi-turn latency, so conversation states are
  kept and reused. That covers Chat Completions prefix matching, Responses
  `previous_response_id` and system-prompt snapshots. Turn 2 of an 8k conversation starts in
  0.3 s instead of 9.3 s.
- **A Rust launcher in front of a Python worker.** HTTP, streaming, cancellation and worker
  restarts stay robust, and the model code stays in Python with MLX.

## Usage

Local OpenAI-compatible server (Chat Completions + Responses), via the Rust launcher in
`launcher/` (see `launcher/README.md`):

```bash
./inkling2                 # serves http://127.0.0.1:8000/v1; OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

Terminal chat:

```bash
.venv/bin/python chat.py -p "Explain looped transformers in two sentences."
.venv/bin/python chat.py -p "What is in this picture?" --image photo.jpg
.venv/bin/python chat.py                       # interactive; /reset, /save PATH, /load PATH, /quit
.venv/bin/python chat.py --file paper.txt      # long document as context, then ask follow-ups
.venv/bin/python chat.py --session doc.safetensors   # resume a saved conversation
.venv/bin/python chat.py --batch prompts.txt   # one prompt per line, decoded together
```

In interactive mode every turn reuses the conversation's KV caches and prefills only the new
message (see "KV cache and long contexts").

| Flag | Meaning |
| --- | --- |
| `--effort` | The chat template's `reasoning_effort`: `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`, or 0 to 0.99. The template default is 0.9, which thinks for a long time; `chat.py` defaults to `low`. |
| `--dense-bits` | Precision of the attention projections and lm_head, affine, group 64. `6` (default): fastest; median KL vs FP32 truth 2.5× that of BF16, top-1 unchanged. `8`: within BF16 noise, about 7 % slower. `16`: the checkpoint's BF16. `5` / `4`: smaller and lossy. |
| `--batch FILE` | Decodes every line of FILE as its own request, in one batch (plain decoding, no DSpark). |
| `--file FILE` | Prepends FILE to the first message, for long-document Q&A. Follow-up questions reuse its KV. |
| `--session PATH` | Resumes a conversation saved with `/save PATH`. |
| `--kv-bits` | Precision of the full-attention KV caches. `8` (default): affine 8-bit read by a fused decode kernel; about 22 % faster decode at 64k and half the KV memory. `16`: BF16. `4`: quarter memory, lossy. Batched decoding always uses BF16. |
| `--no-spec` | Turns DSpark speculative decoding off. It is on by default when `mtp.safetensors` is present. |
| `--block-size N` | Uses a fixed DSpark verify block of 2 to 8 instead of the confidence-gated policy. |
| `--hide-thinking`, `--temp`, `--top-p`, `--top-k`, `--seed`, `--system`, `--max-tokens` | The usual options. |

From Python:

```python
from inkling2_mini.generate import Inkling2Mini
bot = Inkling2Mini("models/Inkling-2-Mini")            # dense_bits=6 by default
reply = bot.chat([{"role": "user", "content": "Hi!"}], reasoning_effort="low", max_tokens=512)
replies, stats = bot.chat_batch([[{"role": "user", "content": q}] for q in questions],
                                reasoning_effort="none", max_tokens=256)

from inkling2_mini.generate import Session
s = Session(bot, reasoning_effort="low")         # kv_bits=8 for very long contexts
s.send(open("paper.txt").read() + "\n\nSummarize section 2.")   # prefills the document once
s.send("And section 3?")                         # prefills only this message
s.save("paper.safetensors"); s = Session.load(bot, "paper.safetensors")
```

## Layout

| Path | What it is |
| --- | --- |
| `inkling2_mini/model.py` | Loop model: `MessageGate`, `Layer` / `LoopDecoderLayer` (fused decode path), `TextModel` (3 passes, visit capture), `GroupedSparseMoE` (fused gate/up), cache layout |
| `inkling2_mini/loader.py` | Original checkpoint → MLX: per-layer NVFP4 / MXFP8 experts, name mapping, `quantize_dense`, fusing |
| `inkling2_mini/kernels.py` | Metal kernels: fused block attention residual + RMSNorm, and the grouped NVFP4 expert matvec |
| `inkling2_mini/prefill_kernels.py` | Prefill glue kernels: causal short convolutions (static and dynamic taps, with the residual fused), MoE un-sort + weighted combine, and the message gates (prep + apply / RMSNorm) |
| `inkling2_mini/decode_attention.py` | Fused split-K decode / verify attention (up to 8 queries) over the 8-bit KV cache, with band bias and per-query sinks (an adapted copy of MLX's `sdpa_vector_2pass` GQA kernels), and the single-token sliding-window variant that reads the BF16 ring in its physical layout |
| `inkling2_mini/decode_kernels.py` | Decode glue kernels: router top-k, attention prep (L2 clips, value norm, log scaling, layout), single-token NVFP4 experts (gate_up + SwiGLU, down + weighted sum); `ENABLED` switches them for A/B tests |
| `inkling2_mini/band_attention.py` | Prefill flash attention for full-attention layers, with the relative-position bias computed only inside its 1,024-token band (an adapted copy of MLX's NAX attention kernel) |
| `inkling2_mini/fused.py` | `FusedAttention` (one projection matrix), batched compiled message gates |
| `inkling2_mini/caches.py` | `SinkKVCache` / `SinkRingKVCache` (the attention-sink slot lives inside the cache) and `QuantSinkKVCache` (8/4-bit full-attention KV) |
| `inkling2_mini/speculative.py` | DSpark draft binding (48 visits, 4-bit draft-side head), `ConfidencePolicy` (default: per-round verify length from draft confidence), `BlockPolicy` (fixed blocks; the earlier geometric policy via `generate.SPEC_POLICY = "geometric"`) |
| `inkling2_mini/generate.py` | Chat template, memory-bounded chunked prefill, sampling, plain / speculative / batched loops, `GenState`, `Session` (multi-turn KV reuse, save / load), streaming events |
| `chat.py` | Command-line chat and batch mode |
| `inkling2`, `launcher/` | Rust launcher and local OpenAI-compatible server. `./inkling2` builds it on first use |
| `docs/strata-comparison.md` | Comparison with Strata (Niko1221/Strata), which also runs Inkling 2 Mini locally |
| `inkling2_mini/server_worker.py` | The launcher's model worker: JSON lines on stdin / stdout, stored conversation states (cloned on use) with prefix matching, system-prompt snapshots |
| `tests/` | Parity, precision, correctness and benchmark scripts (see below) |
| `models/Inkling-2-Mini/` | Checkpoint (not part of the code) |

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python mlx==0.32.4 mlx-vlm==0.8.0
uv pip install -p .venv/bin/python torch      # only for tests/make_tiny_reference.py
```

Weights: all files of `thinkingmachines/Inkling-2-Mini`, about 20.2 GB. `mtp.safetensors`
(636 MB) is the DSpark draft. huggingface.co was unreachable from this machine, so the files
came from the ModelScope mirror
`https://www.modelscope.cn/models/thinkingmachines/Inkling-2-Mini/resolve/master/<file>`.
Its SHA-256 values match the Hugging Face LFS hashes. aria2c with 16 connections reached
about 11 MB/s.

## Performance

All measurements are on a Mac mini with an M6 (12-core GPU), 32 GB and MLX 0.32.4.

### One request

Decode throughput in tok/s over 300-token generations:

| Prompt | BF16 dense: plain | BF16 dense: DSpark | 8-bit dense: plain | 8-bit dense: DSpark |
| --- | --- | --- | --- | --- |
| code, greedy | 16.7 – 19.8 | 21.9 | 25.1 | 26.5 – 34.3 |
| code, T = 1 | 19.8 | 20.9 | 26.0 | 27.9 – 35.2 |
| list, greedy | 16.1 – 19.5 | 19.6 | 26.2 | 27.1 – 27.4 |
| story | 19.6 | 19.5 | 25.2 – 25.9 | 24.0 – 24.5 |
| Chinese + thinking | 19.1 | 19.9 | 25.7 – 26.3 | 25.6 – 26.2 |

- This table predates the second round of decode glue fusion ("What was done", item 8) and
  the 6-bit default (item 9). Plain decoding is now about 31 tok/s at 8-bit dense and about
  34 tok/s at the 6-bit default on these prompts.
- Ranges come from repeated runs. Greedy text, and so DSpark acceptance, changes with tiny
  numeric differences between runs.
- Since the decode fusions, plain decoding is fast enough that DSpark only pays on
  predictable text (code, lists). With the earlier geometric policy, prose ran 5–9 % *slower*
  with DSpark on. The confidence-gated policy (item 10) brings prose back to plain speed.
- A fixed block of 8 reaches 34 tok/s on code but drops to about 12 tok/s on prose, so the
  adaptive policy is the default.

Long contexts (8-bit dense, plain decoding):

| Context (tokens) | 64 | 2,048 | 8,192 | 16,384 |
| --- | --- | --- | --- | --- |
| ms per decode step, before the sink-slot caches | 41.6 | 49.4 | 53.7 | 61.8 |
| ms per decode step, now | 43.0 | 44.5 | 45.0 | 52.3 |
| Prefill (tok/s) | — | 614 | 562 | 511 |

### Several requests (batching)

Aggregate decode throughput, 256-token essays, greedy:

| Batch | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| tok/s, all requests together | 22.9 | 33.6 | 46.5 | 61.8 | **75.6** |
| tok/s per request | 22.9 | 16.8 | 11.6 | 7.7 | 4.7 |

Before the grouped expert kernel, batch 16 reached 66.0 tok/s. Peak memory stays about 20 to
22 GB.

### KV cache and long contexts

Cache memory per token is fixed by the architecture:

- **Sliding-window visits (40 of 48).** They are capped at 1,024 tokens: about 160 MB in
  total, whatever the context length.
- **Full-attention visits (8 of 48).** Layers 5 and 23 run once and layers 11 and 17 run on
  all 3 passes. Each token adds 32 KB of K/V to these visits, and every decode step reads all
  of it.

| Context (tokens) | 16,384 | 32,768 | 65,536 |
| --- | --- | --- | --- |
| Full-attention KV (BF16) | 0.54 GB | 1.07 GB | 2.1 GB |
| Prefill with the materialized mask, 512-token chunks (shrinking), tok/s | 511 | 417 | 299 |
| Prefill with band attention, 1024-token chunks, tok/s | 559 | 530 | 488 |
| Prefill with band attention + fused prefill glue, tok/s | — | 782 | 688 |
| + attention prep kernel, 8-bit KV, tok/s | — | **851** | **798** |
| Peak memory with band attention | 23.2 GB | 23.8 GB | 25.0 GB |
| Decode (ms/token) | 52.7 | 57.3 | 70.9 |

1. **Multi-turn KV reuse (`Session`).** A turn prefills only the new message. The model's own
   replies, thinking included, stay in the caches exactly as generated. In a 6k-token
   document Q&A, the first question prefilled 5,790 tokens (11.7 s) and the follow-up only 23
   (0.24 s). Turn-2 logits match a one-shot prefill of the same token stream (KL 3e-9).
2. **Save / load.** A session file stores every cache, the draft state and the messages: 423 MB
   for 6.5k tokens. The ring buffers are trimmed before they wrap. A reloaded session continues
   token-for-token identically.
3. **Band attention for long prefill (`band_attention.py`).** Full-attention logits are
   q·k·scale + bias(distance) + causal. The bias is zero beyond 1,024 tokens back, and there is
   one per-query zero-valued sink.
   - **The old path** materialized the heads × chunk × context mask: 3.2 GB per full visit for
     1,024 queries over 64k keys. Reading it back took about 70 % of SDPA time, and the
     chunks had to shrink to bound it.
   - **The kernel** is MLX's NAX flash attention with three changes:
     - The mask read becomes a lookup in a heads × chunk × 1024 bias table
       (`relative_states @ proj`, about 50 MB), and only on key blocks that intersect the
       band.
     - The sink is injected as each row's initial softmax state, so no zero K/V is
       concatenated.
     - The sizes are runtime values, so one pipeline serves every chunk and context, and it
       reads K/V straight from strided cache views.
   - **Accuracy.** It matches the mask path to BF16 rounding (≈1e-3). With real weights after
     a 6k prefill: mean KL 4e-3, top-1 100 %.
   - **Speed.** At 1,024 × 64k it takes 77 ms per visit instead of 233 ms, close to plain
     causal SDPA (67 ms). Without the mask, chunks no longer shrink with context; without
     NAX, prefill falls back to the old path with mask-budgeted chunks.
4. **8-bit KV with a fused decode kernel (`--kv-bits 8`, the default).**
   - **Storage.** The full-attention K/V are stored affine 8-bit along head_dim (group 64).
   - **Decode and verification.** `decode_attention.q8_decode_attention` reads them directly.
     It is MLX's split-K GQA `sdpa_vector` design, so each key row is read once for the 3
     query heads that share it. Values are dequantized in registers, the band bias is added
     for distances < 1024, and the per-query sink is folded into the second pass. Partials
     stay in FP32. It handles up to 8 causal queries, so speculative verification never
     dequantizes the cache.
   - **Prefill.** Prefill dequantizes into `band_attention`, at no measurable cost.
   - **Kernel accuracy.** Against an exact FP32 reference over the same quantized cache:
     ≤ 2.6e-3, i.e. BF16 output rounding.
   - **Kernel speed.** At 64k: 1.3 ms instead of 3.0 ms per visit (BF16 mask + SDPA). At
     128k: 2.3 vs 6.6 ms, at about 126 GB/s.

   | Context | BF16 KV decode | 8-bit KV decode | Full-attention KV memory |
   | --- | --- | --- | --- |
   | 32k | 50.3 ms/token | 44.4 ms/token | 1.07 → 0.57 GB |
   | 64k | 61.4 ms/token | **48.1 ms/token** | 2.1 → 1.1 GB |

   Prefill speed is unchanged (64k: 773 vs 777 tok/s), and peak memory drops by 1 GB.

   **Quality** (`tests/kv8_quality.py`, 32k context, 200 teacher-forced positions):

   | Comparison | Mean KL | Median KL | p99 KL | Top-1 |
   | --- | --- | --- | --- | --- |
   | 8-bit vs BF16 KV | 4.3e-3 | 3.7e-5 | 0.14 | 98.5 % |
   | Noise floor: the same BF16 math computed as one block instead of step by step | 1.3e-3 | 2.1e-5 | 2.6e-2 | 99.0 % |

   That is the same order as 8-bit dense layers (mean 2.9e-3).

   **Limits.** With about 7 GB free for KV after the model, the context limit is about 440k
   tokens at 8 bits (about 220k in BF16). 4 bits would reach about 880k, but it is lossy
   (mean KL 1.5e-2 at 32k) and still uses the slower two-matmul path.

### Prefill glue fusion

Profiling one decoder visit over a 1024-token chunk (`tests/profile_prefill_layer.py`) put about
half of the time outside the GEMMs. These were fused, each kept arithmetically identical to the
MLX op chain it replaces:

| What | Before (per visit) | Now | Done with |
| --- | --- | --- | --- |
| Short convolutions: attention and MLP outputs, plus sliding K/V | 3.8 ms (fp32 casts, state concatenation, generic conv1d, residual add) | one pass each, with the following residual add fused | `prefill_kernels.short_conv`, bit-identical |
| Full-layer dynamic K/V convolutions | 3.7 ms | one pass each | `prefill_kernels.dynamic_conv`, bit-identical |
| Sliding-window attention | 8.0 ms (1024 × 2047 materialized mask + SDPA) | 2.9 ms | `band_attention` with `sliding=1024`: blocks outside the window are skipped |
| MoE un-sort + weighted top-8 sum | 2.4–2.7 ms (37 MB gather, FP32 cast, reduction) | one pass reading through the inverse permutation | `prefill_kernels.combine_routes` |
| Message gates on passes 2–3 (q / k / v and MLP) | about 8.5 ms (two RMSNorms + concatenation + cast, tanh / combine, RMSNorm) | about 4 ms: two MLX GEMMs between two fused kernels | `prefill_kernels.message_gates`, 99.9 % bit-identical to the old GPU path |
| Per-layer message attention residual | 1.5 ms | 0.5–0.7 ms | the fused attention-residual kernel, reading the block states in place |

Prefill throughput went from 519 to **880 tok/s** at 8k, 530 to **782** at 32k and 488 to
**688** at 64k. The attention prep kernel from the decode work (`decode_kernels.attn_prep`,
used for B = 1 chunks too) then took the first row of the remaining-glue table below and
raised prefill to **967** at 8k, **851** at 32k and **798** at 64k (8-bit KV). A 64k
compaction input now prefills in 82 s, at a 21.3 GB peak. A 64k-token compaction input now prefills in about 95 s (it took 219 s before
the band kernel). Peak memory also dropped by 1–2 GB, from fewer FP32 intermediates.

About two thirds of a chunk is now GEMMs running at about 12 TFLOPS, and about 15 % is
attention. The remaining glue is roughly:

| Remaining glue | Per chunk |
| --- | --- |
| L2 clips / value norm / transposes | about 60 ms (now one kernel: `decode_kernels.attn_prep`) |
| Route sort + row gather | about 35 ms |
| SwiGLU | about 30 ms |
| Band tables | about 30 ms |
| Router | about 20 ms |

### Why these are the numbers

- **One request is limited by memory bandwidth.** Each token reads about 4.2 GB with 8-bit
  dense layers, or about 3.8 GB with the 6-bit default (`tests/byte_floor.py`, from the
  loaded parameters):
  - dense layer weights: 1.66 GB at 8 bits, 1.34 GB at 6 bits. The 12 loop layers are read
    3 times, so 48 visits;
  - routed experts: 1.84 GB, top-8 of 128 per visit;
  - lm_head: 0.49 GB at 8 bits, 0.38 GB at 6 bits;
  - KV: about 0.25 GB at 4k context (168 MB of sliding rings plus the 8-bit full-attention
    cache).
- **This M6's ceiling.** An 8-bit quantized GEMV over 0.32 GB and a plain reduction both
  measure about 140 GB/s. With 8-bit dense layers, that puts the single-request floor at
  about 30 ms per token (33 tok/s), and plain decoding runs at 33.8 ms, about 90 % of it.
  With the 6-bit default the floor is about 27 ms, and plain decoding runs at about 29.5 ms
  (34 tok/s).
  - Most of the remaining 3 to 4 ms is the serial chain of about 20 dependent kernels per
    visit, 48 visits per token. Each dependent dispatch costs about 3.5 µs on this GPU.
  - Thinking Machines' chart is a theoretical ceiling at 614 GB/s (M5 Max class).
- **Batching reads each weight once per step for all rows.** The cost grows with the number of
  *distinct* experts the rows touch: about 39 of 128 per visit for 16 rows. MLX's
  `gather_qmm` only deduplicates expert reads at ≥ 4 rows per expert, which needs at least 512
  routes. Below that, every route re-reads its expert, so `kernels.grouped_nvfp4_qmv` takes
  over between 32 and 511 routes.

### What was done, and what it bought

1. **Native expert precision.** NVFP4 experts use `gather_qmm(mode="nvfp4", global_scale=2688·amax/denominator)`,
   and their values are bit-identical to the reference dequantizer. Gate and up stay one
   [gate; up] matrix (one w13 tensor, one global scale per expert), which saves one kernel
   per visit.
2. **8-bit dense layers (the default until item 9).** About 1.7 GB less per token. Against BF16, over 311
   teacher-forced positions: median KL 2.2e-4, mean 2.9e-3, top-1 98.1 %. That is below the
   BF16-vs-FP32 noise of the model itself (median 1e-3, mean 2.5e-2). For comparison, 6 bits
   gives median 2.5e-4, and 4 bits gives median 3.0e-3, which is lossy.
3. **Decode fusions.** One step dropped from about 11,900 to about 8,300 graph nodes:
   - **Attention residual kernel.** The block attention residual and its RMSNorm run as one
     Metal kernel that reads the block states in place (97 calls per token).
   - **One projection matrix.** The q / gate / r / sink / k / v projections share one weight
     matrix: one GEMV when the inputs are equal, three otherwise.
   - **Batched message gates.** The q / k / v message gates run as one batched, compiled
     computation.

   Single-request decode went from about 24 to about 26 tok/s. The rest of a step is
   bandwidth-bound kernels running at about 90 to 105 GB/s.
4. **Sink-slot caches.** The per-query attention sink is a zero key / value. It used to be
   concatenated onto the fetched K/V at every visit, and sliding layers also reordered their
   ring into temporal order first: one or two full K/V copies per visit, 48 times per token.
   - `SinkKVCache` keeps a zero slot after the last entry.
   - `SinkRingKVCache` keeps a permanent zero slot in a W + 1 ring. Single-token decode
     attends to the ring in its physical layout and only rotates the small mask.

   The arithmetic is unchanged, and the gain grows with context (table above).
5. **DSpark speculative decoding.** The 3-layer draft reads the hidden states entering
   *visits* 5, 23 and 41 (pre-loop layer 5, pass-2 layer 11, pass-3 layer 17) plus the final
   normalized state, and proposes 7 tokens per round.
   - A draft token is accepted while it equals the target's own sample at that position, so
     the output distribution is unchanged at any temperature.
   - The draft-side lm_head and the folded Markov projection are 4-bit copies, which halves the
     draft cost. Verification uses the target's own head.
   - `BlockPolicy` estimates acceptance and round costs online and picks a block of 1 (plain)
     to 8. Item 10 replaced it as the default. It remains for `--block-size` and as
     `generate.SPEC_POLICY = "geometric"`.
6. **Grouped multi-token MoE.**
   - Verification and batches sort their routes by expert.
   - From 32 routes, `grouped_nvfp4_qmv` reads each distinct expert once per 4 routes. It is
     bit-identical to `gather_qmm`, 17 to 33 % faster per expert matmul, and handles the
     1152-wide down projection with a tail block.
   - An 8-token verify went from 169 ms (mlx-vlm's per-route kernels) to 128 ms (sorted
     `gather_qmm`) to about 105 ms.
7. **Batching.** Each request is prefilled on its own, with the same numerics as a single
   request. The 48 caches are then merged into left-padded batch caches, and all rows decode
   in lockstep. Finished rows are filtered out one step late, which keeps CPU graph building
   overlapped with the GPU.
8. **Decode glue fusion, round 2** (`decode_kernels.py`, `decode_attention.ring_decode_attention`).
   One plain decode step went from 36.3 to 33.8 ms (A/B, 4k context, minimum of 5
   interleaved runs). Contributions:

   | Fusion | What it replaces | Step time |
   | --- | --- | --- |
   | Short-conv kernels for single tokens, emitting the next conv state | 4 convs per visit, each a concatenate / conv / add / scale / state-slice chain | −1.8 ms |
   | Attention prep kernel | q / k / r L2 clips with learned scales, value RMSNorm, log position scaling, transposes | −0.8 ms, together with the next row |
   | Ring decode attention | Sliding layers: mask build, roll, sink concatenation and SDPA over the W + 1 ring | (included above) |
   | Single-token NVFP4 experts | `gather_qmm` gate_up → SwiGLU → `gather_qmm` down → combine, as 2 kernels | −0.2 ms (−8 % per visit in isolation; bit-identical) |
   | Router top-k kernel | mlx-vlm's compiled `_route` | ±0 |

   End to end (`tests/bench_decode_fusion.py`, 300-token greedy generations, best of 2
   alternating runs):

   | Prompt | Plain: off → on (tok/s) | DSpark: off → on (tok/s) |
   | --- | --- | --- |
   | story | 29.4 → 31.0 (+5.5 %) | 27.9 → 29.3 (+5.0 %) |
   | code | 29.8 → 31.1 (+4.4 %) | 29.2 → 38.9 |
   | Chinese prose | 29.5 → 30.7 (+4.2 %) | 27.6 → 28.7 (+3.9 %) |

   The DSpark code jump mostly reflects acceptance. The two runs produce slightly
   different greedy text, and code acceptance swings widely with it.

   All five are bit-identical or within BF16 rounding of the ops they replace
   (`tests/test_decode_kernels.py`). Against CPU-FP32 ground truth they are no worse than
   before (`tests/kernel_precision.py`, `tests/decode_fusion_quality.py`). The remaining gap
   to the bandwidth floor is the serial kernel chain, about 0.17 ms per kernel level
   removed from all 48 visits.
9. **6-bit dense layers (the default since 2026-11-23).** They read 0.44 GB less per token
   than 8 bits (3.8 GB instead of 4.2) and leave 0.3 GB more memory free (resident
   19.16 GB).

   Speed (`tests/bench_dense_bits.py`, 300-token greedy generations, best of 2 alternating
   loads):

   | Prompt | Plain: 8 → 6 bits (tok/s) | DSpark: 8 → 6 bits (tok/s) |
   | --- | --- | --- |
   | story | 31.5 → 33.9 (+7.6 %) | 29.4 → 31.4 (+6.6 %) |
   | code | 31.4 → 33.6 (+7.0 %) | 38.0 → 39.1 (+2.8 %) |
   | Chinese prose | 31.2 → 33.6 (+7.7 %) | 29.4 → 30.7 (+4.2 %) |

   - Prefill is unchanged: 990 → 984 tok/s at 8k, 902 → 879 at 32k (within run-to-run noise).
   - Decode at 32k context: 41.9 → 39.1 ms per token.

   Precision is not free. `tests/dense_precision.py` uses a fixed English + Chinese text and
   compares 437 positions against CPU-FP32 truth (406 prefill, 31 single-token decode,
   8-bit KV):

   | Dense | Mean KL | Median KL | p99 KL | Top-1 vs truth |
   | --- | --- | --- | --- | --- |
   | BF16 (the model's own noise) | 4.3e-3 | 1.45e-3 | 3.0e-2 | 95.7 % |
   | 8-bit | 3.4e-3 | 1.47e-3 | 2.8e-2 | 96.8 % |
   | 6-bit (default) | 8.2e-3 | 3.61e-3 | 4.9e-2 | 95.2 % |

   - **6 bits is lossier than BF16, which 8 bits is not.** The median KL is 2.5× BF16's. The
     added error is about 1.5× the model's own BF16-vs-FP32 noise, and top-1 agreement is
     unchanged. Against BF16, 6 bits gives mean KL 5.9e-3. That is the same order as the
     8-bit KV default (4.3e-3).
   - **Item 2's "6 bits ≈ 8 bits" (median 2.5e-4) understated the cost.** That test
     teacher-forces the model's own greedy text, where almost every token is near-certain.
   - **The error comes from the attention projections, not lm_head.** Mixed variants (median
     KL vs truth, plain tok/s, in one run):

     | Attention projections | lm_head | Median KL | Plain tok/s |
     | --- | --- | --- | --- |
     | 8-bit | 8-bit | 1.47e-3 | 31.7 |
     | 8-bit | 6-bit | 1.65e-3 | 32.0 |
     | loop layers 8-bit, others 6-bit | 6-bit | 2.07e-3 | 32.6 |
     | o_proj 8-bit, the rest 6-bit | 6-bit | 2.53e-3 | 33.6 |
     | loop layers 6-bit, others 8-bit | 6-bit | 2.91e-3 | 33.8 |
     | 6-bit, group 32 | 6-bit | 3.26e-3 | 32.8 |
     | 6-bit | 8-bit | 3.56e-3 | 33.3 |
     | 6-bit | 6-bit | 3.61e-3 | 34.3 |

     The error is spread across loop and non-loop layers, and across the fused q / k / v
     and o_proj. Per byte saved, o_proj is the most sensitive. Keeping o_proj at 8 bits
     costs 2 % of the speed and halves the added error. That variant is not wired to a
     flag.
10. **Confidence-gated DSpark (the default since 2026-11-23).** The DSpark paper
    (arXiv 2607.05147) pairs its drafter with a hardware-aware prefix scheduler. A trained
    confidence head predicts whether each drafted token survives, and each round verifies the
    prefix that maximizes expected tokens per second on a measured step-cost table. The
    bundled draft has no confidence head. Its own probability for each drafted token still
    separates accepted from rejected positions: AUC 0.79–0.87 on prose and 0.92 on code, over
    2,189 traced rounds at block 8 (3,519 positions, English / Chinese / code, T = 0 and 1).

    | Draft probability | 0.4–0.5 | 0.7–0.8 | 0.9–0.95 | 0.98–0.995 | ≥ 0.995 |
    | --- | --- | --- | --- | --- | --- |
    | Accepted (given the previous positions were) | 19 % | 39 % | 59 % | 83 % | 94 % |

    `ConfidencePolicy` (in `speculative.py`) works in four steps:
    - It drafts the full block every round. The draft costs nearly the same however many
      positions are read out.
    - It maps each draft probability to an acceptance probability through a binned table. The
      table's prior comes from these traces, and it is updated online, because code accepts
      more at the same confidence.
    - It verifies the prefix L that maximizes E[tokens | L] − cost(L) / plain-step cost, so a
      round runs only if it beats plain decoding over the same time.
    - If no L ≥ 2 pays, it falls back to a pipelined plain segment of 24 tokens, or 48 after a
      second fallback in a row. Stats report the count as `plain_fallbacks`.

    The cost structure decides where it can win. At 6-bit dense and short context:
    - A plain step costs 33 ms.
    - The draft costs 10.4 ms per round (8.9 ms with 8-bit draft layers, below).
    - Verifying L tokens costs 38 / 48 / 58 / 71 / 82 / 86 / 93 / 101 ms for L = 1 to 8, plus
      about 7 ms of round overhead.

    **Prose cannot win.** Replayed on the traces, even an oracle that verifies exactly the
    accepted prefix stays below plain decoding (23–27 vs 30.5 tok/s). A round costs at least
    draft + V(2) + overhead, about 65 ms, for at most two tokens. Prose accepts only 1.4 to 1.7
    tokens per round at block 8. So the best a policy can do on prose is to stop speculating
    cheaply.

    The geometric policy kept probing and lost 5–9 % on prose. Split for zh, T = 0: plain 31.9,
    geometric DSpark 29.1, DSpark held in plain segments 31.2 tok/s. About 2 points come from
    capturing draft features and 6.6 from losing verify rounds.

    End to end (`tests/bench_spec_policy.py`, 300 tokens, best of 2, 8-bit draft layers):

    | Prompt | Plain | Geometric | Confidence | vs geometric | vs plain |
    | --- | --- | --- | --- | --- | --- |
    | story, T = 0 | 31.9 | 30.3 | 32.7 | +7.9 % | +2.6 % |
    | story, T = 1 | 32.1 | 30.2 | 32.8 | +8.4 % | +2.2 % |
    | zh, T = 0 | 31.9 | 29.8 | 32.6 | +9.6 % | +2.2 % |
    | zh, T = 1 | 32.2 | 30.2 | 31.5 | +4.4 % | −2.1 % |
    | zh_long, T = 0 | 31.5 | 29.8 | 31.3 | +5.0 % | −0.5 % |
    | zh_long, T = 1 | 31.3 | 30.5 | 31.7 | +4.0 % | +1.3 % |
    | 6 code prompts (Python, JS/TS, SQL, Rust), T = 0, geometric mean | 31.5 | 33.7 | 36.7 | +8.9 % | +16.7 % |

    - Code varies widely per prompt and run, because each policy produces slightly different
      greedy text. Confidence won 5 of the 6 code prompts and lost one by 6 %. Across two
      separate benchmark runs, the Fibonacci prompt read +0.4 % and then −13.7 %.
    - Prose now runs at plain speed (−2 % to +3 %, within noise), and code keeps its gain.
    - `tests/check_gated_spec.py` checks the draft-feature bookkeeping: a fallback after
      drafting must not feed the draft the same features twice. It also checks that greedy
      output still matches plain decoding up to near-ties.

    **8-bit draft layers.** The draft's three BF16 layers are now affine 8-bit
    (`load_drafter(layer_bits=8)`). The draft only proposes tokens, so the output is unchanged
    (greedy runs, fixed block 8):

    | Draft layers | Draft per round | Tokens per round: story / code / code 2 / zh |
    | --- | --- | --- | --- |
    | BF16 | 10.4 ms | 1.57 / 4.11 / 2.70 / 1.38 |
    | 8-bit (default) | 8.9 ms | 1.57 / 4.23 / 2.65 / 1.38 |
    | 6-bit | 8.9 ms | 1.60 / 4.17 / 2.73 / 1.37 |
    | 4-bit | 8.4 ms | 1.54 / 4.05 / 2.59 / 1.35 |

    The rest of the draft time is the 4-bit vocabulary head and seven sequential Markov steps.
    The 8-bit layers also free 0.25 GB.

    **Why DSpark's headline numbers don't transfer here.**
    - The paper's "+60–85 %" is per-user speed at matched throughput in GPU serving, against
      MTP-1, at T = 1, with a confidence head trained for the target model. Its offline
      acceptance lengths are 5–6 tokens on math and code and 3.3–3.6 on chat.
    - mlx-dspark on Apple silicon reports 2–4× on dense targets, where verifying several tokens
      costs about one step. On an MoE target (Qwen3.6-35B-A3B) it reports 1.32×.
    - Here each extra verified token costs about 10 ms, 30 % of a plain step. Every token routes
      to its own 8 of 128 experts in each of the 48 visits, so verification, like batching, is
      bounded by the union of experts it touches. The bundled draft also accepts much less on
      prose.

### Hardware utilization and the loop

| Regime | Utilization | Bottleneck |
| --- | --- | --- |
| Decode, one request | About 90 % of bandwidth (6-bit dense: 3.8 GB per token in about 29.5 ms at 4k context, vs about 27 ms at 140 GB/s), about 2 % of compute | Reading weights, plus about 3 to 4 ms per token of serial small-kernel latency |
| Prefill | About 5.6 TFLOPS effective. The NVFP4 expert and 8-bit dense GEMMs themselves run at 12–15 TFLOPS, so about half the time is spent outside them | Non-GEMM work: route sort / gather / scatter, sliding-layer masks, short convolutions, small ops |
| Batch of 16 | 76 tok/s in total | The union of distinct experts per visit |

What the loop permits, measured:

- **The passes of one token are strictly sequential.** Pass p + 1 of every loop layer reads
  messages from pass p's final state. Shared weights can only be amortized over several tokens
  in flight, which is what prefill, verification and batching already do.
- **Running fewer passes doesn't work** (`tests/pass_sensitivity.py`, 836 teacher-forced
  positions of the model's own text):

  | Passes | Top-1 agreement with 3 passes | Mean KL |
  | --- | --- | --- |
  | 2 | 69.7 % | 0.65 |
  | 1 | 56.2 % | 1.23 |

  This rules out early exit and pass-level self-speculation. A 2-pass draft would share 30 of
  48 visits with the target but still costs about 31 ms per draft token serially, at about
  70 % acceptance.
- **A cross-chunk wavefront in prefill doesn't help.** Running chunk c + 1 pass 1, chunk c
  pass 2 and chunk c − 1 pass 3 as one batch per loop layer is a valid schedule, but its only
  effect is more rows per weight read, like larger chunks. Prefill throughput is flat from 512
  to 2048-token chunks (509 → 515 tok/s at 8k), and 4096 runs out of memory.
- **Spending bytes to rescue 6-bit dense precision buys little.** Group size 32 at 6 bits
  (7 bits per weight) or an 8-bit lm_head barely move the error (item 9). The error is in the
  attention projections. Only putting them, or part of them, back to 8 bits helps.
- **Prompt-lookup (n-gram) drafting doesn't beat DSpark on compaction** (`tests/prompt_lookup_sim.py`).
  On a real 10k-token compaction, DSpark already reaches 4.40 tokens per round (34 tok/s), since
  its attention to the context captures the copied spans. Context n-gram drafting would give
  about 2 tokens per round (k = 4 to 16), even though about half of the summary tokens do
  come from copied spans.
- **Memory.**
  - Resident after load: 18.91 GB with the 6-bit default and 8-bit draft layers (19.16 GB with
    BF16 draft layers, 19.45 GB at 8-bit dense). Breakdown:

    | Part | GB |
    | --- | --- |
    | Experts | 15.67 |
    | Attention and other layer weights (6-bit) | 0.60 (0.79 at 8 bits) |
    | Embedding | 0.93 |
    | lm_head (6-bit) | 0.38 (0.49 at 8 bits) |
    | Vision tower | 0.57 |
    | DSpark draft: 8-bit layers, 4-bit heads | 0.68 (0.93 with BF16 layers) |

  - A short chat peaks at about 19.7 GB.
  - KV grows by about 17 KB per token at 8 bits (32 KB in BF16), plus fixed 168 MB of
    sliding rings.

### What did not help, or what to know before trying

- **Splitting long-prefill attention into far / near parts with MLX ops.** `mx.fast`'s SDPA
  does not return the log-sum-exp needed to merge two partial softmaxes, and a blocked
  online softmax built from matmuls materializes FP32 scores, which is as much traffic as the
  mask. Hence the custom kernel. `mx.fast.metal_kernel` has no include path for MLX's
  headers, so `band_attention.py` inlines `steel/attn/nax.h` from the installed wheel.
- **Folding the sink into SDPA's `sinks=` argument.** It only takes one sink per head. The
  equivalent per-query shift (`mask - sink`, zero head sinks) is exact in real arithmetic but
  adds a BF16 rounding of the shifted mask, about 2.5 ulp. The sink-slot caches do the same
  job exactly.
- **Removing graph nodes buys little by itself.** About 3,600 fewer nodes saved about 2 ms,
  because most of them were views or tiny kernels.
- **Don't misread ablations:**
  - Making passes 2 and 3 ignore their messages "saves" 18 ms, but only because the passes
    stop depending on each other and the GPU overlaps them.
  - Batch benchmarks fed the same token to every row look fast at first, until the rows
    diverge and touch more distinct experts.
- **Grouped-kernel variants.**
  - A run-wide version (up to 16 routes accumulating against one weight load) performs about
    the same as V = 4.
  - V = 8 is slower because of register pressure.
- **DSpark policy variants.** A per-depth conditional-acceptance policy and larger probe
  blocks did not beat the simple geometric policy. Both pick one length for every round. What
  did beat it is a per-round length from the draft's confidence (item 10). Within that policy:
  - Plain-segment caps of 96 or 256 tokens gain nothing on prose and miss code that follows a
    prose stretch (simulated mixed text: 34.7 tok/s at a 48 cap vs 33.2 at 256).
  - The calibration table's prior strength hardly matters.
  - Thresholding the draft's entropy or top-2 margin instead of its token probability
    separates accepted from rejected positions slightly worse.
- **Strata-style screened drafting buys about 1 %, so it is off** (`DSparkDrafter.SCREEN = 0`).
  Strata's Inkling engine computes the draft's 7 Markov steps over a 512-token shortlist
  instead of the full 200k vocabulary, and reports 16.5 → 2.5 ms per draft. Measured here
  (`tests/check_screened_draft.py`):
  - Our full Markov loop already takes only 2.6 ms of the 10.4 ms draft, because each step
    is a 4-bit matrix-vector product. The draft layers and vocabulary head dominate.
  - This draft's Markov bias is not a small correction. Its static per-token part has std
    2.0 and its state-dependent part 1.2–1.6, against 3.0 for the base logits.
    - Screening on the base logits misses the full loop's token at 24 % of positions.
    - Screening on base + static bias gets the full loop's token to a p99 rank of about 430.
  - Slice maxima are a poor screen, because top tokens sit at neighbouring ids (Chinese
    especially). Coverage of the full loop's token:

    | Screen | Coverage | Markov loop |
    | --- | --- | --- |
    | 512 slice maxima | 76 % | 0.9 ms |
    | 512 slices × top-8 | 97.4 % | 1.4 ms |
    | 256 slices × top-16 (4,096 candidates) | 98.8 % agreement | 1.6 ms |
    | 128 slices × top-32 | 99.4 % agreement | 2.6 ms (slow kernel) |

  - The best setting saves 1.0 ms per round, about 1 % of a code round. The 1.2 % of changed
    deep drafts give that back in acceptance. An exact top-1024 (99.8 % coverage) through
    `mx.argpartition` costs 1.7 ms by itself.
- **Strata's tree verification wouldn't rescue prose either.** Strata verifies several
  candidates per position and picks the length by expected gain on a calibrated cost model.
  In its own model (M5 Max), each extra verified token also costs about 30 % of a step, as
  here. A tree would need tree masks in the fused verify kernels and branch-aware rollback of
  the conv and ring states.
- **mistral.rs (checked 2026-11-23) has nothing to add for item 9.** Its `topology` files are
  manual per-layer / per-regex precision, like the mixed variants above. Its imatrix
  calibration only applies to GGUF K-quants, not to the MLX-style AFQ affine formats. Its
  automatic precision boost covers embeddings and the output head, not attention
  projections. Its docs publish no KL or perplexity numbers.
- **Decode glue that MLX already handled.**
  - The router (sigmoid, bias, top-k, normalization) as one kernel is faster in isolation
    but gains nothing end to end. mlx-vlm's `_route` is `mx.compile`d, so it was already
    only a few kernels.
  - A branch-free E2M1 decode in the expert kernels (bit tricks instead of the
    `fp4_e2m1` struct) changes nothing, because those kernels are bandwidth-bound at
    about 131 of 140 GB/s.
- **MLX command-buffer sizes.** `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`
  (200 to 2,000 ops, up to 5 GB) leave decode within ±0.2 ms. `async_eval` blocking for
  about 28 ms of a 34 ms step is not CPU overhead: the encoder waits for free command
  buffers while the GPU works.

## Verification

| Test | What it checks | Result |
| --- | --- | --- |
| `tests/make_tiny_reference.py` + `tests/parity_mlx.py` | A tiny random `Inkling2Loop` built with the **transformers reference**. It has 8 layers, a 2–6 × 3 loop, full and sliding layers inside and outside the loop, and an 8-token window. It is saved in the original checkpoint layout and loaded through this loader. Compared on the CPU in FP32: one-shot forward, chunked prefill + decode (rings wrap), vision tower, and image + text forward | ≤ 6e-5 relative error, argmax 100 % |
| `tests/check_batch.py` | Tiny model, FP32, CPU. Prompts of 5, 13 and 9 tokens, caches merged (including a wrapped ring), batched teacher-forced decode with one row filtered out half way, vs each row alone | ≤ 1e-5 |
| `tests/check_rollback.py` | Tiny model, FP32, CPU. 18 rounds of verify-L / commit-k with interleaved single-token steps, vs plain decoding. Covers ring replay, sink-slot trim and conv-state rollback | 3e-5 |
| `tests/test_kernels.py` | Fused attention-residual kernel (GPU) vs an exact FP32 CPU reference | ≤ 1.2e-6 |
| `tests/test_grouped.py` | Grouped NVFP4 kernel vs `gather_qmm` on real experts, plus timing | bit-identical |
| `tests/check_experts.py` | Native NVFP4 / MXFP8 experts vs a numpy decode of the raw checkpoint bytes, following transformers' `Inkling2LoopDequantizeExperts` | bit-identical |
| `tests/check_spec.py` | Real weights, informational. Oracle and garbage drafts vs plain greedy decoding, with the logit margin printed at the first divergence. One-token decode and 8-token verification use different kernels (GEMV vs GEMM, two SDPA variants), so they diverge at near-ties (margins around 0.2–0.4) after about 50 tokens. Exact rollback correctness is what `tests/check_rollback.py` checks | near-tie divergence only |
| `tests/check_ring.py` | Real weights, 1,500-token context. Ring decode (physical layout, now the fused ring kernel) vs the chronological path | top-1 96.7 % (1 of 30), mean KL 2.3e-3 (BF16 noise; 1.2e-3 before the ring kernel) |
| `tests/test_decode_kernels.py` | Decode glue kernels vs the op chains they replace, on real weights: router (1 and 8 tokens), attention prep (1 to 300 tokens, sliding and full layers), ring attention (5 to 3,000 tokens through the ring), single-token experts, plus timing | router: same experts, weights ≤ 2.4e-7; prep and experts bit-identical; ring ≤ 8.3e-4 |
| `tests/kernel_precision.py` | Real weights, BF16 dense, fixed text: GPU variants (decode fusions on, off, each off alone) vs CPU-FP32 truth over 157 prefill positions + 23 single-token decode positions | prefill: mean KL 3.75e-3 on vs 4.44e-3 off; decode: 2.9e-3 vs 2.2e-3 (noise level, top-1 100 %) |
| `tests/decode_fusion_quality.py` | Real weights: 200 teacher-forced decode steps after 3,000 tokens, fusions on and off, each vs the multi-token (prefill) path's logits | fused: mean KL 3.1e-3, top-1 95.5 %; unfused: 4.6e-3, 94.5 % (no added error) |
| `tests/test_band_attention.py` | Band-attention kernel vs banded mask + zero sink + fused SDPA, on aligned / ragged sizes up to 1,024 × 64k, plus timing | ≤ 2e-3 (BF16 rounding), 2.7 to 3.0× faster |
| `tests/check_band_model.py` | Real weights: a 6k-token prefill through the band kernel vs the mask path | mean KL 4e-3, top-1 100 % |
| `tests/bench_prefill.py` | Long-context prefill throughput, peak memory and decode | see "KV cache and long contexts" |
| `tests/test_prefill_kernels.py` | Short-convolution, dynamic-convolution and combine kernels vs the mlx-vlm modules (CPU) on real weights, 37 and 1024 tokens | bit-identical (combine: ≤ 1 BF16 ulp, from summation order) |
| `tests/test_message_gates.py` | Fused message gates vs the reference `MessageGate` math (CPU), and vs the old GPU path with the same GEMMs | bit-identical at 1 token; 99.9 % bit-identical vs the old GPU path |
| `tests/check_session.py` | Tiny model, FP32, CPU: a continued `GenState` equals a fresh prefill of the concatenated stream. Real weights: turn-2 KL vs a one-shot prefill (the first reply token is near-certain, so this catches only gross errors), prefill cost, and a save / load round trip, with 8-bit KV | exact / ≈0 / identical |
| `tests/test_decode_attention.py` | 8-bit fused decode / verify kernel vs an exact FP32 reference over the same quantized cache: 1–8 queries, 3 to 131k keys, strided cache views, plus timing | ≤ 2.6e-3 (BF16 rounding); 2.2× at 64k, 2.9× at 128k |
| `tests/kv8_quality.py` | 8-bit vs BF16 KV after a 32k prompt, against the BF16 block-vs-step noise floor | see above |
| `tests/check_kvquant.py` | 8 / 4-bit full-attention KV: tiny-model error, then real-weight KL, speed and memory at 16k | see "KV cache and long contexts" |
| `tests/precision_vs_fp32.py` | Real weights: GPU-BF16 and CPU-BF16 vs a CPU-FP32 ground truth | GPU: median KL 1.2e-3, top-1 97.2 % (no worse than CPU) |
| `tests/dense_quality.py` | KL cost of `--dense-bits` 8 / 6 / 4 vs BF16, on the model's own greedy text (optimistic: near-certain tokens) | see Performance, item 2 |
| `tests/dense_precision.py` | Real weights, fixed English + Chinese text: GPU at dense 16 / 8 / 6 bits vs CPU-FP32 truth over 406 prefill + 31 decode positions | median KL 1.45e-3 / 1.47e-3 / 3.61e-3 |
| `tests/bench_dense_bits.py` | End-to-end plain / DSpark tok/s and resident memory at two `--dense-bits` | 6 bits +7 % plain |
| `tests/check_gated_spec.py` | Real weights, confidence-gated DSpark: draft-feature bookkeeping (draft KV offset + pending features = target length, after generations with plain fallbacks and across two turns), and greedy output vs plain decoding with the margin at the first divergence | bookkeeping exact; divergences at near-ties only (margins 0.00–0.16) |
| `tests/bench_spec_policy.py` | End-to-end plain vs geometric vs confidence-gated DSpark, prose and code, T = 0 / 1 | see Performance, item 10 |
| `tests/check_server_worker.py` | Worker protocol on real weights. Turn 2 with the history echoed as clients send it reuses turn 1's KV, two clones of one stored state give token-identical output, a long system prompt is snapshotted and reused, plus tool calls, stop sequences and cancel | all pass; snapshot hit 1,045 of 1,052 tokens |
| `tests/check_launcher.py`, `tests/check_launcher_sdk.py` | The Rust launcher end to end over HTTP, and with the official `openai` SDK: both APIs streaming and not, tool calls, `previous_response_id`, GET / DELETE, disconnect cancels, errors, turn-2 latency | all pass; turn 2 of an 8k conversation 9.25 s cold → 0.29–0.32 s |
| `tests/check_screened_draft.py` | Screened vs full-vocabulary Markov drafting on the same draft features every round (prose and code, T = 0 / 1): draft agreement up to BF16 ties in the reference, shortlist coverage, confidence difference, Markov-loop time | 256 × top-16: 98.8 % agreement, 1.6 vs 2.6 ms |
| `tests/trace_spec.py` + `tests/sim_spec_policy.py` | Fixed-block-8 DSpark traces (per-position draft probability, accepted count, draft / verify time, the V(L) curve), and a replay of verify-length policies on them, including an oracle | see Performance, item 10 |
| `tests/bench_*.py`, `tests/profile_decode.py` | Speed tables and the decode breakdown | numbers above |

## Architecture notes

- 24 decoder layers, hidden 2304, 128 experts with top-8 sigmoid routing. Layers 5, 11, 17
  and 23 use full attention with dynamic K/V convolutions; the others use 1024-token
  sliding windows. Attention has learned relative-position bias, per-token sinks and
  L2-clipped Q/K. Block attention residuals take one block per 6 layers.
- **Loop**: layers 6–17 run 3 times (48 layer *visits* per token). Passes 2 and 3 restart
  from the pre-loop hidden state and block list. Each loop layer then reads *messages* from
  the previous pass: an attention residual over `[blocks after loop entry, final hidden]`
  with 4 learned queries per layer (q, k, v, mlp). The messages are mixed into the layer's
  inputs by `MessageGate`: `(1 - tanh(F z)) * h + tanh(G z) * m`, where
  `z = D [rms(h), rms(m)]`.
- **Caches**: every visit has its own KV and short-conv cache: 24 + 2 × 12 = 48.
- **Experts**: layers 0–19 are NVFP4 (E2M1 values, an E4M3 scale per 16, and a per-expert
  `amax / outer_scale_denominator` factor). Layers 20–23 are MXFP8 (E4M3 values, a UE8M0
  scale per 32). The per-layer assignment comes from `hf_quant_config.json`.

## Pitfalls

- **Small precision tests mislead.** `check_band_model.py`, `check_kvquant.py` and
  `check_ring.py` take their input text from this README, so their numbers move whenever it
  is edited. With 13 to 36 positions, one outlier dominates the mean KL. To judge whether
  a kernel costs precision, use `tests/kernel_precision.py`: fixed text, about 180
  positions, paired variants against CPU-FP32 truth.
- **`lm(ids)` returns only the last position's logits** unless `last_only=False`.
- **Wrong quantization from mlx-vlm's auto-detection.** `config.json` says
  `quantization_config: {quant_method: modelopt, quant_algo: NVFP4}`, and mlx-vlm's generic
  loader would apply NVFP4 everywhere. The truth is per layer, in `hf_quant_config.json`
  (`quant_method: inkling2`).
- **NVFP4 global scale.** MLX's `gather_qmm(..., global_scale=g)` multiplies by `g / 2688`
  per expert, on the GPU only. The CPU path has to multiply the output instead.
- `w13` rows interleave gate (even) and up (odd), and so do their scales. Both halves share
  one global scale per expert.
- `middle_cycle_cross_pass_queries` holds `[q, mlp]` pairs per loop layer, and
  `middle_cycle_cross_pass_kv_queries` holds `[k, v]` pairs. Merge them as `[q, k, v, mlp]`.
  `controllers.attn` is the **query** gate.
- On later passes, the message source skips the block appended at loop entry.
- **FP32 on the M-series GPU isn't FP32 for matmuls**: about 1e-3 relative error, because
  they run on the neural accelerators. Exact parity checks have to use `mx.cpu`; custom Metal
  kernels are exact FP32.
- **BF16 noise is large for this model.** MoE routing flips at near-ties, so even CPU-BF16 vs
  FP32 gives a mean KL of about 0.025. Greedy outputs of different but correct code paths
  diverge after tens of tokens at margins around 0.2 to 0.3. Strict checks need FP32 on the
  CPU, or one shared kernel path.
- **`gather_qmm` deduplicates expert reads only from 4 rows per expert on average.** Below
  that it re-reads per route (see Performance).
- DSpark `target_layer_ids` `[5, 23, 41, 48]` index **visits** (inputs to them), not stored
  layers. mlx-vlm's DSpark config validation assumes `ids[-1] == num_hidden_layers`, so the
  draft is configured with 48.
- Metal program-scope constants in `mx.fast.metal_kernel` headers must be
  `constant constexpr`.
- The chat template has no generation prompt. The model writes
  `<|message_model|><|content_thinking|>…<|end_message|><|message_model|><|content_text|>…<|end_message|>`
  and stops with `<|content_model_end_sampling|>` (id 200006). Even at `reasoning_effort=none`
  it writes a short plan in the thinking segment first.
- **Never process past the last emitted token.** Ring buffers and short-convolution states
  cannot be rewound by trimming. A speculative round that stops at EOS or at `max_tokens`
  therefore commits only up to the last emitted token, and `GenState.pending` records emitted
  tokens the model has not processed yet.
- **Multi-turn reuse depends on the chat template being append-only per message.** It is,
  including the reasoning-effort message, which is emitted once before the first non-system
  message. `Session` checks the prefix and falls back to a full prefill if it does not hold.
- The system Python here is 3.15. Use a 3.12 venv for MLX wheels.

## Not done / next steps

- KV eviction or compression for the full-attention visits (StreamingLLM-style sinks + window,
  SnapKV and similar). These would bound decode cost at very long contexts but change the
  model's output; nothing here does that.
- Decode-time band attention. Single-token decode still builds a small 1 × context mask per
  full visit, which is cheap next to reading the KV.

- A server with *continuous* batching: admit new requests into the running batch with
  `CacheList.extend`. The batch API here is static.
- DSpark inside batches (ragged acceptance per row).
- A grouped MXFP8 expert kernel. Layers 20–23 (4 of 48 visits) still use `gather_qmm`.
- Faster fused kernels for the remaining per-visit glue, and a grouped kernel that reaches the
  bandwidth ceiling.
- The variable-loop research model `thinkingmachines/TML-d24-varloop-research` can run with
  1 or 2 passes for speed. It is a different checkpoint and is not supported here
  (`is_varloop`).
