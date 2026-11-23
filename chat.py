#!/usr/bin/env python
"""Chat with Inkling 2 Mini on Apple silicon (MLX, native NVFP4 / MXFP8 experts).

  python chat.py -p "Explain looped transformers in two sentences."
  python chat.py -p "What is in this picture?" --image cat.jpg
  python chat.py            # interactive; /reset clears history, /quit exits
  python chat.py --batch prompts.txt   # one prompt per line, decoded together
  python chat.py --file paper.txt     # long document as context, then ask away
  Interactive commands: /reset, /save PATH, /load PATH, /quit
"""

import argparse
import sys
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "Inkling-2-Mini"
DIM, RESET = "\033[2m", "\033[0m"


def parse_effort(value):
    try:
        return float(value)
    except ValueError:
        return value


def run_turn(session, text, images, args):
    reply = {"thinking": "", "text": "", "tool": ""}
    last = None
    stats = None
    for ev in session.stream(
        text,
        images=images,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        speculative=not args.no_spec,
        block_size=args.block_size,
    ):
        if ev.kind == "done":
            stats = ev.stats
            continue
        if ev.kind != last:
            if last is not None:
                sys.stdout.write(RESET + "\n")
            if ev.kind == "thinking" and not args.hide_thinking:
                sys.stdout.write(DIM + "[thinking] ")
            elif ev.kind == "tool":
                sys.stdout.write("[tool call] ")
            last = ev.kind
        reply[ev.kind] += ev.text
        if not (ev.kind == "thinking" and args.hide_thinking):
            sys.stdout.write(ev.text)
        sys.stdout.flush()
    sys.stdout.write(RESET + "\n")
    if stats:
        print(
            f"{DIM}[context {stats['context_tokens']} | new prompt {stats['prompt_tokens']} tok @ {stats['prefill_tps']:.1f} tok/s | "
            f"gen {stats['gen_tokens']} tok @ {stats['gen_tps']:.1f} tok/s | "
            f"peak {stats['peak_gb']:.1f} GB"
            + (f" | DSpark {stats['tokens_per_round']:.2f} tok/round, "
               f"accept {stats['acceptance']:.0%}" if "tokens_per_round" in stats else "")
            + f"]{RESET}"
        )
    return reply


from inkling2_mini.generate import user_message  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("-p", "--prompt")
    ap.add_argument("--image", action="append", default=[], help="image path (repeatable)")
    ap.add_argument("--system")
    ap.add_argument("--effort", type=parse_effort, default="low",
                    help="reasoning effort: none|minimal|low|medium|high|xhigh|max or 0..0.99 (template default 0.9)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--hide-thinking", action="store_true")
    ap.add_argument("--batch", metavar="FILE", help="decode every line of FILE as its own request, batched")
    ap.add_argument("--file", help="text file prepended to the first message (long-document Q&A)")
    ap.add_argument("--session", help="load a saved session (/save PATH in interactive mode)")
    ap.add_argument("--kv-bits", type=int, choices=[4, 8, 16], default=8,
                    help="full-attention KV cache precision: 8 (default; fused decode kernel, ~22%% faster "
                         "decode at 64k, half the memory), 16 = BF16, 4 = lossy")
    ap.add_argument("--no-spec", action="store_true", help="disable DSpark speculative decoding")
    ap.add_argument("--block-size", type=int, help="fixed DSpark verify block 2-8 (default: adaptive)")
    ap.add_argument("--passes", type=int,
                    help="loop passes (default: the checkpoint's, 3); variable-loop checkpoints "
                         "(TML-d24-varloop-research) take 1-5. Interactive: /passes N starts over with N")
    ap.add_argument("--dense-bits", type=int, choices=[4, 5, 6, 8, 16], default=6,
                    help="attention projections + lm_head precision: 6 (default, at the model's own "
                         "BF16 noise, fastest), 8 = near-lossless, 16 = the checkpoint's native BF16, "
                         "5/4 = smaller and lossy")
    args = ap.parse_args()

    from PIL import Image

    from inkling2_mini.generate import Inkling2Mini

    bot = Inkling2Mini(args.model, speculative=not args.no_spec and not args.batch, dense_bits=args.dense_bits,
                       kv_bits=args.kv_bits, passes=args.passes)
    history = [{"role": "system", "content": args.system}] if args.system else []

    if args.batch:
        prompts = [line.strip() for line in open(args.batch, encoding="utf-8") if line.strip()]
        convs = [history + [user_message(p, [])] for p in prompts]
        replies, st = bot.chat_batch(convs, reasoning_effort=args.effort, max_tokens=args.max_tokens,
                                     temperature=args.temp, top_p=args.top_p, top_k=args.top_k, seed=args.seed)
        for i, (p, r) in enumerate(zip(prompts, replies)):
            print(f"\n=== [{i + 1}] {p}")
            if r.thinking and not args.hide_thinking:
                print(DIM + "[thinking] " + r.thinking.strip() + RESET)
            print(r.text.strip())
        print(f"{DIM}[{st['batch']} requests | {st['gen_tokens']} tokens in {st['gen_s']:.1f} s = "
              f"{st['gen_tps']:.1f} tok/s total | prefill {st['prefill_s']:.1f} s | peak {st['peak_gb']:.1f} GB]{RESET}")
        return

    from inkling2_mini.generate import Session

    session = Session.load(bot, args.session) if args.session else Session(
        bot, system=args.system, reasoning_effort=args.effort)
    prefix = open(args.file, encoding="utf-8").read().strip() + "\n\n" if args.file else ""

    if args.prompt is not None:
        images = [Image.open(p) for p in args.image]
        run_turn(session, prefix + args.prompt, images, args)
        return

    print("Inkling 2 Mini — /reset, /save PATH, /load PATH, /passes N, /quit"
          + (f"  (session with {session.context_tokens} tokens loaded)" if args.session else ""))
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            session.reset()
            print("(history cleared)")
            continue
        if text.startswith("/save "):
            session.save(text[6:].strip())
            print(f"(saved {session.context_tokens} tokens of context)")
            continue
        if text.startswith("/load "):
            session = Session.load(bot, text[6:].strip())
            print(f"(loaded {session.context_tokens} tokens of context)")
            continue
        if text.startswith("/passes"):
            # The KV of a conversation belongs to its pass count's layer visits: start over.
            try:
                bot.passes = bot.check_passes(text.split()[1])
            except (IndexError, ValueError) as e:
                print(f"({e or 'usage: /passes N'}; now {bot.passes})")
                continue
            session.reset()
            print(f"({bot.passes} pass{'es' if bot.passes > 1 else ''}; history cleared)")
            continue
        run_turn(session, prefix + text, None, args)
        prefix = ""


if __name__ == "__main__":
    main()
