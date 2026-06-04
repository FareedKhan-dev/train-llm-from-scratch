"""
Chat / inference CLI for any stage checkpoint (base, SFT, DPO, PPO, GRPO).

Model dimensions are read from the checkpoint, so you only pass the path. Use the chat
template for instruction-tuned models, or --raw for base-model continuation.

One-shot:
    PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/sft.pt --prompt "What is 13 + 29?"
    PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/grpo.pt --prompt "..." --greedy
    PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/base_pretrained.pt --raw --prompt "Once upon a time"
Interactive REPL (no --prompt):
    PYTHONPATH=. python scripts/chat.py --ckpt /ephemeral/ckpts/sft.pt
"""

from __future__ import annotations

import argparse

import torch

from src.post_training.inference import generate_reply, load_model_from_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default=None, help="one-shot prompt; omit for interactive REPL")
    p.add_argument("--system", default=None, help="optional system message (chat mode)")
    p.add_argument("--raw", action="store_true", help="base-model continuation (no chat template)")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--greedy", action="store_true", help="deterministic argmax decoding")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model = load_model_from_ckpt(args.ckpt, args.device)
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.ckpt} ({n/1e6:.0f}M params) on {args.device} | "
          f"mode={'raw' if args.raw else 'chat'} {'greedy' if args.greedy else f'T={args.temperature} top_p={args.top_p}'}")

    def reply(text):
        return generate_reply(model, text, device=args.device, system=args.system, raw=args.raw,
                              max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                              top_p=args.top_p if args.top_p < 1 else None, top_k=args.top_k, greedy=args.greedy)

    if args.prompt is not None:
        print(reply(args.prompt))
        return

    print("Interactive chat (Ctrl-D / 'exit' to quit).")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            break
        if text in ("exit", "quit"):
            break
        if text:
            print("bot>", reply(text))


if __name__ == "__main__":
    main()
