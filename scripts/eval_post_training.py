"""
Evaluate any stage checkpoint on GSM8K (greedy) and optionally dump sample generations.
Use it to build the headline "GSM8K accuracy across stages" table:

    for s in base_pretrained sft dpo ppo grpo; do
      PYTHONPATH=. python scripts/eval_post_training.py --ckpt /ephemeral/ckpts/$s.pt \
        --label $s --limit 200 --append /ephemeral/logs/stage_table.jsonl
    done
    PYTHONPATH=. python scripts/eval_post_training.py --table /ephemeral/logs/stage_table.jsonl

Model dimensions are read from the checkpoint's stored ``cfg`` so you don't have to repeat
them. Reward checkpoints (which have a reward head, not an LM head only) still load because
we keep just the backbone keys for generation.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from src.models.transformer import Transformer
from src.post_training.evaluation import gsm8k_accuracy, load_gsm8k_eval


def model_from_ckpt(ckpt_path: str, device: str, overrides: dict | None = None) -> Transformer:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {}) or {}
    cfg = {**cfg, **(overrides or {})}
    model = Transformer(
        n_head=cfg.get("n_head", 16), n_embed=cfg.get("n_embed", 1024),
        context_length=cfg.get("context_length", 1024), vocab_size=cfg.get("vocab_size", 50304),
        N_BLOCKS=cfg.get("n_blocks", 24),
    )
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    state = {k.removeprefix("module.").removeprefix("transformer."): v for k, v in state.items()}
    backbone_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in backbone_keys}
    model.load_state_dict(filtered, strict=False)
    return model.to(device).eval()


def print_table(path: str):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(f"\n{'stage':<18}{'GSM8K acc':>10}{'n':>8}")
    print("-" * 36)
    for r in rows:
        print(f"{r['label']:<18}{r['accuracy']*100:>9.1f}%{r['n']:>8}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--label", default="model")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--split", default="test")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--append", default=None, help="append the result row to this JSONL")
    p.add_argument("--table", default=None, help="just print a stage table from this JSONL and exit")
    args = p.parse_args()

    if args.table:
        print_table(args.table)
        return

    model = model_from_ckpt(args.ckpt, args.device)
    qa = load_gsm8k_eval(args.split, limit=args.limit)
    res = gsm8k_accuracy(model, qa, device=args.device, max_new_tokens=args.max_new_tokens,
                         greedy=True, return_samples=args.samples)
    print(f"[{args.label}] GSM8K {args.split} accuracy: {res['accuracy']*100:.1f}%  ({res['correct']}/{res['n']})")
    for s in res["samples"]:
        print(f"\n  Q: {s['q'][:120]}\n  gold={s['gold']} correct={s['correct']}\n  A: {s['response'][:300]}")

    if args.append:
        os.makedirs(os.path.dirname(args.append) or ".", exist_ok=True)
        with open(args.append, "a") as f:
            f.write(json.dumps({"label": args.label, "accuracy": res["accuracy"],
                                "correct": res["correct"], "n": res["n"]}) + "\n")
        print(f"\nappended -> {args.append}")


if __name__ == "__main__":
    main()
