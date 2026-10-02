"""
Evaluate any stage checkpoint on GSM8K (greedy) and optionally dump sample generations.
Use it to build the headline "GSM8K accuracy across stages" table:

    for s in base_pretrained sft dpo ppo grpo; do
      python scripts/eval_post_training.py --ckpt models/$s.pt \
        --label $s --limit 200 --append logs/stage_table.jsonl
    done
    python scripts/eval_post_training.py --table logs/stage_table.jsonl

Model dimensions are read from the checkpoint's stored ``cfg`` so you don't have to repeat
them. Reward checkpoints (which have a reward head, not an LM head only) still load because
we keep just the backbone keys for generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the repo without installing

import argparse
import json
import os

import torch

from src.post_training.evaluation import gsm8k_accuracy, load_gsm8k_eval
from src.post_training.inference import load_model_from_ckpt

# Kept so older notebooks that imported this name keep working.
model_from_ckpt = load_model_from_ckpt


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

    model = load_model_from_ckpt(args.ckpt, args.device)
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
