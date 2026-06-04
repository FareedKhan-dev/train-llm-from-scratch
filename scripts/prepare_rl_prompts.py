"""
Build RL prompt sets ({"prompt", "gold"}) for PPO/GRPO:
  - GSM8K train -> rl_prompts_train.jsonl   (the RL training prompts)
  - GSM8K test  -> rl_prompts_test.jsonl    (held-out benchmark for eval)
  - a programmatic arithmetic set -> arithmetic_prompts.jsonl  (RL curriculum warm-up,
    where even a weak model gets some non-zero reward so RL has signal to start from)

Example:
    PYTHONPATH=. HF_HOME=/ephemeral/hf_cache python scripts/prepare_rl_prompts.py --out_dir /ephemeral/data
"""

from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("HF_HOME", "/ephemeral/hf_cache")

from src.post_training.rewards import gsm8k_gold_answer


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} prompts -> {path}")


def gsm8k_prompts(split: str, limit: int | None):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for ex in ds:
        gold = gsm8k_gold_answer(ex["answer"])
        if gold is not None:
            rows.append({"prompt": ex["question"].strip(), "gold": gold})
    return rows


def arithmetic_prompts(n: int, max_val: int, seed: int):
    rng = random.Random(seed)
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
    rows = []
    for _ in range(n):
        a, b = rng.randint(0, max_val), rng.randint(0, max_val)
        sym, fn = rng.choice(ops)
        rows.append({"prompt": f"What is {a} {sym} {b}?", "gold": float(fn(a, b))})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/ephemeral/data")
    p.add_argument("--train_limit", type=int, default=None)
    p.add_argument("--test_limit", type=int, default=500)
    p.add_argument("--arith_n", type=int, default=5000)
    p.add_argument("--arith_max", type=int, default=20)
    args = p.parse_args()

    print("Loading GSM8K train ...")
    write_jsonl(gsm8k_prompts("train", args.train_limit), os.path.join(args.out_dir, "rl_prompts_train.jsonl"))
    print("Loading GSM8K test ...")
    write_jsonl(gsm8k_prompts("test", args.test_limit), os.path.join(args.out_dir, "rl_prompts_test.jsonl"))
    print("Generating arithmetic curriculum ...")
    write_jsonl(arithmetic_prompts(args.arith_n, args.arith_max, seed=0),
                os.path.join(args.out_dir, "arithmetic_prompts.jsonl"))


if __name__ == "__main__":
    main()
