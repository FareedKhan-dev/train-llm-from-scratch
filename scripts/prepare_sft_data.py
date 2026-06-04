"""
Build the SFT dataset from real public instruction data + GSM8K, render it through the
chat template (masking prompt tokens), pack to fixed-length rows, and write a packed
HDF5 (``tokens`` + ``loss_mask``) plus a held-out dev file.

Datasets (downloaded via HuggingFace ``datasets`` to /ephemeral/hf_cache):
  - tatsu-lab/alpaca                (general instruction following)
  - databricks/databricks-dolly-15k (general instruction following)
  - openai/gsm8k (main, train)      (math, reformatted into <think>/<answer> so the model
                                     learns the reasoning format the RL verifier rewards)

Example:
    PYTHONPATH=. HF_HOME=/ephemeral/hf_cache python scripts/prepare_sft_data.py \
        --context_length 1024 --out_dir /ephemeral/data
"""

from __future__ import annotations

import argparse
import os
import re

import h5py
import numpy as np

from src.post_training.chat_template import encode_chat, ANSWER_OPEN, ANSWER_CLOSE, THINK_OPEN, THINK_CLOSE
from src.post_training.sft import pack_examples

os.environ.setdefault("HF_HOME", "/ephemeral/hf_cache")
_CALC_RE = re.compile(r"<<[^>]*>>")          # GSM8K calculator annotations
_HASH_RE = re.compile(r"####\s*(.+)\s*$")


def gsm8k_to_messages(question: str, answer: str) -> list[dict]:
    """Reformat a GSM8K (question, answer) into chat messages whose assistant turn uses
    the <think>...</think><answer>N</answer> structure."""
    answer = _CALC_RE.sub("", answer).strip()
    m = _HASH_RE.search(answer)
    final = m.group(1).strip() if m else answer
    reasoning = _HASH_RE.sub("", answer).strip()
    completion = f"{THINK_OPEN}{reasoning}{THINK_CLOSE}{ANSWER_OPEN}{final}{ANSWER_CLOSE}"
    return [{"role": "user", "content": question.strip()},
            {"role": "assistant", "content": completion}]


def alpaca_to_messages(ex: dict) -> list[dict]:
    instr = ex["instruction"].strip()
    inp = (ex.get("input") or "").strip()
    user = f"{instr}\n\n{inp}" if inp else instr
    return [{"role": "user", "content": user}, {"role": "assistant", "content": ex["output"].strip()}]


def dolly_to_messages(ex: dict) -> list[dict]:
    instr = ex["instruction"].strip()
    ctx = (ex.get("context") or "").strip()
    user = f"{instr}\n\n{ctx}" if ctx else instr
    return [{"role": "user", "content": user}, {"role": "assistant", "content": ex["response"].strip()}]


def collect_examples(context_length: int, limit_per_set: int | None) -> list[tuple[list[int], list[int]]]:
    from datasets import load_dataset

    examples: list[tuple[list[int], list[int]]] = []
    n_kept = n_skipped = 0

    def add(messages):
        nonlocal n_kept, n_skipped
        ids, mask = encode_chat(messages)
        if len(ids) <= context_length and sum(mask) > 0:
            examples.append((ids, mask))
            n_kept += 1
        else:
            n_skipped += 1

    print("Loading tatsu-lab/alpaca ...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for ex in (alpaca.select(range(min(limit_per_set, len(alpaca)))) if limit_per_set else alpaca):
        add(alpaca_to_messages(ex))

    print("Loading databricks/databricks-dolly-15k ...")
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        for ex in (dolly.select(range(min(limit_per_set, len(dolly)))) if limit_per_set else dolly):
            add(dolly_to_messages(ex))
    except Exception as e:  # noqa: BLE001
        print(f"  (skipping dolly: {e})")

    print("Loading openai/gsm8k (main/train) ...")
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for ex in (gsm.select(range(min(limit_per_set, len(gsm)))) if limit_per_set else gsm):
        add(gsm8k_to_messages(ex["question"], ex["answer"]))

    print(f"Collected {n_kept} examples (skipped {n_skipped} that exceeded context_length).")
    return examples


def write_packed(examples, context_length: int, out_path: str) -> int:
    tokens, masks = pack_examples(examples, context_length)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("tokens", data=tokens)
        f.create_dataset("loss_mask", data=masks)
    print(f"  wrote {tokens.shape[0]} packed rows x {context_length} -> {out_path}")
    return tokens.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context_length", type=int, default=1024)
    p.add_argument("--out_dir", default="/ephemeral/data")
    p.add_argument("--dev_frac", type=float, default=0.02)
    p.add_argument("--limit_per_set", type=int, default=None, help="cap examples per dataset (debug)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    examples = collect_examples(args.context_length, args.limit_per_set)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(examples)
    n_dev = max(1, int(len(examples) * args.dev_frac))
    dev, train = examples[:n_dev], examples[n_dev:]

    write_packed(train, args.context_length, os.path.join(args.out_dir, "sft_packed.h5"))
    write_packed(dev, args.context_length, os.path.join(args.out_dir, "sft_dev_packed.h5"))


if __name__ == "__main__":
    main()
