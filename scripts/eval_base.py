"""
Benchmark a base model: ARC-Easy, ARC-Challenge and LAMBADA, scored as lm-evaluation-harness
scores them, so the numbers compare with published ones.

    python scripts/eval_base.py --ckpt models/base_pretrained.pt
    python scripts/eval_base.py --ckpt models/base_pretrained.pt --tasks arc_easy --limit 500
    python scripts/eval_base.py --hf_model openai-community/gpt2      # a reference point

Works with every checkpoint in the repo (either trainer, either architecture, any stage).
The model is only asked how likely each answer is, so it needs no chat format and no
post-training. ``--hf_model`` scores a Hugging Face model with the same code instead, as a
reference (needs ``pip install transformers``). ``--append FILE`` adds the result as a JSON
line, to build a table across checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the repo without installing

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import torch

from src.benchmarks import TASKS, evaluate_task, load_examples
from src.checkpoint import (
    load_checkpoint,
    load_model_weights,
    model_config_from_checkpoint,
    model_state_from_checkpoint,
)
from src.device import resolve_device
from src.models.factory import build_model
from src.tokenizer import tokenizer_from_spec

# The settings a checkpoint predating stored configs was trained with (as in inference.py).
_DEFAULTS = {"n_head": 16, "n_embed": 1024, "context_length": 1024, "vocab_size": 50304, "n_blocks": 24}


def load_repo_model(path: str, device: str) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[str], list[int]], int]:
    """A checkpoint from this repo -> (logits_fn, encode, window)."""
    ck = load_checkpoint(path, map_location="cpu")
    cfg = {**_DEFAULTS, **model_config_from_checkpoint(ck)}
    model = build_model(cfg)
    load_model_weights(model, model_state_from_checkpoint(ck, extra_prefixes=("transformer.",)), source=path)
    model.to(device).eval()
    tokenizer = tokenizer_from_spec(cfg.get("tokenizer"))
    # The legacy trainer may train on windows shorter than the model's context; positions it
    # never saw have untrained embeddings, so scoring stays inside the trained window.
    window = int(min(cfg.get("t_context_length") or cfg["context_length"], cfg["context_length"]))
    return (lambda ids: model(ids)[0]), tokenizer.encode_ordinary, window


def load_hf_model(name: str, device: str) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[str], list[int]], int]:
    """A Hugging Face causal LM -> (logits_fn, encode, window), scored by the same code."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit("--hf_model needs transformers: pip install transformers") from e
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    window = int(getattr(model.config, "n_positions", None) or model.config.max_position_embeddings)
    return (lambda ids: model(ids).logits), (lambda s: tok.encode(s, add_special_tokens=False)), window


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="a checkpoint from this repo")
    src.add_argument("--hf_model", help="a Hugging Face model id, as a reference point")
    p.add_argument("--tasks", default="arc_easy,arc_challenge,lambada",
                   help=f"comma-separated, from: {', '.join(TASKS)}")
    p.add_argument("--limit", type=int, default=None, help="score only the first N examples of each task")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=8, help="continuations per forward pass")
    p.add_argument("--append", default=None, help="append the result as a JSON line to this file")
    p.add_argument("--label", default=None, help="name of this row in --append (default: the model)")
    args = p.parse_args()

    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in names if t not in TASKS]
    if unknown or not names:
        raise SystemExit(f"unknown task(s) {unknown}; choose from {', '.join(TASKS)}")
    device = resolve_device(args.device)
    logits_fn, encode, window = (load_repo_model(args.ckpt, device) if args.ckpt
                                 else load_hf_model(args.hf_model, device))
    print(f"{args.ckpt or args.hf_model} on {device}, window {window} tokens")

    results: dict[str, Any] = {}
    for name in names:
        task = TASKS[name]
        examples = load_examples(task, args.limit)
        t0 = time.perf_counter()
        r = evaluate_task(task, examples, logits_fn, encode, max_length=window, device=device,
                          batch_size=args.batch_size)
        results[name] = r
        extra = (f"acc_norm {100 * r['acc_norm']:.2f}" if "acc_norm" in r
                 else f"perplexity {r['perplexity']:.2f}")
        print(f"  {name:<14} acc {100 * r['acc']:.2f} +- {100 * r['acc_stderr']:.2f} | {extra} | "
              f"n={r['n']} | {time.perf_counter() - t0:.0f}s")

    if args.append:
        row = {"label": args.label or args.ckpt or args.hf_model, "limit": args.limit, "results": results}
        Path(args.append).parent.mkdir(parents=True, exist_ok=True)
        with open(args.append, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"appended to {args.append}")


if __name__ == "__main__":
    main()
