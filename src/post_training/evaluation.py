"""
Shared evaluation utilities for post-training stages.

The headline metric across the whole pipeline is GSM8K accuracy (greedy decoding) on a
fixed held-out set, so Base -> SFT -> DPO -> PPO -> GRPO can be compared on one axis.
``batched_generate`` length-buckets prompts so the mask-free model can decode many at once.
"""

from __future__ import annotations

import torch

from src.post_training.chat_template import EOT_ID, decode, encode_prompt
from src.post_training.rollout import generate_with_logprobs
from src.post_training.rewards import gsm8k_gold_answer, is_correct


def _model_context_length(model) -> int:
    inner = getattr(model, "transformer", model)
    return getattr(inner, "context_length", 1024)


@torch.no_grad()
def batched_generate(
    model,
    prompts: list[list[int]],
    max_new_tokens: int,
    *,
    device: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    micro_batch: int = 32,
) -> list[str]:
    """
    Generate a response string for each tokenized prompt, returned in input order.

    Prompts are grouped by identical length (the model has no padding-aware attention
    mask) and decoded in micro-batches. ``greedy=True`` forces argmax decoding (top_k=1),
    which is what we use for comparable eval numbers.
    """
    if greedy:
        temperature, top_k, top_p = 1.0, 1, None

    cap = _model_context_length(model)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results: list[str | None] = [None] * len(prompts)

    i = 0
    while i < len(order):
        j = i
        L = len(prompts[order[i]])
        # Gather a run of same-length prompts, up to micro_batch.
        while j < len(order) and len(prompts[order[j]]) == L and (j - i) < micro_batch:
            j += 1
        idxs = order[i:j]
        i = j
        budget = min(max_new_tokens, cap - L)
        if budget <= 0:  # prompt already fills the context; nothing to generate
            for k in idxs:
                results[k] = ""
            continue
        batch = torch.tensor([prompts[k] for k in idxs], dtype=torch.long, device=device)
        rb = generate_with_logprobs(model, batch, budget, temperature=temperature,
                                    top_k=top_k, top_p=top_p, stop_tokens=(EOT_ID,), pad_id=EOT_ID)
        gen = rb.sequences[:, L:]
        for row, k in enumerate(idxs):
            toks = gen[row].tolist()
            if EOT_ID in toks:
                toks = toks[: toks.index(EOT_ID)]
            results[k] = decode(toks)
    return [r or "" for r in results]


@torch.no_grad()
def gsm8k_accuracy(
    model,
    qa_pairs: list[tuple[str, str]],
    *,
    device: str,
    max_new_tokens: int = 300,
    greedy: bool = True,
    return_samples: int = 0,
) -> dict:
    """
    Compute greedy GSM8K accuracy over ``(question, answer_field)`` pairs.

    Returns ``{"accuracy", "n", "correct", "samples"}`` where ``samples`` holds a few
    (question, response, gold, correct) tuples for qualitative inspection.
    """
    prompts = [encode_prompt([{"role": "user", "content": q}]) for q, _ in qa_pairs]
    responses = batched_generate(model, prompts, max_new_tokens, device=device, greedy=greedy)

    correct = 0
    samples = []
    for (q, ans), resp in zip(qa_pairs, responses):
        gold = gsm8k_gold_answer(ans)
        ok = is_correct(resp, gold)
        correct += int(ok)
        if len(samples) < return_samples:
            samples.append({"q": q, "response": resp, "gold": gold, "correct": ok})
    n = len(qa_pairs)
    return {"accuracy": correct / max(1, n), "n": n, "correct": correct, "samples": samples}


def load_gsm8k_eval(split: str = "test", limit: int | None = 200) -> list[tuple[str, str]]:
    """Load ``(question, answer_field)`` pairs from GSM8K for evaluation."""
    import os
    os.environ.setdefault("HF_HOME", "/ephemeral/hf_cache")
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [(ex["question"], ex["answer"]) for ex in ds]
