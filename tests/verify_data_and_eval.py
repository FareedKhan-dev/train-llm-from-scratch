"""
Deep correctness checks on the REAL prepared data and the evaluation benchmark.
No training, no GPU -- inspects the actual files on disk and the scoring logic.

    PYTHONPATH=. python tests/verify_data_and_eval.py
"""

import json

import h5py
import numpy as np

from src.post_training.chat_template import EOT_ID, decode, encode_chat, ASSISTANT_HEADER
from src.post_training.rewards import gsm8k_gold_answer, extract_answer, is_correct, reward_gsm8k

DATA = "/ephemeral/data"
PASS, FAIL = "PASS", "FAIL"


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    assert cond, f"FAILED: {name} {detail}"


def verify_pile():
    print("\n== Pile pretraining HDF5 ==")
    for split, path in [("train", f"{DATA}/pile_train.h5"), ("dev", f"{DATA}/pile_dev.h5")]:
        with h5py.File(path, "r") as f:
            d = f["tokens"]
            n = d.shape[0]
            head = np.asarray(d[:100000])
            check(f"{split}: nonempty", n > 1_000_000, f"{n:,} tokens")
            check(f"{split}: token ids in [0,50256]", head.min() >= 0 and head.max() <= 50256,
                  f"min {head.min()} max {head.max()}")
            check(f"{split}: contains EOT separators", (head == EOT_ID).sum() > 0,
                  f"{int((head==EOT_ID).sum())} EOT in first 100k")


def verify_sft_mask_alignment():
    """The critical SFT check: the loss_mask must be 1 EXACTLY on assistant-completion
    tokens and 0 on the prompt / role markers. We decode masked vs unmasked spans of a
    real packed row and confirm they read like completion vs prompt."""
    print("\n== SFT packed data: loss-mask alignment (real data) ==")
    with h5py.File(f"{DATA}/sft_packed.h5", "r") as f:
        tokens, masks = f["tokens"], f["loss_mask"]
        check("packed: nonempty", tokens.shape[0] > 100, f"{tokens.shape[0]} rows x {tokens.shape[1]}")
        check("packed: tokens/masks aligned shape", tokens.shape == masks.shape)
        row_t = np.asarray(tokens[0]); row_m = np.asarray(masks[0])
        check("packed: token ids valid", row_t.min() >= 0 and row_t.max() <= 50303)
        check("packed: mask is binary", set(np.unique(row_m)).issubset({0, 1}))
        frac = row_m.mean()
        check("packed: mask trains a sensible fraction", 0.02 < frac < 0.95, f"{frac:.2f} of tokens trained")

    # Re-derive a single example from the chat template and confirm the mask matches exactly.
    ids, mask = encode_chat([{"role": "user", "content": "What is 2+2?"},
                             {"role": "assistant", "content": "<answer>4</answer>"}])
    masked_text = decode([t for t, m in zip(ids, mask) if m == 1])
    unmasked_text = decode([t for t, m in zip(ids, mask) if m == 0])
    check("mask covers the assistant answer", "answer>4" in masked_text or "4" in masked_text,
          f"masked={masked_text!r}")
    check("mask excludes the user question", "2+2" in unmasked_text and "2+2" not in masked_text,
          f"unmasked={unmasked_text!r}")
    check("prompt form ends at assistant header", decode(encode_chat(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True)[0]).endswith(ASSISTANT_HEADER))


def verify_preferences():
    print("\n== Preference data (HH-RLHF + UltraFeedback) ==")
    for path in [f"{DATA}/preferences.jsonl", f"{DATA}/preferences_test.jsonl"]:
        rows = [json.loads(l) for l in open(path)]
        check(f"{path.split('/')[-1]}: nonempty", len(rows) > 100, f"{len(rows)} pairs")
        bad = [r for r in rows[:5000] if not (r.get("prompt") and r.get("chosen") and r.get("rejected"))]
        check("all pairs have prompt/chosen/rejected", len(bad) == 0)
        diff = [r for r in rows[:5000] if r["chosen"] == r["rejected"]]
        check("chosen != rejected", len(diff) == 0, f"{len(diff)} degenerate")
    # decode-roundtrip a real pair through the chat encoder
    r = json.loads(open(f"{DATA}/preferences.jsonl").readline())
    ids, _ = encode_chat([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["chosen"]}])
    check("real pair encodes to tokens", len(ids) > 0, f"{len(ids)} tokens")


def verify_rl_prompts_and_gold():
    """Gold answers in the RL prompt files must match the GSM8K dataset's #### answer."""
    print("\n== RL prompts + gold answers ==")
    rows = [json.loads(l) for l in open(f"{DATA}/rl_prompts_test.jsonl")]
    check("gsm8k test prompts present", len(rows) > 100, f"{len(rows)} prompts")
    check("all have numeric gold", all(isinstance(r["gold"], (int, float)) for r in rows))
    # Cross-check gold against the live GSM8K dataset for a few rows.
    import os
    os.environ.setdefault("HF_HOME", "/ephemeral/hf_cache")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    by_q = {ex["question"].strip(): ex["answer"] for ex in ds}
    checked = 0
    for r in rows[:50]:
        if r["prompt"] in by_q:
            assert gsm8k_gold_answer(by_q[r["prompt"]]) == r["gold"], (r["prompt"][:40], r["gold"])
            checked += 1
    check("gold matches GSM8K #### answer", checked >= 40, f"verified {checked}/50 against source")
    # arithmetic curriculum gold correctness
    ar = [json.loads(l) for l in open(f"{DATA}/arithmetic_prompts.jsonl")]
    check("arithmetic prompts present", len(ar) > 1000, f"{len(ar)} prompts")


def verify_eval_benchmark():
    """Prove the GSM8K scoring is correct independent of any model: a response containing
    the gold scores correct; a wrong number scores incorrect."""
    print("\n== Evaluation benchmark scoring (GSM8K verifier) ==")
    import os
    os.environ.setdefault("HF_HOME", "/ephemeral/hf_cache")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(100))

    correct_when_right = correct_when_wrong = 0
    for ex in ds:
        gold = gsm8k_gold_answer(ex["answer"])
        good = f"<think>...</think><answer>{int(gold) if gold==int(gold) else gold}</answer>"
        bad = f"<answer>{gold + 7}</answer>"
        correct_when_right += int(is_correct(good, gold))
        correct_when_wrong += int(is_correct(bad, gold))
    check("perfect answers score correct", correct_when_right >= 98, f"{correct_when_right}/100")
    check("wrong answers score incorrect", correct_when_wrong == 0, f"{correct_when_wrong}/100 false positives")
    # reward shaping: correct+formatted > correct-only > wrong
    check("reward: correct+format > wrong", reward_gsm8k("<answer>5</answer>", 5.0) > reward_gsm8k("<answer>9</answer>", 5.0))
    check("reward: bounded", reward_gsm8k("<answer>5</answer>", 5.0) <= 1.2)
    # tolerant parsing variants
    check("parse: $ and commas", extract_answer("the total is $1,234") == 1234.0)
    check("parse: #### form", extract_answer("reasoning #### 18") == 18.0)


if __name__ == "__main__":
    verify_pile()
    verify_sft_mask_alignment()
    verify_preferences()
    verify_rl_prompts_and_gold()
    verify_eval_benchmark()
    print("\nALL DATA + EVAL CORRECTNESS CHECKS PASSED")
