"""
Truncation of preference pairs (reward model and DPO data).

Background: before #41 each side was encoded on its own and cut to ``max_len``, so a long
prompt could push both answers out of the window. The pair then had identical tokens and
all-zero response masks, and DPO learned nothing from it. These tests pin the current rules
(see ``encode_preference_pair``).
"""

from __future__ import annotations

import random

from data_loader.preference_dataset import _collate, encode_preference_pair
from src.post_training import chat_template as ct


def _prompt_len(mask: list[int]) -> int:
    return mask.index(1) if 1 in mask else len(mask)


def test_short_pairs_are_encoded_exactly_like_a_full_conversation() -> None:
    (c_ids, c_mask), (r_ids, r_mask) = encode_preference_pair("What is 2+2?", "4", "5", max_len=128)
    full_ids, full_mask = ct.encode_chat(
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]
    )
    assert (c_ids, c_mask) == (full_ids, full_mask)
    assert r_ids[: _prompt_len(r_mask)] == c_ids[: _prompt_len(c_mask)]


def test_long_prompt_keeps_both_full_answers() -> None:
    prompt = "Explain every step in detail. " * 100
    (c_ids, c_mask), (r_ids, r_mask) = encode_preference_pair(prompt, "The answer is yes.", "The answer is no.", 48)
    assert len(c_ids) <= 48 and len(r_ids) <= 48
    assert sum(c_mask) == sum(ct.encode_chat([{"role": "assistant", "content": "The answer is yes."}])[1])
    assert c_ids[-1] == ct.EOT_ID and r_ids[-1] == ct.EOT_ID
    # the kept prompt is the END of the prompt, so it still finishes with the assistant header
    header = ct.encode_prompt([])
    assert c_ids[: _prompt_len(c_mask)][-len(header):] == header
    assert c_ids[: _prompt_len(c_mask)] == r_ids[: _prompt_len(r_mask)]


def test_long_prompt_and_long_answers_split_the_window() -> None:
    prompt = "Tell me about the printing press. " * 200
    chosen = "It spread knowledge fast. " * 200
    rejected = "It was invented in 1440. " * 200
    (c_ids, c_mask), (r_ids, r_mask) = encode_preference_pair(prompt, chosen, rejected, max_len=256)
    assert len(c_ids) == 256 and len(r_ids) == 256
    assert _prompt_len(c_mask) == 128, "the prompt keeps half of the window when both are long"
    assert sum(c_mask) == 128 and sum(r_mask) == 128
    assert c_ids != r_ids


def test_late_difference_is_kept_by_shortening_the_prompt() -> None:
    prompt = "Some long context. " * 100
    prefix = "Let me think about this carefully. " * 8          # ~64 shared answer tokens
    (c_ids, _), (r_ids, _) = encode_preference_pair(prompt, prefix + "Yes.", prefix + "No.", max_len=96)
    assert c_ids != r_ids, "the first differing token must survive truncation"


def test_indistinguishable_pairs_do_not_crash() -> None:
    same = "Sure! " * 300
    (c_ids, c_mask), (r_ids, r_mask) = encode_preference_pair("Hi", same + "yes", same + "no", max_len=64)
    assert len(c_ids) <= 64 and len(r_ids) <= 64
    assert c_ids == r_ids  # no signal left, but training keeps going (zero DPO gradient)


def test_random_pairs_always_fit_and_share_the_prompt() -> None:
    rng = random.Random(0)
    words = ["alpha", "beta", "gamma", "delta", "the", "a", "model", "token", "\n", "42", "!"]

    def text(n: int) -> str:
        return " ".join(rng.choice(words) for _ in range(n))

    for _ in range(200):
        max_len = rng.choice([16, 24, 64, 128])
        pair = encode_preference_pair(text(rng.randint(0, 120)), text(rng.randint(0, 120)),
                                      text(rng.randint(0, 120)), max_len)
        (c_ids, c_mask), (r_ids, r_mask) = pair
        assert len(c_ids) == len(c_mask) <= max_len
        assert len(r_ids) == len(r_mask) <= max_len
        assert c_ids[: _prompt_len(c_mask)] == r_ids[: _prompt_len(r_mask)]


def test_collate_pads_both_sides_to_one_length() -> None:
    rows = [
        {"prompt": "short", "chosen": "a", "rejected": "b c d e"},
        {"prompt": "a much longer prompt " * 20, "chosen": "yes", "rejected": "no"},
    ]
    batch = _collate(rows, max_len=64, device="cpu")
    assert batch["chosen_ids"].shape == batch["rejected_ids"].shape
    assert batch["chosen_mask"].sum(1).min() > 0 and batch["rejected_mask"].sum(1).min() > 0
    assert (batch["chosen_len"] <= 64).all() and (batch["rejected_len"] <= 64).all()
