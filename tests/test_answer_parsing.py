"""
The GSM8K answer checker reads the number a person would read.

The RL stages are rewarded by this checker and evaluation is scored by it, so a parse that
marks a wrong answer correct is a reward the policy can learn to exploit.
"""

from __future__ import annotations

import pytest

from src.post_training.rewards import extract_answer, is_correct


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<answer>42</answer>", 42.0),
        ("<answer>.5</answer>", 0.5),                      # used to lose its point and read 5
        ("<answer>-.25</answer>", -0.25),
        ("<answer>5 + 3 = 8</answer>", 8.0),               # the result, not the first number
        ("<answer>x = 12 - 4 = 8</answer>", 8.0),
        ("<answer>$1,234.50</answer>", 1234.5),
        ("<answer>-7</answer>", -7.0),
        ("<answer>-$5</answer>", -5.0),
        ("<answer>8 apples</answer>", 8.0),
        ("It takes 10-15 minutes", 15.0),                  # used to read -15
        ("<answer>COVID-19</answer>", 19.0),               # used to read -19
        ("The temperature fell to -3 degrees", -3.0),
        ("the total is 1,234 dollars", 1234.0),
        ("pi is about 3.14.", 3.14),
        ("reasoning #### 18", 18.0),
        ("no number at all", None),
    ],
)
def test_extract_answer(text: str, expected: float | None) -> None:
    assert extract_answer(text) == expected


def test_wrong_answers_are_not_marked_correct() -> None:
    assert not is_correct("<answer>.5</answer>", 5.0)
    assert is_correct("<answer>.5</answer>", 0.5)
    assert not is_correct("<answer>COVID-19</answer>", -19.0)
    assert not is_correct("<answer>5 + 3 = 8</answer>", 5.0)
    assert is_correct("<answer>5 + 3 = 8</answer>", 8.0)
    assert is_correct("It takes 10-15 minutes", 15.0)
