"""The runtime shape checks from conftest.py are really on: wrong shapes fail at the call."""

from __future__ import annotations

import os

import pytest
import torch

from src.inference.sampling import sample_next_token
from src.models.modern import apply_rope, rope_cache

pytestmark = pytest.mark.skipif(os.environ.get("SHAPE_CHECKS") == "0", reason="shape checks turned off")


def test_a_shape_mismatch_is_caught_at_the_call() -> None:
    x = torch.randn(2, 4, 8)  # 4 positions
    cos, sin = rope_cache(8, 5)  # tables for 5 positions: "seq" does not match
    with pytest.raises(TypeError, match="seq"):
        apply_rope(x, cos, sin)


def test_a_wrong_dtype_is_caught_at_the_call() -> None:
    with pytest.raises(TypeError):
        sample_next_token(torch.ones(1, 5, dtype=torch.long))  # logits must be floats
