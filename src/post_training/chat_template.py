"""
Chat formatting + loss masking for post-training.

The repo's tokenizer is tiktoken ``r50k_base``, whose ONLY special token is
``<|endoftext|>`` (id 50256). We therefore cannot register new special tokens for
chat roles. Instead we use plain-text role markers that simply tokenize as ordinary
multi-token strings -- the model learns them during SFT just like any other text.
``<|endoftext|>`` (EOT) is reused as the turn/sequence terminator and the single
generation stop token.

Conversation format (one user turn -> one assistant turn)::

    <|user|>
    {user content}<|endoftext|><|assistant|>
    {assistant content}<|endoftext|>

For reasoning / RLVR tasks the assistant content carries the structure itself::

    <think>step by step ...</think><answer>42</answer>

which needs no special handling here -- it is just the assistant message body.

Two outputs matter for training:
- ``ids``       : the full token id sequence.
- ``loss_mask`` : 1 on tokens the model should be trained to PRODUCE (the assistant
                  content plus its terminating EOT), 0 on prompt / role-marker tokens.
                  This is exactly the SFT prompt-masking mechanism and the RL
                  ``response_mask`` for generated positions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import tiktoken

# tiktoken r50k_base end-of-text id; the only true special token and our stop token.
EOT_ID = 50256

# Plain-text role markers (ordinary tokens, NOT registered specials).
USER_HEADER = "<|user|>\n"
ASSISTANT_HEADER = "<|assistant|>\n"
SYSTEM_HEADER = "<|system|>\n"

# Reasoning structure markers (also ordinary tokens). Exposed so reward parsing and
# data generation share a single source of truth.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"


@lru_cache(maxsize=1)
def get_tokenizer() -> "tiktoken.Encoding":
    """Return the shared r50k_base encoder (cached so we build it once)."""
    return tiktoken.get_encoding("r50k_base")


def _encode_ordinary(text: str) -> list[int]:
    """Encode text as ordinary tokens (no special-token handling)."""
    return get_tokenizer().encode_ordinary(text)


def _header_for(role: str) -> str:
    if role == "user":
        return USER_HEADER
    if role == "assistant":
        return ASSISTANT_HEADER
    if role == "system":
        return SYSTEM_HEADER
    raise ValueError(f"Unknown chat role: {role!r}")


def render_chat(messages: Iterable[dict], add_generation_prompt: bool = False) -> str:
    """
    Render a list of ``{"role", "content"}`` messages to the plain-text chat format.

    This is purely for display/debugging; tokenization goes through :func:`encode_chat`
    so that loss masks stay exactly aligned with token boundaries.

    Args:
        messages: iterable of ``{"role": "user"|"assistant"|"system", "content": str}``.
        add_generation_prompt: if True, append a trailing ``<|assistant|>\\n`` so the
            model is cued to generate (used at inference / rollout time).
    """
    parts: list[str] = []
    for m in messages:
        parts.append(_header_for(m["role"]))
        parts.append(m["content"])
        parts.append("<|endoftext|>")
    if add_generation_prompt:
        parts.append(ASSISTANT_HEADER)
    return "".join(parts)


def encode_chat(
    messages: Iterable[dict],
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Tokenize a conversation and build an aligned per-token loss mask.

    The mask is 1 over assistant content (and its terminating EOT) and 0 everywhere
    else (role markers, user/system content). When ``add_generation_prompt`` is True
    the conversation ends with an ``<|assistant|>\\n`` header and no completion, so the
    returned mask is all zeros -- this is the prompt form used for rollouts.

    Returns:
        (ids, loss_mask) as equal-length python lists of ints.
    """
    ids: list[int] = []
    mask: list[int] = []

    for m in messages:
        role = m["role"]
        # Role header: always masked out (the model should not be trained to emit it,
        # and at rollout time it is part of the fixed prompt).
        header_ids = _encode_ordinary(_header_for(role))
        ids.extend(header_ids)
        mask.extend([0] * len(header_ids))

        content_ids = _encode_ordinary(m["content"])
        is_completion = role == "assistant"
        ids.extend(content_ids)
        mask.extend([1 if is_completion else 0] * len(content_ids))

        # Turn terminator. Trained on (mask=1) only when it closes an assistant turn,
        # so the model learns to stop.
        ids.append(EOT_ID)
        mask.append(1 if is_completion else 0)

    if add_generation_prompt:
        header_ids = _encode_ordinary(ASSISTANT_HEADER)
        ids.extend(header_ids)
        mask.extend([0] * len(header_ids))

    return ids, mask


def encode_prompt(messages: Iterable[dict]) -> list[int]:
    """Token ids for the prompt form (ends in the assistant header, ready to generate)."""
    ids, _ = encode_chat(messages, add_generation_prompt=True)
    return ids


def decode(ids: Iterable[int]) -> str:
    """
    Decode token ids back to text (for eval / verifier parsing / sample dumps).

    Defensive: drops the EOT terminator and any id >= EOT (50256), since the model's vocab
    is padded to 50304 but r50k_base can only decode ordinary tokens 0..50255. An
    under-trained model can emit those padding ids, which would otherwise crash decoding.
    """
    clean = [t for t in ids if 0 <= t < EOT_ID]
    return get_tokenizer().decode(clean)
