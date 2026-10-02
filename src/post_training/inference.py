"""
Inference helpers shared by the chat CLI and the eval scripts.

Loads any stage checkpoint (base / sft / dpo / ppo / grpo) by reading the model
dimensions from the checkpoint's stored ``cfg`` (so you never have to repeat them), and
generates a reply either in chat-template form (for instruction-tuned models) or as raw
continuation (for the base model).
"""

from __future__ import annotations

import torch

from src.checkpoint import (
    load_checkpoint,
    load_model_weights,
    model_config_from_checkpoint,
    model_state_from_checkpoint,
)
from src.models.factory import build_model
from src.models.transformer import Transformer
from src.post_training.chat_template import encode_prompt, get_tokenizer
from src.post_training.evaluation import batched_generate


def load_model_from_ckpt(ckpt_path: str, device: str, overrides: dict | None = None) -> Transformer:
    """
    Build a model from the settings stored in a checkpoint and load its weights.

    Works for every checkpoint in the repo: the legacy trainer (settings under ``config``),
    every post-training stage (settings under ``cfg``), and reward-model checkpoints, whose
    backbone keys start with ``transformer.``. DDP and torch.compile prefixes are stripped
    too, so a model trained with ``--compile true`` loads correctly (issue #36).
    """
    ck = load_checkpoint(ckpt_path, map_location="cpu")
    defaults = {"n_head": 16, "n_embed": 1024, "context_length": 1024, "vocab_size": 50304, "n_blocks": 24}
    cfg = {**defaults, **model_config_from_checkpoint(ck), **(overrides or {})}
    model = build_model(cfg)
    state = model_state_from_checkpoint(ck, extra_prefixes=("transformer.",))
    load_model_weights(model, state, source=ckpt_path)
    return model.to(device).eval()


@torch.no_grad()
def generate_reply(
    model,
    user_text: str,
    *,
    device: str,
    system: str | None = None,
    raw: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = 0.95,
    greedy: bool = False,
) -> str:
    """
    Generate a response to ``user_text``.

    - chat mode (default): wraps the prompt in the chat template (optionally with a
      ``system`` message) and returns the decoded assistant turn.
    - raw mode (``raw=True``): treats ``user_text`` as a prefix and returns the base
      model's continuation (no chat template) -- the right mode for the pretrained base.
    """
    if raw:
        ids = get_tokenizer().encode_ordinary(user_text)
        out = batched_generate(model, [ids], max_new_tokens, device=device, temperature=temperature,
                               top_k=top_k, top_p=top_p, greedy=greedy)
        return out[0]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    prompt_ids = encode_prompt(messages)
    out = batched_generate(model, [prompt_ids], max_new_tokens, device=device, temperature=temperature,
                           top_k=top_k, top_p=top_p, greedy=greedy)
    return out[0]
