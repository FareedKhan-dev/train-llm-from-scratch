"""
Inference helpers shared by the chat CLI and the eval scripts.

Loads any stage checkpoint (base / sft / dpo / ppo / grpo) by reading the model
dimensions from the checkpoint's stored ``cfg`` (so you never have to repeat them), and
generates a reply either in chat-template form (for instruction-tuned models) or as raw
continuation (for the base model).
"""

from __future__ import annotations

import torch

from src.models.transformer import Transformer
from src.post_training.chat_template import EOT_ID, decode, encode_prompt, get_tokenizer
from src.post_training.evaluation import batched_generate


def load_model_from_ckpt(ckpt_path: str, device: str, overrides: dict | None = None) -> Transformer:
    """Build a :class:`Transformer` from a checkpoint's stored cfg and load its backbone
    weights (tolerates ``module.``/``transformer.`` prefixes and reward-head extras)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = {**(ck.get("cfg") or {}), **(overrides or {})}
    model = Transformer(
        n_head=cfg.get("n_head", 16), n_embed=cfg.get("n_embed", 1024),
        context_length=cfg.get("context_length", 1024), vocab_size=cfg.get("vocab_size", 50304),
        N_BLOCKS=cfg.get("n_blocks", 24),
    )
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    state = {k.removeprefix("module.").removeprefix("transformer."): v for k, v in state.items()}
    keys = set(model.state_dict().keys())
    model.load_state_dict({k: v for k, v in state.items() if k in keys}, strict=False)
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
