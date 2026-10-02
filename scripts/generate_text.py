"""
Generate text with a model trained by scripts/train_transformer.py.

Everything the model needs is read from the checkpoint: its size, its architecture
(classic or modern), its tokenizer (r50k_base or the BPE trained by prepare_tiny_data.py),
and the window it was trained on. So you only pass the checkpoint path:

    python scripts/generate_text.py --model_path models/tiny.pt
    python scripts/generate_text.py --model_path models/student.pt --input_text "The little dog" \
        --max_new_tokens 200 --temperature 0.7 --top_k 40 --num_samples 3

Very old checkpoints that stored only the weights fall back to the sizes in config/config.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the repo without installing

from config.config import default_config as config  # noqa: E402
from src.checkpoint import (  # noqa: E402
    load_checkpoint,
    load_model_weights,
    model_config_from_checkpoint,
    model_state_from_checkpoint,
)
from src.device import resolve_device  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.tokenizer import safe_decode, tokenizer_from_spec  # noqa: E402


def load_trained_model(model_path: str, device: str):
    """Rebuild the model, its tokenizer and its training window from a checkpoint."""
    checkpoint = load_checkpoint(model_path, map_location="cpu")
    model_cfg = {**config, **model_config_from_checkpoint(checkpoint)}
    model = build_model(model_cfg)
    load_model_weights(model, model_state_from_checkpoint(checkpoint), source=model_path)
    tokenizer = tokenizer_from_spec(model_cfg.get("tokenizer"))
    # The legacy trainer may train on windows shorter than the model's context (t_context_length).
    # Positions it never saw have untrained embeddings, so generation stays inside that window.
    window = min(model_cfg.get("t_context_length") or model_cfg["context_length"], model_cfg["context_length"])
    return model.to(device).eval(), tokenizer, window, model_cfg


def generate_text(model_path: str, input_text: str, max_new_tokens: int = 100, device: str = "auto",
                  temperature: float = 0.8, top_k: int | None = 50) -> str:
    """
    Generates text using a pre-trained Transformer model.

    Args:
        model_path (str): Path to the saved model checkpoint.
        input_text (str): The initial text to start generation from.
        max_new_tokens (int): The maximum number of new tokens to generate.
        device (str): "auto", "cuda", "mps" or "cpu".
        temperature (float): Below 1 is safer and more repetitive, above 1 more random.
        top_k (int, optional): Sample only from the k most likely next tokens.

    Returns:
        str: The prompt followed by the generated continuation.
    """
    device = resolve_device(device)
    model, tokenizer, window, _ = load_trained_model(model_path, device)
    start_ids = tokenizer.encode_ordinary(input_text)
    context = torch.tensor([start_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature,
                             top_k=top_k, context_window=window)
    return safe_decode(tokenizer, out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained Transformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint.")
    parser.add_argument("--input_text", type=str, default="Once upon a time", help="The text to continue.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Sample from the k most likely tokens (0 = all).")
    parser.add_argument("--num_samples", type=int, default=1, help="How many continuations to print.")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible samples.")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, tokenizer, window, model_cfg = load_trained_model(args.model_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.model_path}: {n_params:,} parameters, {model_cfg.get('arch', 'classic')} "
          f"architecture, window {window} tokens, on {device}")

    context = torch.tensor([tokenizer.encode_ordinary(args.input_text)], dtype=torch.long, device=device)
    for i in range(args.num_samples):
        with torch.no_grad():
            out = model.generate(context, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 top_k=args.top_k or None, context_window=window)
        print(f"\n--- sample {i + 1} ---\n{safe_decode(tokenizer, out[0].tolist())}")


if __name__ == "__main__":
    main()
