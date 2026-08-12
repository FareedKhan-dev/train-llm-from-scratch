#!/usr/bin/env python3
"""
generate.py - Generate text using a trained Transformer model (CPU/GPU-friendly).

This script is part of the CPU/Low-Hardware Student Mode. It loads any model checkpoint
(including tiny, student, small, or large gpu checkpoints) and generates text starting from
a user prompt.

It automatically infers model dimensions from the checkpoint, making it incredibly easy
to use without having to specify layers, heads, or embedding sizes on the command line.

Usage:
    python generate.py --checkpoint checkpoints/tiny.pt --prompt "Once upon a time" --tokens 100
"""

import argparse
import os
import sys
import torch
import tiktoken
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.transformer import Transformer

def load_checkpoint(path: str, device: str = "cpu"):
    """Loads checkpoint dictionary, supporting fallback for old/new PyTorch versions."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def infer_model_config(checkpoint: dict) -> dict:
    """
    Infers the model configuration from the checkpoint structure or model_state_dict.
    This guarantees that any trained checkpoint (tiny, student, small, gpu, legacy)
    can be loaded seamlessly without manual configuration.
    """
    # 1. Check if config dictionary was saved in checkpoint (as done by train.py)
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        return {
            "n_head": cfg.get("n_head"),
            "n_embed": cfg.get("n_embed"),
            "context_length": cfg.get("context_length"),
            "vocab_size": cfg.get("vocab_size", 50304),
            "n_blocks": cfg.get("n_blocks")
        }

    # 2. Infer configuration dynamically from model_state_dict shapes
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Infer n_embed and vocab_size from token_embed weights
    if 'token_embed.weight' in state_dict:
        vocab_size, n_embed = state_dict['token_embed.weight'].shape
    else:
        # Defaults if key not found
        vocab_size = 50304
        n_embed = 128

    # Infer context_length from position_embed weights
    if 'position_embed.weight' in state_dict:
        context_length = state_dict['position_embed.weight'].shape[0]
    else:
        context_length = 128

    # Count transformer blocks
    # Check block keys like: token_embed.weight, position_embed.weight, attn_blocks.0.ln1.weight, ...
    block_indices = set()
    for key in state_dict.keys():
        if key.startswith("attn_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    n_blocks = len(block_indices) if block_indices else 1

    # Infer n_head from attention heads keys
    # Check head keys like: attn_blocks.0.attn.heads.0.query.weight, heads.1...
    head_indices = set()
    for key in state_dict.keys():
        if "attn.heads." in key:
            parts = key.split("attn.heads.")[1].split(".")
            if len(parts) > 0 and parts[0].isdigit():
                head_indices.add(int(parts[0]))
    n_head = len(head_indices) if head_indices else 4

    return {
        "n_head": n_head,
        "n_embed": n_embed,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "n_blocks": n_blocks
    }

def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer checkpoint (CPU/GPU compatible).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to the saved model checkpoint file (.pt)."
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Once upon a time",
        help="Input prompt to start the text generation."
    )
    parser.add_argument(
        "--tokens", "-t",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run inference on. 'auto' selects cuda if available."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="r50k_base",
        help="tiktoken encoding pattern to use."
    )

    args = parser.parse_args()

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print(f"Loading checkpoint from: {args.checkpoint} on {device.upper()}...")
    try:
        checkpoint = load_checkpoint(args.checkpoint, device)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    # Infer configurations
    cfg = infer_model_config(checkpoint)
    print("\n" + "=" * 60)
    print("                 INFERRED MODEL SPECIFICATION")
    print("=" * 60)
    print(f"Layers (Blocks):   {cfg['n_blocks']}")
    print(f"Heads per Layer:   {cfg['n_head']}")
    print(f"Embedding Size:    {cfg['n_embed']}")
    print(f"Context Window:    {cfg['context_length']}")
    print(f"Vocabulary Size:   {cfg['vocab_size']}")
    print("=" * 60 + "\n")

    # Instantiate model
    model = Transformer(
        n_head=cfg["n_head"],
        n_embed=cfg["n_embed"],
        context_length=cfg["context_length"],
        vocab_size=cfg["vocab_size"],
        N_BLOCKS=cfg["n_blocks"]
    ).to(device)

    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Strip prefixes if any exist (e.g. DDP prefix '_orig_mod.')
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[len("_orig_mod."):]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict)
    model.eval()

    # Load tokenizer
    enc = tiktoken.get_encoding(args.tokenizer)

    # Encode prompt
    start_ids = enc.encode_ordinary(args.prompt)
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    print(f"Prompt: \"{args.prompt}\"")
    print(f"Generating up to {args.tokens} tokens...")

    # Generation
    with torch.no_grad():
        # Keep generation constrained by max context length
        generated_tokens = model.generate(context, max_new_tokens=args.tokens)[0].tolist()

    # Decode and print
    # Filter out any token IDs that are out of bounds for the tokenizer (e.g., from an untrained/partially-trained model)
    valid_tokens = [t for t in generated_tokens if t < enc.n_vocab]
    output_text = enc.decode(valid_tokens)
    print("\n" + "=" * 60)
    print("                      GENERATED TEXT")
    print("=" * 60)
    print(output_text)
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
