#!/usr/bin/env python3
"""
train.py - Train a Transformer Language Model from Scratch (Beginner-friendly CPU / GPU CLI)

This script is part of the CPU/Low-Hardware Student Mode. It allows students to pretrain
a custom-sized Transformer model using one of several presets (tiny, student, small, gpu)
or custom parameters, entirely on CPU or GPU.

Presets Overview:
  - tiny:    ~100K-300K parameters. Perfect first experiment for a normal laptop (CPU).
  - student: ~1M-2M parameters. Main educational configuration, runs easily on 4-8 GB RAM.
  - small:   ~5M-6M parameters. Runs on modern CPUs or small GPUs.
  - gpu:     The larger/legacy configuration (needs a dedicated GPU).

Usage:
  # Quick CPU smoke test on tiny configuration (takes seconds)
  python train.py --preset tiny --device cpu --steps 20

  # Train the main student model on CPU
  python train.py --preset student --device cpu
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path if not already there
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator

# Define preset configurations
PRESETS = {
    "tiny": {
        "n_blocks": 2,
        "n_head": 2,
        "n_embed": 64,
        "context_length": 128,
        "batch_size": 4,
        "lr": 1e-3,
        "train_steps": 1000,
        "eval_steps": 100,
        "eval_iters": 20,
        "checkpoint_steps": 500,
        "train_path": "data/student/train.h5",
        "dev_path": "data/student/val.h5",
        "out_path": "checkpoints/tiny.pt",
    },
    "student": {
        "n_blocks": 4,
        "n_head": 4,
        "n_embed": 128,
        "context_length": 128,
        "batch_size": 4,
        "lr": 5e-4,
        "train_steps": 2000,
        "eval_steps": 200,
        "eval_iters": 20,
        "checkpoint_steps": 1000,
        "train_path": "data/student/train.h5",
        "dev_path": "data/student/val.h5",
        "out_path": "checkpoints/student.pt",
    },
    "small": {
        "n_blocks": 4,
        "n_head": 8,
        "n_embed": 256,
        "context_length": 256,
        "batch_size": 8,
        "lr": 5e-4,
        "train_steps": 5000,
        "eval_steps": 500,
        "eval_iters": 50,
        "checkpoint_steps": 2500,
        "train_path": "data/student/train.h5",
        "dev_path": "data/student/val.h5",
        "out_path": "checkpoints/small.pt",
    },
    "gpu": {
        "n_blocks": 64,
        "n_head": 16,
        "n_embed": 2048,
        "context_length": 512,
        "batch_size": 32,
        "lr": 5e-4,
        "train_steps": 200000,
        "eval_steps": 1000,
        "eval_iters": 250,
        "checkpoint_steps": 5000,
        "train_path": "data/train/pile_train.h5",
        "dev_path": "data/val/pile_dev.h5",
        "out_path": "models/transformer_B.pt",
    }
}

def print_hardware_diagnostics(device: str):
    """Prints a friendly breakdown of the CPU/GPU environment."""
    print("=" * 60)
    print("                HARDWARE & RUNTIME DIAGNOSTICS")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device:     {torch.cuda.get_device_name(0)}")
        print(f"CUDA VRAM:       {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")

    # Core CPU diagnostics
    cpu_cores = os.cpu_count() or 1
    print(f"CPU Core Count:  {cpu_cores}")

    # Try retrieving RAM size cleanly without external dependencies
    try:
        ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        ram_gib = ram_bytes / (1024**3)
        print(f"System RAM:      {ram_gib:.2f} GiB")
    except Exception:
        print("System RAM:      N/A")

    print(f"Target Device:   {device.upper()}")

    if device == "cpu":
        print("\n[NOTE] CPU Training is active. This mode is explicitly intended for")
        print("learning, debugging, and experimenting with tiny models (1M-5M params).")
        print("To make training faster, we will optimize PyTorch thread settings.")
        # Optimizing thread settings for CPU training
        torch.set_num_threads(cpu_cores)
        print(f"PyTorch intra-op threads set to CPU core count: {torch.get_num_threads()}")
    print("=" * 60)

@torch.no_grad()
def evaluate_loss(model: Transformer, train_path: str, dev_path: str, batch_size: int, context_length: int, eval_iters: int, device: str) -> dict:
    """Evaluates the loss on both training and validation splits."""
    model.eval()
    out = {}
    for split, path in [('train', train_path), ('dev', dev_path)]:
        if not os.path.exists(path):
            out[split] = float('nan')
            continue

        # Create fresh batch iterator for evaluation
        eval_iter = get_batch_iterator(path, batch_size, context_length, device=device)
        losses = []
        for _ in range(eval_iters):
            try:
                xb, yb = next(eval_iter)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            except StopIteration:
                break
        out[split] = float(np.mean(losses)) if losses else float('nan')
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer Language Model from scratch (CPU/GPU-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--preset",
        choices=["tiny", "student", "small", "gpu"],
        default="tiny",
        help="Select a model preset size. 'tiny' is perfect for a CPU laptop; 'student' is the main educational config."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to train on. 'auto' selects cuda if available, otherwise cpu."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total number of training steps. Great for quick checks (e.g. --steps 20)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override initial learning rate."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Override context sequence length."
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Override path to the tokenized HDF5 training dataset."
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Override path to the tokenized HDF5 validation dataset."
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Override destination path for the saved model checkpoint (.pt)."
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a temporary checkpoint file every N steps."
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Run evaluation on train/dev sets every N steps."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume training from."
    )

    args = parser.parse_args()

    # 1. Resolve preset defaults and CLI overrides
    preset_cfg = dict(PRESETS[args.preset])

    # Overrides
    if args.steps is not None:
        preset_cfg["train_steps"] = args.steps
    if args.lr is not None:
        preset_cfg["lr"] = args.lr
    if args.batch_size is not None:
        preset_cfg["batch_size"] = args.batch_size
    if args.context_length is not None:
        preset_cfg["context_length"] = args.context_length
    if args.train_path is not None:
        preset_cfg["train_path"] = args.train_path
    if args.val_path is not None:
        preset_cfg["val_path"] = args.val_path
    if args.out_path is not None:
        preset_cfg["out_path"] = args.out_path
    if args.checkpoint_every is not None:
        preset_cfg["checkpoint_steps"] = args.checkpoint_every
    if args.eval_every is not None:
        preset_cfg["eval_steps"] = args.eval_every

    # 2. Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device = "cpu"

    # Print diagnostics
    print_hardware_diagnostics(device)

    # Validate dataset existence before starting
    if not os.path.exists(preset_cfg["train_path"]):
        print(f"\n[ERROR] Training dataset HDF5 file not found at: {preset_cfg['train_path']}")
        print("Please run data preparation first, for example:")
        print("  python prepare_data.py --input data/student/train.txt --output data/student/train.h5")
        sys.exit(1)

    # 3. Instantiate the model
    vocab_size = preset_cfg.get("vocab_size", 50304)
    print("Initializing Transformer Model...")
    model = Transformer(
        n_head=preset_cfg["n_head"],
        n_embed=preset_cfg["n_embed"],
        context_length=preset_cfg["context_length"],
        vocab_size=vocab_size,
        N_BLOCKS=preset_cfg["n_blocks"]
    ).to(device)

    # Print model parameters and educational notes
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 60)
    print("                     MODEL SPECIFICATION")
    print("=" * 60)
    print(f"Preset Profile:   {args.preset.upper()}")
    print(f"Embedding Size:   {preset_cfg['n_embed']}")
    print(f"Attention Heads:  {preset_cfg['n_head']}")
    print(f"Model Layers:     {preset_cfg['n_blocks']}")
    print(f"Context Length:   {preset_cfg['context_length']}")
    print(f"Batch Size:       {preset_cfg['batch_size']}")
    print(f"Learning Rate:    {preset_cfg['lr']}")
    print(f"Total Parameters: {total_params:,}")
    print("\n*Note: Parameter counts are approximate because embeddings, layer normalization,")
    print("biases, and other architecture choices affect the final count.")
    print("=" * 60 + "\n")

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=preset_cfg["lr"])

    # Track loss history and completed steps
    start_step = 0
    losses = []

    # Resume from checkpoint if requested
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint.get('step', -1) + 1
            losses = checkpoint.get('losses', [])
            print(f"Resumed successfully at step {start_step}.")
        else:
            print(f"[ERROR] Checkpoint file for resume not found at: {args.resume}", file=sys.stderr)
            sys.exit(1)

    # Create batch iterator
    train_iter = get_batch_iterator(
        preset_cfg["train_path"],
        preset_cfg["batch_size"],
        preset_cfg["context_length"],
        device=device
    )

    tokens_per_step = preset_cfg["batch_size"] * preset_cfg["context_length"]
    running_loss = 0.0
    accum_loss_iters = 0

    print("Beginning training loop...")
    for step in range(start_step, preset_cfg["train_steps"]):
        step_start_time = time.perf_counter()

        try:
            xb, yb = next(train_iter)
        except StopIteration:
            print("Finished epoch / training data iterator exhausted.")
            break

        # Forward pass and loss calculation
        _, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        step_time = time.perf_counter() - step_start_time
        tokens_per_sec = tokens_per_step / step_time if step_time > 0 else float('inf')

        loss_val = loss.item()
        losses.append(loss_val)
        running_loss += loss_val
        accum_loss_iters += 1

        # Print progress periodically
        if (step + 1) % 10 == 0 or step == 0:
            avg_loss = running_loss / accum_loss_iters
            print(f"Step {step + 1:5d}/{preset_cfg['train_steps']} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Throughput: {tokens_per_sec:.1f} tokens/s | "
                  f"Step Time: {step_time:.3f}s")
            running_loss = 0.0
            accum_loss_iters = 0

        # Periodic Evaluation
        if (step + 1) % preset_cfg["eval_steps"] == 0:
            print("-" * 60)
            print(f"Evaluating model at step {step + 1}...")
            eval_start = time.perf_counter()
            eval_res = evaluate_loss(
                model=model,
                train_path=preset_cfg["train_path"],
                dev_path=preset_cfg["dev_path"],
                batch_size=preset_cfg["batch_size"],
                context_length=preset_cfg["context_length"],
                eval_iters=preset_cfg["eval_iters"],
                device=device
            )
            eval_time = time.perf_counter() - eval_start
            print(f"Evaluation finished in {eval_time:.2f}s | "
                  f"Train Loss: {eval_res['train']:.4f} | "
                  f"Dev Loss: {eval_res['dev']:.4f}")
            print("-" * 60)

        # Periodic Checkpoint Saving
        if preset_cfg["checkpoint_steps"] > 0 and (step + 1) % preset_cfg["checkpoint_steps"] == 0:
            os.makedirs(os.path.dirname(preset_cfg["out_path"]) or ".", exist_ok=True)
            chk_name = os.path.splitext(preset_cfg["out_path"])[0] + f"_step_{step+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'step': step,
                'config': preset_cfg
            }, chk_name)
            print(f"Saved periodic checkpoint: {chk_name}")

    # Final Save
    os.makedirs(os.path.dirname(preset_cfg["out_path"]) or ".", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'step': preset_cfg["train_steps"] - 1,
        'config': preset_cfg
    }, preset_cfg["out_path"])
    print("\n" + "=" * 60)
    print(f"Training successfully complete! Saved final model to: {preset_cfg['out_path']}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
