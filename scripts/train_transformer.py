"""
Pretrain the original Transformer with plain Python settings (config/config.py).

This is the simplest training loop in the repo: one device, AdamW, a step decay of the
learning rate, periodic evaluation and checkpoints. Named presets make it easy to start
small, and every preset can be overridden from the command line.

On a laptop CPU (no GPU needed), after `python scripts/prepare_tiny_data.py`:
    python scripts/train_transformer.py --preset tiny
    python scripts/train_transformer.py --preset student
    python scripts/train_transformer.py --preset student --arch modern   # same size, 2026 architecture
On a GPU, with the Pile data from scripts/data_download.py + scripts/data_preprocess.py:
    python scripts/train_transformer.py --preset 13m
    python scripts/train_transformer.py                                  # the constants in config/config.py

The vocabulary size and tokenizer are read from the training file when it stores them (files
made by prepare_tiny_data.py do) and saved in the checkpoint, so scripts/generate_text.py
needs nothing but the checkpoint path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import default_config as config
from config.presets import PRESETS, apply_preset
from src.checkpoint import model_state_from_checkpoint
from src.device import DEVICE_CHOICES, configure_cpu_threads, resolve_device, sync_step
from src.models.factory import ARCHITECTURES, build_model
from src.models.transformer import Transformer
from src.tokenizer import DEFAULT_TOKENIZER, safe_decode, tokenizer_from_spec


# --- Runtime Diagnostics Helpers ---

def bytes_to_gib(num_bytes: int) -> float:
    """Convert a byte count to gibibytes for human-readable memory reports."""
    return num_bytes / (1024 ** 3)


def get_device_report(device: str) -> str:
    """
    Build a short report describing the runtime environment: PyTorch/CUDA
    versions and, when running on a GPU, its name, capability, and total VRAM.
    This makes it easy to collect comparable training reports across machines.
    """
    lines = [
        f"PyTorch version: {torch.__version__}",
        f"Configured device: {device}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"CUDA version: {torch.version.cuda}",
    ]

    if device.startswith('cuda') and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_vram_gib = bytes_to_gib(props.total_memory)
        lines.extend([
            f"GPU name: {torch.cuda.get_device_name(device_index)}",
            f"GPU capability: {props.major}.{props.minor}",
            f"Total VRAM: {total_vram_gib:.2f} GiB",
        ])
    else:
        lines.append("GPU name: N/A (running without CUDA)")

    return "\n".join(lines)


def get_peak_memory_report(device: str) -> str:
    """Report peak GPU memory (allocated/reserved) since the last reset, or N/A on CPU."""
    if device.startswith('cuda') and torch.cuda.is_available():
        peak_allocated = bytes_to_gib(torch.cuda.max_memory_allocated())
        peak_reserved = bytes_to_gib(torch.cuda.max_memory_reserved())
        return (
            f"Peak VRAM allocated: {peak_allocated:.2f} GiB | "
            f"Peak VRAM reserved: {peak_reserved:.2f} GiB"
        )
    return "Peak VRAM allocated: N/A | Peak VRAM reserved: N/A"


def estimate_memory_budget(num_params: int, device: str, use_amp: bool) -> str:
    """
    Print a rough training VRAM budget so users can predict OOM before launching.

    AdamW keeps fp32 weights + grads + two moment buffers (~16 bytes/param). This is an
    estimate of optimizer/parameter state only (activations depend on batch/context and
    are reduced a lot by gradient checkpointing). CUDA-only; returns N/A otherwise.
    """
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return "VRAM budget: N/A (no CUDA device)"
    # weights(4) + grad(4) + Adam m(4) + Adam v(4); AMP adds bf16/fp16 copies but keeps
    # the fp32 master state, so ~16 B/param is a reasonable floor either way.
    state_gib = bytes_to_gib(num_params * 16)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_gib = bytes_to_gib(props.total_memory)
    note = " (+ activations; reduce with --grad-checkpointing / --grad-accum)"
    return (
        f"VRAM budget: ~{state_gib:.2f} GiB params+optimizer state vs {total_gib:.2f} GiB "
        f"total on {torch.cuda.get_device_name()}{note}"
    )


# --- Data Helpers ---

def read_data_info(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """Tokenizer spec and vocab size stored in a token file (None for files that lack them)."""
    with h5py.File(path, "r") as f:
        attrs = f["tokens"].attrs
        spec = json.loads(attrs["tokenizer"]) if "tokenizer" in attrs else None
        vocab = int(attrs["vocab_size"]) if "vocab_size" in attrs else None
    return spec, vocab


def round_up(value: int, multiple: int = 64) -> int:
    """Pad the vocabulary to a multiple of 64: matrix multiplies run faster on those sizes."""
    return ((value + multiple - 1) // multiple) * multiple


@torch.no_grad()
def sample_text(model: torch.nn.Module, train_config: Dict[str, Any], prompt: str, n_tokens: int = 120) -> str:
    """Generate a short continuation of ``prompt`` with the training tokenizer."""
    tok = tokenizer_from_spec(train_config.get("tokenizer"))
    ids = torch.tensor([tok.encode_ordinary(prompt)], dtype=torch.long, device=train_config["device"])
    model.eval()
    out = model.generate(ids, max_new_tokens=n_tokens, temperature=0.8, top_k=50,
                         context_window=train_config["t_context_length"])
    model.train()
    return safe_decode(tok, out[0].tolist())


# --- Checkpoint Helpers ---

CHECKPOINT_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


def load_checkpoint_file(path: str, device: str) -> Dict[str, Any]:
    """Load a checkpoint while supporting both newer and older PyTorch versions."""
    try:
        return torch.load(path, map_location=torch.device(device), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device(device))


def default_checkpoint_dir(out_path: str) -> str:
    """Return a checkpoint directory tied to the configured final model path."""
    model_path = Path(out_path)
    return str(model_path.with_suffix("")) + "_checkpoints"


def checkpoint_path(checkpoint_dir: str, step: int) -> str:
    """Build a stable checkpoint path for the last completed training step."""
    return os.path.join(checkpoint_dir, f"checkpoint_step_{step:08d}.pt")


def checkpoint_step(path: str) -> int:
    """Extract the step number from a checkpoint filename."""
    match = CHECKPOINT_RE.search(os.path.basename(path))
    if not match:
        return -1
    return int(match.group(1))


def list_checkpoints(checkpoint_dir: str) -> List[str]:
    """Return periodic checkpoints sorted by training step."""
    if not os.path.isdir(checkpoint_dir):
        return []
    paths = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if CHECKPOINT_RE.search(name)
    ]
    return sorted(paths, key=checkpoint_step)


def resolve_resume_path(resume: Optional[str], checkpoint_dir: str) -> Optional[str]:
    """
    Resolve a resume argument.

    ``--resume`` with no value uses the latest periodic checkpoint in checkpoint_dir.
    ``--resume path/to/file.pt`` loads that exact checkpoint.
    """
    if resume is None:
        return None
    if resume == "latest":
        checkpoints = list_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return checkpoints[-1]
    return resume


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Read the learning rate from the first optimizer parameter group."""
    return float(optimizer.param_groups[0]["lr"])


def lr_for_step(train_config: Dict[str, Any], step: int) -> float:
    """Return the learning rate that should be active at a given step."""
    if step > train_config['t_lr_decay_step']:
        return float(train_config['t_lr_decayed'])
    return float(train_config['t_lr'])


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set all optimizer parameter groups to the same learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_training_checkpoint(
    path: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    train_config: Dict[str, Any],
    losses: List[float],
    *,
    step: int,
    train_loss: Optional[float] = None,
    dev_loss: Optional[float] = None,
    is_final: bool = False,
) -> None:
    """
    Save model, optimizer, loss history, and LR schedule metadata.

    ``step`` is the last completed zero-based training step, so resume starts at
    ``step + 1``.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'step': step,
        'last_completed_step': step,
        'steps': step + 1,
        'is_final': is_final,
        'config': dict(train_config),
        'device': train_config['device'],
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'lr_state': {
            'current_lr': current_lr(optimizer),
            'initial_lr': train_config['t_lr'],
            'decayed_lr': train_config['t_lr_decayed'],
            'decay_step': train_config['t_lr_decay_step'],
        },
    }
    target_dir = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(
        dir=target_dir,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def restore_training_checkpoint(
    path: str,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    train_config: Dict[str, Any],
    device: str,
) -> Tuple[int, List[float]]:
    """
    Restore model/optimizer state and return ``(next_step, losses)``.

    Older checkpoints did not have ``last_completed_step``. For those, ``steps``
    is treated as the number of completed optimizer steps.
    """
    checkpoint = load_checkpoint_file(path, device)
    # Strip DDP / torch.compile key prefixes so checkpoints saved from a wrapped model load too.
    model.load_state_dict(model_state_from_checkpoint(checkpoint))

    optimizer_state = checkpoint.get('optimizer_state_dict')
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    if 'last_completed_step' in checkpoint:
        last_completed_step = int(checkpoint['last_completed_step'])
        next_step = last_completed_step + 1
    else:
        next_step = int(checkpoint.get('steps', 0))
        last_completed_step = next_step - 1

    if not optimizer_state:
        set_optimizer_lr(optimizer, lr_for_step(train_config, next_step))

    losses = [float(loss) for loss in checkpoint.get('losses', [])]
    print(
        f"Resumed from {path}. "
        f"Last completed step: {last_completed_step}. Next step: {next_step}."
    )
    return next_step, losses


def prune_old_checkpoints(checkpoint_dir: str, keep_last: int) -> None:
    """Keep only the most recent N periodic checkpoints when requested."""
    if keep_last <= 0:
        return
    checkpoints = list_checkpoints(checkpoint_dir)
    for old_path in checkpoints[:-keep_last]:
        os.remove(old_path)


def unique_output_path(out_path: str) -> str:
    """Avoid overwriting an existing final model checkpoint."""
    modified_model_out_path = out_path
    save_tries = 0
    while os.path.exists(modified_model_out_path):
        save_tries += 1
        model_out_name = os.path.splitext(out_path)[0]
        modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"
    return modified_model_out_path


def as_float(value: Any) -> Optional[float]:
    """Convert scalar tensors/numbers to plain floats for checkpoint metadata."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


# --- Training / Evaluation ---

@torch.no_grad()
def estimate_loss(model: Transformer, train_config: Dict[str, Any], steps: int) -> Dict[str, float]:
    """
    Evaluate the model on training and development datasets and calculate average loss.

    Args:
        model (Transformer): The model being trained.
        train_config (dict): Training configuration values.
        steps (int): Number of steps to evaluate.

    Returns:
        dict: Dictionary containing average losses for 'train' and 'dev' splits.
    """
    out = {}
    model.eval()  # Set the model to evaluation mode.
    from data_loader.data_loader import get_batch_iterator

    for split in ['train', 'dev']:
        # Select the appropriate data path for the current split.
        data_path = train_config['train_path'] if split == 'train' else train_config['dev_path']

        # Create a batch iterator for evaluation.
        batch_iterator_eval = get_batch_iterator(
            data_path,
            train_config['t_batch_size'],
            train_config['t_context_length'],
            device=train_config['device'],
        )

        # Track loss values for each evaluation step.
        losses_eval = []
        for _ in range(steps):
            try:
                # Fetch a batch and calculate the loss.
                xb, yb = next(batch_iterator_eval)
                _, loss = model(xb, yb)
                losses_eval.append(float(loss.item()))
            except StopIteration:
                # Handle the case where the data iterator ends early.
                print(f"Warning: Iterator for {split} ended early.")
                break

        # Compute the mean loss for the current split.
        out[split] = float(np.mean(losses_eval)) if losses_eval else float("nan")

    model.train()  # Restore the model to training mode.
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Transformer model from scratch.",
        epilog="Presets: " + ", ".join(sorted(PRESETS)) + " (see config/presets.py).",
    )
    # --- Model size, data and device (all optional; default = the values in config/config.py) ---
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="Named model/training size. tiny/student/small run on a laptop CPU.")
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default=None,
                        help="classic = the original Transformer, modern = RoPE/RMSNorm/SwiGLU/GQA.")
    parser.add_argument("--device", choices=list(DEVICE_CHOICES), default=None,
                        help="auto picks CUDA, then Apple MPS, then the CPU.")
    parser.add_argument("--train-path", default=None, help="Tokenized training file (.h5).")
    parser.add_argument("--dev-path", default=None, help="Tokenized validation file (.h5).")
    parser.add_argument("--out-path", default=None, help="Where to save the final model (.pt).")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Sequences per step.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (decayed 10x for the last 20%% of steps).")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate every N steps.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs.")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads (default: one per physical core). Fewer can be faster on hybrid CPUs.")
    parser.add_argument("--sample", default=None,
                        help="Prompt to continue after training (the CPU presets use 'Once upon a time').")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help=(
            "Resume from a checkpoint path. Pass --resume with no value, or "
            "--resume latest, to use the newest checkpoint in the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a periodic checkpoint every N completed steps. 0 disables periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for periodic checkpoints. Defaults to a directory next to t_out_path.",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help="Keep only the most recent N periodic checkpoints. 0 keeps all.",
    )
    # --- Memory-optimisation flags (opt-in; all default to the config values, which are OFF) ---
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=None,
        help="Enable bf16/fp16 mixed-precision autocast (CUDA only; ignored on CPU).",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["bf16", "fp16"],
        default=None,
        help="Autocast dtype when --amp is set: bf16 (default, no GradScaler) or fp16.",
    )
    parser.add_argument(
        "--grad-checkpointing",
        dest="grad_checkpointing",
        action="store_true",
        default=None,
        help="Recompute transformer-block activations in backward to save VRAM.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Accumulate gradients over N micro-batches per optimizer step (effective batch xN).",
    )
    parser.add_argument(
        "--report-memory",
        dest="report_memory",
        action="store_true",
        default=None,
        help="Print a rough VRAM budget (params + optimizer state) before training (CUDA only).",
    )
    return parser.parse_args()


def resolve_train_config(args: argparse.Namespace) -> Dict[str, Any]:
    """config/config.py, then the preset, then command-line flags (later wins)."""
    train_config = dict(config)
    if args.preset:
        train_config = apply_preset(train_config, args.preset)
    flags = {
        "arch": args.arch, "train_path": args.train_path, "dev_path": args.dev_path,
        "t_out_path": args.out_path, "t_batch_size": args.batch_size, "t_lr": args.lr,
        "t_eval_steps": args.eval_every, "sample_prompt": args.sample,
    }
    train_config.update({k: v for k, v in flags.items() if v is not None})
    if args.lr is not None:
        train_config["t_lr_decayed"] = args.lr / 10
    if args.steps is not None:
        train_config["t_train_steps"] = args.steps
        train_config["t_lr_decay_step"] = int(args.steps * 0.8)
    train_config["device"] = resolve_device(args.device or train_config.get("device", "auto"))

    if not os.path.exists(train_config["train_path"]):
        laptop = args.preset in ("tiny", "student", "small") or "tiny" in train_config["train_path"]
        hint = ("python scripts/prepare_tiny_data.py" if laptop
                else "python scripts/data_download.py && python scripts/data_preprocess.py")
        raise SystemExit(f"Training data not found at {train_config['train_path']}. Create it first:\n    {hint}")
    spec, vocab = read_data_info(train_config["train_path"])
    train_config["tokenizer"] = spec or DEFAULT_TOKENIZER
    if vocab is not None and round_up(vocab) != train_config["vocab_size"]:
        print(f"Vocabulary from the data file: {vocab} tokens -> vocab_size {round_up(vocab)}")
        train_config["vocab_size"] = round_up(vocab)
    return train_config


def main() -> None:
    args = parse_args()
    train_config = resolve_train_config(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    if train_config["device"] == "cpu":
        configure_cpu_threads(args.threads)
        if not args.preset and train_config["n_embed"] >= 2048:
            print("Note: config/config.py describes a ~3B parameter model, far too big for a CPU. "
                  "Try --preset tiny (see config/presets.py).")
    checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else train_config.get('t_checkpoint_steps', 0)
    )
    keep_last = (
        args.keep_last
        if args.keep_last is not None
        else train_config.get('t_keep_last_checkpoints', 0)
    )
    checkpoint_dir = (
        args.checkpoint_dir
        or train_config.get('t_checkpoint_dir')
        or default_checkpoint_dir(train_config['t_out_path'])
    )

    # --- Resolve memory-optimisation options (CLI overrides config; all default OFF) ---
    use_amp = args.amp if args.amp is not None else bool(train_config.get('use_amp', False))
    amp_dtype_name = args.amp_dtype or train_config.get('amp_dtype', 'bf16')
    use_grad_ckpt = (
        args.grad_checkpointing if args.grad_checkpointing is not None
        else bool(train_config.get('use_gradient_checkpointing', False))
    )
    grad_accum = max(1, args.grad_accum if args.grad_accum is not None
                     else int(train_config.get('grad_accum_steps', 1)))
    report_memory = (
        args.report_memory if args.report_memory is not None
        else bool(train_config.get('report_memory_budget', False))
    )

    device_is_cuda = train_config['device'].startswith('cuda') and torch.cuda.is_available()
    if use_amp and not device_is_cuda:
        print("[mem-opt] --amp requested but no CUDA device available; disabling AMP.")
        use_amp = False
    amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type='cuda', dtype=amp_dtype)
        return contextlib.nullcontext()

    # GradScaler is only needed for fp16; bf16 has enough range. A disabled scaler is a no-op,
    # so the scale/unscale_/step/update calls below work unchanged for bf16 and CPU.
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # --- Initialize the Model and Print Parameters ---

    # Print runtime/device diagnostics and reset GPU peak-memory stats before training.
    print(get_device_report(train_config['device']))
    if train_config['device'].startswith('cuda') and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # classic = the Transformer in src/models/transformer.py, modern = src/models/modern.
    model = build_model(train_config).to(train_config['device'])

    # Print the total number of parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params:,} "
          f"({train_config.get('arch', 'classic')} architecture, vocab {train_config['vocab_size']})")

    # Apply opt-in memory optimisations.
    model.gradient_checkpointing = use_grad_ckpt
    if report_memory:
        print(estimate_memory_budget(total_params, train_config['device'], use_amp))
    if use_amp or use_grad_ckpt or grad_accum > 1:
        print(
            f"[mem-opt] amp={use_amp}"
            f"{'(' + amp_dtype_name + ')' if use_amp else ''} "
            f"grad_checkpointing={use_grad_ckpt} grad_accum={grad_accum}"
        )

    # --- Optimizer Setup and Loss Tracking ---

    # Set up the AdamW optimizer with the specified learning rate.
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['t_lr'])

    # List to track loss values during training.
    losses: List[float] = []
    start_step = 0
    last_completed_step = -1
    resume_path = resolve_resume_path(args.resume, checkpoint_dir)
    if resume_path is not None:
        start_step, losses = restore_training_checkpoint(
            resume_path,
            model,
            optimizer,
            train_config,
            train_config['device'],
        )
        last_completed_step = start_step - 1

    # Define a window size for averaging recent losses in the training loop.
    avg_window = 64

    # --- Training Loop ---

    from data_loader.data_loader import get_batch_iterator

    # Create a batch iterator for the training data.
    batch_iterator = get_batch_iterator(
        train_config['train_path'],
        train_config['t_batch_size'],
        train_config['t_context_length'],
        device=train_config['device'],
    )

    # Number of tokens processed per optimizer step (batch * context * grad_accum), for throughput.
    tokens_per_step = train_config['t_batch_size'] * train_config['t_context_length'] * grad_accum
    last_eval_time = time.perf_counter()
    latest_train_loss = None
    latest_dev_loss = None

    # Create a progress bar to monitor training progress.
    pbar = tqdm(range(start_step, train_config['t_train_steps']))
    for step in pbar:
        try:
            # Start the step timer.
            step_start_time = time.perf_counter()

            # Accumulate gradients over `grad_accum` micro-batches (==1 => original behaviour).
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(grad_accum):
                xb, yb = next(batch_iterator)
                with autocast_ctx():
                    _, loss = model(xb, yb)
                    # Scale so the accumulated gradient equals the full-batch mean gradient.
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
                step_loss += float(loss.item())

            # Record the (accumulated) loss for tracking.
            losses.append(step_loss)
            pbar.set_description(f"Train loss: {np.mean(losses[-avg_window:]):.4f}")

            # Clip gradients to prevent exploding gradients (unscale first for fp16 AMP).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            sync_step(train_config['device'])  # runs the queued graph on TPUs, a no-op elsewhere
            last_completed_step = step

            # Measure step time and instantaneous throughput for diagnostics.
            step_time = time.perf_counter() - step_start_time
            tokens_per_second = tokens_per_step / step_time if step_time > 0 else float('inf')

            # Periodically evaluate the model on training and development data.
            if step % train_config['t_eval_steps'] == 0:
                evaluation_losses = estimate_loss(model, train_config, train_config['t_eval_iters'])
                latest_train_loss = evaluation_losses['train']
                latest_dev_loss = evaluation_losses['dev']
                # Report timing/throughput for the most recent step and wall-time since last eval.
                now = time.perf_counter()
                elapsed_since_eval = now - last_eval_time
                last_eval_time = now
                print(
                    f"Step: {step}, Train loss: {latest_train_loss:.4f}, Dev loss: {latest_dev_loss:.4f}, "
                    f"Step time: {step_time:.3f}s, Throughput: {tokens_per_second:.2f} tokens/s, "
                    f"Elapsed since last eval: {elapsed_since_eval:.2f}s"
                )
                print(get_peak_memory_report(train_config['device']))

            # Decay the learning rate at the specified step.
            if step == train_config['t_lr_decay_step']:
                print('Decaying learning rate')
                set_optimizer_lr(optimizer, train_config['t_lr_decayed'])

            if checkpoint_every and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                path = checkpoint_path(checkpoint_dir, step)
                save_training_checkpoint(
                    path,
                    model,
                    optimizer,
                    train_config,
                    losses,
                    step=step,
                    train_loss=as_float(latest_train_loss),
                    dev_loss=as_float(latest_dev_loss),
                )
                prune_old_checkpoints(checkpoint_dir, int(keep_last or 0))
                print(f"Saved checkpoint to {path}")
        except StopIteration:
            # Handle the case where the training data iterator ends early.
            print("Training data iterator finished early.")
            break

    # --- Save Model and Final Evaluation ---

    # Perform a final evaluation of the model on training and development datasets.
    evaluation_losses = estimate_loss(model, train_config, 200)
    train_loss = evaluation_losses['train']
    dev_loss = evaluation_losses['dev']

    final_step = max(last_completed_step, start_step - 1)
    modified_model_out_path = unique_output_path(train_config['t_out_path'])

    # Save the model's state dictionary, optimizer state, and training metadata
    # (including the runtime device / PyTorch / CUDA versions for reproducibility).
    save_training_checkpoint(
        modified_model_out_path,
        model,
        optimizer,
        train_config,
        losses,
        step=final_step,
        train_loss=train_loss,
        dev_loss=dev_loss,
        is_final=True,
    )
    print(f"Saved model to {modified_model_out_path}")
    print(get_peak_memory_report(train_config['device']))
    print(f"Finished training. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
    if train_config.get("sample_prompt"):
        print("\nSample:\n" + sample_text(model, train_config, train_config["sample_prompt"]))
        print(f"\nMore: python scripts/generate_text.py --model_path {modified_model_out_path}")


if __name__ == "__main__":
    main()
