from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.config import default_config as config
from src.models.transformer import Transformer


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
    checkpoint_dir = os.path.dirname(path) or "."
    os.makedirs(checkpoint_dir, exist_ok=True)
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
    tmp_path = os.path.join(checkpoint_dir, f".{os.path.basename(path)}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
    model.load_state_dict(checkpoint['model_state_dict'])

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
    parser = argparse.ArgumentParser(description="Train the Transformer model from scratch.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = dict(config)
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

    # --- Initialize the Model and Print Parameters ---

    # Print runtime/device diagnostics and reset GPU peak-memory stats before training.
    print(get_device_report(train_config['device']))
    if train_config['device'].startswith('cuda') and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = Transformer(
        n_head=train_config['n_head'],
        n_embed=train_config['n_embed'],
        context_length=train_config['context_length'],
        vocab_size=train_config['vocab_size'],
        N_BLOCKS=train_config['n_blocks'],
    ).to(train_config['device'])

    # Print the total number of parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params:,}")

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

    # Number of tokens processed per step (batch_size * context_length), used for throughput.
    tokens_per_step = train_config['t_batch_size'] * train_config['t_context_length']
    last_eval_time = time.perf_counter()
    latest_train_loss = None
    latest_dev_loss = None

    # Create a progress bar to monitor training progress.
    pbar = tqdm(range(start_step, train_config['t_train_steps']))
    for step in pbar:
        try:
            # Fetch a batch of input and target data (and start the step timer).
            step_start_time = time.perf_counter()
            xb, yb = next(batch_iterator)

            # Perform a forward pass and compute the loss.
            _, loss = model(xb, yb)

            # Record the loss for tracking.
            losses.append(float(loss.item()))
            pbar.set_description(f"Train loss: {np.mean(losses[-avg_window:]):.4f}")

            # Backpropagate the loss and update the model parameters.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Clip gradients to prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
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


if __name__ == "__main__":
    main()
