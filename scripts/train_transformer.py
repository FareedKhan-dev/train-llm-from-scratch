import torch
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
import numpy as np
from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator
from typing import Dict


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


def estimate_param_count(config: dict) -> int:
    """
    Estimate the total number of parameters for the transformer defined by *config*.
    This runs before the model is instantiated so users can reason about VRAM.
    """
    n_embed = config['n_embed']
    vocab_size = config['vocab_size']
    context_length = config['context_length']
    n_blocks = config['n_blocks']

    # Embedding tables
    token_emb = vocab_size * n_embed
    pos_emb = context_length * n_embed

    # Per-block: attention (QKV + output projection) + MLP (hidden + projection)
    attn = 4 * n_embed * n_embed          # key, query, value, proj (all bias=False)
    mlp_hidden = n_embed * (4 * n_embed)  # expand
    mlp_proj = (4 * n_embed) * n_embed    # shrink back
    per_block = attn + mlp_hidden + mlp_proj

    # Final lm_head
    lm_head = n_embed * vocab_size

    total = token_emb + pos_emb + n_blocks * per_block + lm_head
    return total


def estimate_memory_budget(config: dict, total_params: int) -> str:
    """
    Print a human-readable VRAM budget estimate based on config flags.
    Uses heuristics and is intended as a rough guide, not a precise allocator trace.
    """
    use_amp = config.get('use_amp', False)
    grad_ckpt = config.get('use_gradient_checkpointing', False)
    micro_batch = config.get('micro_batch_size', 0)

    fp_bytes = 2 if use_amp else 4
    param_gib = (total_params * fp_bytes) / (1024 ** 3)
    grad_gib = param_gib                               # same size as params
    optim_gib = (total_params * 8) / (1024 ** 3)       # AdamW m+v in fp32 regardless of AMP

    # Activation heuristic: ~n_blocks * B * T * n_embed * (mlp_expansion_factor)
    effective_batch = micro_batch if micro_batch > 0 else config['t_batch_size']
    act_factor = 10.0  # rough per-element byte factor for intermediates
    if grad_ckpt:
        act_factor = 2.0  # drastically smaller with checkpointing
    act_gib = (
        config['n_blocks']
        * effective_batch
        * config['t_context_length']
        * config['n_embed']
        * act_factor
    ) / (1024 ** 3)

    total_gib = param_gib + grad_gib + optim_gib + act_gib

    lines = [
        "",
        "=" * 72,
        "  VRAM Budget Estimate (rough heuristic)",
        "=" * 72,
        f"  Total parameters           : {total_params / 1e9:.2f} B",
        f"  Precision                  : {'BF16 / FP16' if use_amp else 'FP32'}",
        f"  Gradient checkpointing     : {'ON' if grad_ckpt else 'OFF'}",
        f"  Effective micro-batch size : {effective_batch}",
        "-" * 72,
        f"  Parameters  : {param_gib:7.2f} GiB",
        f"  Gradients   : {grad_gib:7.2f} GiB",
        f"  Optimizer   : {optim_gib:7.2f} GiB  (AdamW m+v, always fp32)",
        f"  Activations : {act_gib:7.2f} GiB",
        "-" * 72,
        f"  Estimated total VRAM : {total_gib:.2f} GiB",
        "=" * 72,
    ]

    if hasattr(torch.cuda, 'get_device_properties'):
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        avail_gib = props.total_memory / (1024 ** 3)
        lines.append(f"  GPU available  : {avail_gib:.2f} GiB  ({props.name})")
        if total_gib > avail_gib * 0.85:
            lines.append(
                "  ⚠ WARNING: Estimated usage exceeds 85% of available VRAM!"
            )
            lines.append(
                "    Try: reduce micro_batch_size, enable gradient checkpointing, or use AMP."
            )
    lines.append("")
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


# --- Initialize the Model and Print Parameters ---

# Print runtime/device diagnostics and reset GPU peak-memory stats before training.
print(get_device_report(config['device']))
if config['device'].startswith('cuda') and torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# Estimate parameter count and show a VRAM budget before building the model.
if config.get('report_memory_budget', False):
    estimated_params = estimate_param_count(config)
    print(estimate_memory_budget(config, estimated_params))

model = Transformer(
    n_head=config['n_head'],
    n_embed=config['n_embed'],
    context_length=config['context_length'],
    vocab_size=config['vocab_size'],
    N_BLOCKS=config['n_blocks']
).to(config['device'])

# Print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params:,}")

# --- Gradient Checkpointing ---
# Wrap every transformer block so activations are recomputed during backward
# instead of being stored. Saves significant VRAM at a ~20-30% speed cost.
if config.get('use_gradient_checkpointing', False):
    print("Enabling gradient checkpointing on all transformer blocks ...")
    for block in model.attn_blocks:
        # Save the original forward before we replace it so checkpoint can
        # call it directly (avoids infinite recursion through __call__).
        _orig_forward = block.forward  # bound method
        block.forward = lambda x, _fwd=_orig_forward: torch.utils.checkpoint.checkpoint(
            _fwd, x, use_reentrant=False
        )  # type: ignore[assignment]
    print("  Gradient checkpointing enabled.")

# --- Automatic Mixed Precision Setup ---
# Use torch.autocast for forward passes to run in BF16/FP16,
# halving param + activation VRAM with negligible accuracy impact.
use_amp = config.get('use_amp', False)
amp_dtype = config.get('amp_dtype', 'bf16')
if use_amp:
    amp_dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16}
    autocast_dtype = amp_dtype_map.get(amp_dtype, torch.bfloat16)
    # Only create a GradScaler for fp16; bf16 does not need one (same dynamic range as fp32).
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == 'fp16'))
    print(f"AMP enabled: forward passes will use {autocast_dtype}.")
else:
    scaler = None

# --- Optimizer Setup and Loss Tracking ---

# Set up the AdamW optimizer with the specified learning rate.
optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])

# List to track loss values during training.
losses = []

# Define a window size for averaging recent losses in the training loop.
AVG_WINDOW = 64

# Helper function to estimate the average loss for training and development data.
@torch.no_grad()
def estimate_loss(steps: int) -> Dict[str, float]:
    """
    Evaluate the model on training and development datasets and calculate average loss.

    Args:
        steps (int): Number of steps to evaluate.

    Returns:
        dict: Dictionary containing average losses for 'train' and 'dev' splits.
    """
    out = {}
    model.eval()  # Set the model to evaluation mode.

    for split in ['train', 'dev']:
        # Select the appropriate data path for the current split.
        data_path = config['train_path'] if split == 'train' else config['dev_path']

        # Create a batch iterator for evaluation.
        batch_iterator_eval = get_batch_iterator(
            data_path, config['t_batch_size'], config['t_context_length'], device=config['device']
        )

        # Initialize a tensor to track loss values for each evaluation step.
        losses_eval = torch.zeros(steps)
        for k in range(steps):
            try:
                # Fetch a batch and calculate the loss.
                xb, yb = next(batch_iterator_eval)
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                        _, loss = model(xb, yb)
                else:
                    _, loss = model(xb, yb)
                losses_eval[k] = loss.item()
            except StopIteration:
                # Handle the case where the data iterator ends early.
                print(f"Warning: Iterator for {split} ended early.")
                break

        # Compute the mean loss for the current split.
        out[split] = losses_eval[:k + 1].mean()

    model.train()  # Restore the model to training mode.
    return out

# --- Training Loop ---

# Create a batch iterator for the training data.
batch_iterator = get_batch_iterator(
    config['train_path'],
    config['t_batch_size'],
    config['t_context_length'],
    device=config['device']
)

# Gradient accumulation: split each batch into micro-batches when
# micro_batch_size > 0, otherwise process the full batch at once.
micro_batch_size = config.get('micro_batch_size', 0)
if micro_batch_size <= 0 or micro_batch_size >= config['t_batch_size']:
    micro_batch_size = config['t_batch_size']  # disabled → full batch

# Number of tokens processed per step (batch_size * context_length), used for throughput.
tokens_per_step = config['t_batch_size'] * config['t_context_length']
last_eval_time = time.perf_counter()

# Create a progress bar to monitor training progress.
pbar = tqdm(range(config['t_train_steps']))
for step in pbar:
    try:
        # Fetch a batch of input and target data.
        step_start_time = time.perf_counter()
        xb, yb = next(batch_iterator)

        # --- Micro-batch loop for gradient accumulation ---
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        batch_tokens = xb.numel()

        for i in range(0, config['t_batch_size'], micro_batch_size):
            xb_micro = xb[i : i + micro_batch_size]
            yb_micro = yb[i : i + micro_batch_size]
            micro_tokens = xb_micro.numel()  # may be smaller for last chunk

            # Run forward in AMP autocast when enabled.
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                    _, loss_raw = model(xb_micro, yb_micro)
                    # Scale by micro-batch's share of total tokens for correct gradient accumulation
                    loss_micro = loss_raw * (micro_tokens / batch_tokens)
                if scaler is not None:
                    scaler.scale(loss_micro).backward()
                else:
                    loss_micro.backward()
            else:
                _, loss_raw = model(xb_micro, yb_micro)
                loss_micro = loss_raw * (micro_tokens / batch_tokens)
                loss_micro.backward()

            # Accumulate weighted mean loss for logging (matches full-batch mean)
            step_loss += loss_raw.item() * (micro_tokens / batch_tokens)

        # Clip gradients and step the optimizer.
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Record the (denormalised) loss for tracking.
        losses.append(step_loss)
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        # Measure step time and instantaneous throughput for diagnostics.
        step_time = time.perf_counter() - step_start_time
        tokens_per_second = tokens_per_step / step_time if step_time > 0 else float('inf')

        # Periodically evaluate the model on training and development data.
        if step % config['t_eval_steps'] == 0:
            evaluation_losses = estimate_loss(config['t_eval_iters'])
            train_loss = evaluation_losses['train']
            dev_loss = evaluation_losses['dev']
            # Report timing/throughput for the most recent step and wall-time since last eval.
            now = time.perf_counter()
            elapsed_since_eval = now - last_eval_time
            last_eval_time = now
            print(
                f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}, "
                f"Step time: {step_time:.3f}s, Throughput: {tokens_per_second:.2f} tokens/s, "
                f"Elapsed since last eval: {elapsed_since_eval:.2f}s"
            )
            print(get_peak_memory_report(config['device']))

        # Decay the learning rate at the specified step.
        if step == config['t_lr_decay_step']:
            print('Decaying learning rate')
            for g in optimizer.param_groups:
                g['lr'] = config['t_lr_decayed']
    except StopIteration:
        # Handle the case where the training data iterator ends early.
        print("Training data iterator finished early.")
        break

# --- Save Model and Final Evaluation ---

# Create the output directory if it does not exist.
os.makedirs(config['t_out_path'].split('/')[0], exist_ok=True)

# Perform a final evaluation of the model on training and development datasets.
evaluation_losses = estimate_loss(200)
train_loss = evaluation_losses['train']
dev_loss = evaluation_losses['dev']

# Ensure unique model save path in case the file already exists.
modified_model_out_path = config['t_out_path']
save_tries = 0
while os.path.exists(modified_model_out_path):
    save_tries += 1
    model_out_name = os.path.splitext(config['t_out_path'])[0]
    modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"

# Save the model's state dictionary, optimizer state, and training metadata
# (including the runtime device / PyTorch / CUDA versions for reproducibility).
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'steps': len(losses),
        'device': config['device'],
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
    },
    modified_model_out_path
)
print(f"Saved model to {modified_model_out_path}")
print(get_peak_memory_report(config['device']))
print(f"Finished training. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
