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

# Number of tokens processed per step (batch_size * context_length), used for throughput.
tokens_per_step = config['t_batch_size'] * config['t_context_length']
last_eval_time = time.perf_counter()

# Create a progress bar to monitor training progress.
pbar = tqdm(range(config['t_train_steps']))
for step in pbar:
    try:
        # Fetch a batch of input and target data (and start the step timer).
        step_start_time = time.perf_counter()
        xb, yb = next(batch_iterator)

        # Perform a forward pass and compute the loss.
        _, loss = model(xb, yb)

        # Record the loss for tracking.
        losses.append(loss.item())
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        # Backpropagate the loss and update the model parameters.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Clip gradients to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

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
