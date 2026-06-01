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


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def get_device_report(device: str) -> str:
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
    if device.startswith('cuda') and torch.cuda.is_available():
        peak_allocated = bytes_to_gib(torch.cuda.max_memory_allocated())
        peak_reserved = bytes_to_gib(torch.cuda.max_memory_reserved())
        return (
            f"Peak VRAM allocated: {peak_allocated:.2f} GiB | "
            f"Peak VRAM reserved: {peak_reserved:.2f} GiB"
        )
    return "Peak VRAM allocated: N/A | Peak VRAM reserved: N/A"


# --- Initialize the Model and Print Parameters ---

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

optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])
losses = []
AVG_WINDOW = 64


@torch.no_grad()
def estimate_loss(steps: int) -> Dict[str, float]:
    out = {}
    model.eval()

    for split in ['train', 'dev']:
        data_path = config['train_path'] if split == 'train' else config['dev_path']
        batch_iterator_eval = get_batch_iterator(
            data_path, config['t_batch_size'], config['t_context_length'], device=config['device']
        )

        losses_eval = torch.zeros(steps)
        for k in range(steps):
            try:
                xb, yb = next(batch_iterator_eval)
                _, loss = model(xb, yb)
                losses_eval[k] = loss.item()
            except StopIteration:
                print(f"Warning: Iterator for {split} ended early.")
                break

        out[split] = losses_eval[:k + 1].mean()

    model.train()
    return out


batch_iterator = get_batch_iterator(
    config['train_path'],
    config['t_batch_size'],
    config['t_context_length'],
    device=config['device']
)

tokens_per_step = config['t_batch_size'] * config['t_context_length']
last_eval_time = time.perf_counter()

pbar = tqdm(range(config['t_train_steps']))
for step in pbar:
    try:
        step_start_time = time.perf_counter()
        xb, yb = next(batch_iterator)
        _, loss = model(xb, yb)

        losses.append(loss.item())
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_time = time.perf_counter() - step_start_time
        tokens_per_second = tokens_per_step / step_time if step_time > 0 else float('inf')

        if step % config['t_eval_steps'] == 0:
            evaluation_losses = estimate_loss(config['t_eval_iters'])
            train_loss = evaluation_losses['train']
            dev_loss = evaluation_losses['dev']
            now = time.perf_counter()
            elapsed_since_eval = now - last_eval_time
            last_eval_time = now
            print(
                f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}, "
                f"Step time: {step_time:.3f}s, Throughput: {tokens_per_second:.2f} tokens/s, "
                f"Elapsed since last eval: {elapsed_since_eval:.2f}s"
            )
            print(get_peak_memory_report(config['device']))

        if step == config['t_lr_decay_step']:
            print('Decaying learning rate')
            for g in optimizer.param_groups:
                g['lr'] = config['t_lr_decayed']
    except StopIteration:
        print("Training data iterator finished early.")
        break

os.makedirs(config['t_out_path'].split('/')[0], exist_ok=True)

evaluation_losses = estimate_loss(200)
train_loss = evaluation_losses['train']
dev_loss = evaluation_losses['dev']

modified_model_out_path = config['t_out_path']
save_tries = 0
while os.path.exists(modified_model_out_path):
    save_tries += 1
    model_out_name = os.path.splitext(config['t_out_path'])[0]
    modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"

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
