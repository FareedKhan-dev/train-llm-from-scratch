import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator
from typing import Dict

# --- Initialize the Model and Print Parameters ---

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

# Create a progress bar to monitor training progress.
pbar = tqdm(range(config['t_train_steps']))
for step in pbar:
    try:
        # Fetch a batch of input and target data.
        xb, yb = next(batch_iterator)

        # Perform a forward pass and compute the loss.
        _, loss = model(xb, yb)

        # Record the loss for tracking.
        losses.append(loss.item())
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        # Backpropagate the loss and update the model parameters.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Periodically evaluate the model on training and development data.
        if step % config['t_eval_steps'] == 0:
            evaluation_losses = estimate_loss(config['t_eval_iters'])
            train_loss = evaluation_losses['train']
            dev_loss = evaluation_losses['dev']
            print(f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")

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

# Save the model's state dictionary, optimizer state, and training metadata.
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'steps': len(losses),
    },
    modified_model_out_path
)
print(f"Saved model to {modified_model_out_path}")
print(f"Finished training. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")