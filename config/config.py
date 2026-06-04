# --- Configuration ---

import torch

# Define vocabulary size and transformer configuration (3 Billion)
VOCAB_SIZE = 50304          # Number of unique tokens in the vocabulary
CONTEXT_LENGTH = 512        # Maximum sequence length for the model
N_EMBED = 2048              # Dimension of the embedding space
N_HEAD = 16                 # Number of attention heads in each transformer block
N_BLOCKS = 64               # Number of transformer blocks in the model

# Paths to training and development datasets
TRAIN_PATH = "data/train/pile_train.h5"  # File path for the training dataset
DEV_PATH = "data/val/pile_dev.h5"      # File path for the validation dataset

# Transformer training parameters
T_BATCH_SIZE = 32          # Number of samples per training batch
T_CONTEXT_LENGTH = 16      # Context length for training batches
T_TRAIN_STEPS = 200000     # Total number of training steps
T_EVAL_STEPS = 1000        # Frequency (in steps) to perform evaluation
T_EVAL_ITERS = 250         # Number of iterations to evaluate the model
T_LR_DECAY_STEP = 50000    # Step at which to decay the learning rate
T_LR = 5e-4                # Initial learning rate for training
T_LR_DECAYED = 5e-5        # Learning rate after decay
T_OUT_PATH = "models/transformer_B.pt"  # Path to save the trained model

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Memory Optimisation Flags ---
# These flags help the default 3B-param model fit on consumer GPUs (e.g. RTX 3090/4090/5090)
# and even on an A100 40GB, which OOMs with vanilla FP32 training.

# Automatic Mixed Precision: switches to BF16 (NVIDIA >=Ampere) or FP16 for forward+activations.
# Saves ~50 % VRAM on parameters and activations with negligible accuracy impact.
USE_AMP = True
AMP_DTYPE = 'bf16'  # 'bf16' (recommended for Ampere+) or 'fp16'

# Gradient checkpointing: recomputes activations during backward instead of storing them.
# Trades ~20-30 % slower training for dramatically lower activation VRAM.
USE_GRADIENT_CHECKPOINTING = True

# Split each batch into micro-batches of this size for gradient accumulation.
# A lower micro_batch_size reduces peak activation VRAM proportionally.
# Set to 0 to disable (use the full batch).
MICRO_BATCH_SIZE = 4

# If True, print an estimated VRAM budget before training starts.
REPORT_MEMORY_BUDGET = True

# Store all configurations in a dictionary for easy access and modification
default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,

    # Memory optimisation
    'use_amp': USE_AMP,
    'amp_dtype': AMP_DTYPE,
    'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
    'micro_batch_size': MICRO_BATCH_SIZE,
    'report_memory_budget': REPORT_MEMORY_BUDGET,
}
