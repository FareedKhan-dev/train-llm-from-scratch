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
}
