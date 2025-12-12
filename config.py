###
# config.py
#
# Hyperparameters for MoE experiment runs.
# Dylan Everingham
# 09.12.2025
###

import torch

# Hyperparameters
batch_size = 32
block_size = 64  # Maximum context length for predictions
max_iters = 5000     # Total training steps
eval_interval = 500  # How often to evaluate loss
learning_rate = 3e-4
eval_iters = 200     # How many steps to average for evaluation
embedding_dim = 256  # Embedding dimension
n_heads = 4           # Number of attention heads
n_encoder_layers = 1 # Number of transformer encoder layers
n_decoder_layers = 1 # Number of transformer decoder layers
dropout = 0.1
n_epoch = 1
n_experts = 64
n_samples_train = 100
n_samples_val = 100

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'