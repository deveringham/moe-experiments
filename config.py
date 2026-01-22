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
learning_rate = 3e-4
embedding_dim = 256 # Embedding dimension
n_heads = 4 # Number of attention heads
n_encoder_layers = 1 # Number of transformer encoder layers
n_decoder_layers = 1 # Number of transformer decoder layers
dropout = 0.1
n_epochs = 4
n_experts = 64
ff_dim = 2048 # Hidden dimension of FFNs in basic Transformer
expert_dim = ff_dim//n_experts # Hidden dimension of expert FFNs
n_samples_val = 100
n_samples_train = n_samples_val*5
n_samples_test = n_samples_val

# Dictionary containing names and weights of auxiliary loss terms
auxiliary_losses = {
    "loss_load_balancing": 0.2,
    "loss_z": 0.0
}

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'