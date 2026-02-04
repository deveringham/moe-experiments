###
# config.py
#
# Hyperparameters for MoE experiment runs.
# Dylan Everingham
# 02.02.2026
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
n_epochs = 10
n_experts = 64
ff_dim = 2048 # Hidden dimension of FFNs in basic Transformer
expert_dim = ff_dim//n_experts # Hidden dimension of expert FFNs

n_samples_train = 100 # Sample counts used for string reverse dataset
n_samples_val = 100
n_samples_test = 10

sampling_context_size = 32 # Context Size used for sliding window sampling

# String sizes used in string reverse dataset
string_reverse_min_len = 2
string_reverse_max_len = 20

# Dictionary containing names and weights of auxiliary loss terms
auxiliary_losses= {
    "loss_load_balancing": 0.0,
    "loss_z": 0.0
}

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'