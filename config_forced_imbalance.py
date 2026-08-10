###
# config.py
#
# Hyperparameters for MoE experiment runs with forced imbalance models.
# Dylan Everingham
# 10.08.2026
###

import torch

# Model configuration
imbalance_level_low = 0
imbalance_level_high = 100
n_layers = 8
n_local_experts = 8
k = 1
model_scale_factor = 1 # Scale the parameter count by multiplying with the intermediate dim
hidden_size = 2048 
intermediate_size = 8192 * model_scale_factor

# Inference deployment configuration
port = 8000
max_new_tokens = 100
max_model_len = 2048
gpu_memory_utilization = 0.6
n_gpus = 2
enable_expert_parallel = True
enable_prefix_caching = False
batch_size = 16
n_warmup_samples = 10

# Data configuration
# For full dataset: n_samples = 15000, max_new_tokens = 100, batch_size = 16
n_samples = 100

# Profiling confiuguration
trace_dir = './vllm_benchmarking/traces/'
trace_id_balanced = f'gpu{n_gpus}_batch{batch_size}_samples{n_samples}_imbalance{imbalance_level_low}_layers{n_layers}_n{n_local_experts}_k{k}_hiddensize{hidden_size}_intermediatesize{intermediate_size}'
trace_path_balanced = trace_dir + trace_id_balanced
trace_id_imbalanced = f'gpu{n_gpus}_batch{batch_size}_samples{n_samples}_imbalance{imbalance_level_high}_layers{n_layers}_n{n_local_experts}_k{k}_hiddensize{hidden_size}_intermediatesize{intermediate_size}'
trace_path_imbalanced = trace_dir + trace_id_imbalanced

# Output configuration
results_file_balanced = f'./vllm_benchmarking/results_{trace_id_balanced}.pkl'
results_file_imbalanced = f'./vllm_benchmarking/results_{trace_id_imbalanced}.pkl'

# torch device
device = torch.device("cuda")