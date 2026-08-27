###
# config_eplb.py
#
# Hyperparameters for MoE experiment toggling EPLB.
# Dylan Everingham
# 10.08.2026
###

import torch
from data import *

# Model configuration
model_id_deepseek = "deepseek-ai/DeepSeek-V2-Lite-Chat"
model_id_qwen = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
model_id_mistral = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = model_id_deepseek
model_id_simple = "deepseek"

# Inference deployment configuration
port = 8000
max_new_tokens = 100
max_model_len = 2048
gpu_memory_utilization = 0.6
n_gpus = 8
enable_expert_parallel = True
enable_prefix_caching = False
batch_size = 16
n_warmup_samples = 10
n_runs = 1

# Data configuration
n_samples = 100

# Profiling confiuguration
trace_dir = './vllm_benchmarking/traces/'
trace_id_eplb = f'{model_id_simple}_eplb_gpu{n_gpus}_batch{batch_size}_samples{n_samples}_runs{n_runs}'
trace_path_eplb = trace_dir + trace_id_eplb
trace_id_noeplb = f'{model_id_simple}_noeplb_gpu{n_gpus}_batch{batch_size}_samples{n_samples}_runs{n_runs}'
trace_path_noeplb = trace_dir + trace_id_noeplb

# Output configuration
results_file_eplb = f'./vllm_benchmarking/results_{trace_id_eplb}.pkl'
results_file_eplb = f'./vllm_benchmarking/results_{trace_id_eplb}.pkl'

# torch device
device = torch.device("cuda")