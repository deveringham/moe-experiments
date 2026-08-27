###
# config_synthetic_workloads.py
#
# Hyperparameters for MoE experiment runs using pretrained models and MMLU prompts.
# Dylan Everingham
# 10.08.2026
###

import torch
from data import *

# Model configuration
model_id_deepseek = "deepseek-ai/DeepSeek-V2-Lite-Chat"
model_id_qwen = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
model_id_mistral = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = model_id_qwen
model_id_simple = "qwen_noquant"

# Inference deployment configuration
port = 8000
max_new_tokens = 100
max_model_len = 2048
gpu_memory_utilization = 0.6
n_gpus = 8
enable_expert_parallel = True
enable_prefix_caching = False
batch_size = 128
n_warmup_samples = 10
max_prompt_repeats = 0

# Data configuration
workloads_file = f'./workloads/workloads_repeats{max_prompt_repeats}_{model_id_simple}.pkl'

# Profiling confiuguration
trace_dir = './vllm_benchmarking/traces/'
trace_id = f'{model_id_simple}_gpu{n_gpus}_batch{batch_size}'

# Output configuration
results_file = f'./vllm_benchmarking/results_syntheticworkloads_{trace_id}.pkl'

# torch device
device = torch.device("cuda")