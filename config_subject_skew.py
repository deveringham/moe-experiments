###
# config_subject_skew.py
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
model_id = model_id_mistral
model_id_simple = "mistral"

# Inference deployment configuration
port = 8000
max_new_tokens = 100
max_model_len = 2048
gpu_memory_utilization = 0.6
n_gpus = 2
enable_expert_parallel = True
enable_prefix_caching = False
batch_size = 8
n_warmup_samples = 10

# Data configuration
# For full dataset: n_samples = 15000
n_samples = 100
n_subjects = 1
subjects = get_mmlu_subjects()[:n_subjects]

# Profiling confiuguration
trace_dir = './vllm_benchmarking/traces/'
trace_id_general = f'{model_id_simple}_subjectgeneral_gpu{n_gpus}_batch{batch_size}_samples{n_samples}'
trace_path_general = trace_dir + trace_id_general
trace_ids_subjects = [f'{model_id_simple}_subject{s}_gpu{n_gpus}_batch{batch_size}_samples{n_samples}' for s in subjects]
trace_paths_subjects = [trace_dir + i for i in trace_ids_subjects]

# Output configuration
results_file_general = f'./vllm_benchmarking/results_{trace_id_general}.pkl'
results_files_subjects = [f'./vllm_benchmarking/results_{i}.pkl' for i in trace_ids_subjects]

# torch device
device = torch.device("cuda")