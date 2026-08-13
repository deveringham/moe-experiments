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

# Data configuration
# For full dataset: n_samples = 15000
n_samples = 100

# Inference deployment configuration
max_new_tokens = 100
batch_size = 32

# Output configuration
# Get unique string id for the run
import datetime
timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
output_dir = './activations/'
run_id = f'{model_id_simple}_samples{n_samples}'
results_file = f'{output_dir}/activations_{run_id}_{timestamp}.pkl'

# torch device
device = torch.device("cuda")