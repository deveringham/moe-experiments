###
# utils.py
#
# Misc. utility and helper functions for MoE experiments.
# Dylan Everingham
# 04.21.2026
###

import torch

# For VRAM memory leak debugging
ENABLE_VRAM_PRINTING = True
def print_vram(label):
    if ENABLE_VRAM_PRINTING:
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"{label} VRAM: {mem:.2f} GB")