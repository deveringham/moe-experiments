###
# monitoring.py
#
# Classes for monitoring experiments.
# Dylan Everingham
# 12.12.2025
###

# Dependencies
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# MoE monitoring probe. Attaches a callback to routing modules which computes routing metrics
class MoEProbe:

    def __init__(self, k=2):
        self.k = k
        self.logs = [] # Stores per-step data
        self.layer_names = {}
    
    # Default function used for identifying router modules
    # Designed to work with the custom MoE implementation from moe.py
    def attach_fn(self, name, module):
        return name.endswith(".gating_func")
    
    # Default function used for extracting router metrics
    # Designed to work with the custom MoE implementation from moe.py
    def hook_fn(self, module, inputs, outputs):
        
        # outputs are the raw logits [batch, seq_len, num_experts]
        router_logits = outputs
        
        # Calculate probabilities
        probs = torch.softmax(router_logits, dim=-1)
        
        # Metric: Router Entropy (Uncertainty)
        # High entropy = Router is unsure (or load balancing is forcing uniformity)
        # Low entropy = Strong specialization
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        
        # Metric: Expert Activation (Load)
        # Manually recalculate Top-K to see which experts won
        topk_weights, topk_indices = torch.topk(probs, self.k, dim=-1)
        
        # Store lightweight statistics (move to CPU to save VRAM)
        step_data = {
            "layer": self.layer_names[module],
            "entropy": entropy.item(),
            "active_experts": topk_indices.flatten().cpu().numpy().tolist()
        }
        self.logs.append(step_data)
        
    
    def clear(self):
        self.logs = []

    def register(self, model):
        
        print(f"MoEProbe: Scanning model for routers...")
        count = 0
        for name, module in model.named_modules():
            if self.attach_fn(name, module): 
                self.layer_names[module] = name
                module.register_forward_hook(self.hook_fn)
                count += 1
        print(f"MoEProbe: Attached probes to {count} router layers.")

    def print_count(self):
        print(f"MoEProbe: Captured {len(self.logs)} routing events.")
    
    def plot_loadbalance(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # Collate data
        all_indices = []
        all_entropies = []
        for log in self.logs:
            all_indices.extend(log['active_experts'])
            all_entropies.append(log['entropy'])
            
        counts = Counter(all_indices)
        avg_entropy = np.mean(all_entropies)
        expert_ids = np.array(sorted(counts.keys()))
        counts_per_token = np.array([counts[i] for i in expert_ids])
        tokens = counts.total()
        freqs = counts_per_token / tokens
        ax.bar(expert_ids, freqs, \
                label='tokens: %d entropy: %2.4f' % (tokens, avg_entropy), \
                alpha=0.7)
        n_experts = max(expert_ids)

        ax.set_title("Expert Activation Frequency (Load)")
        ax.set_xlabel(f"Expert Index (0-{n_experts})")
        ax.set_ylabel("Activation Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# MoE monitoring probe specialized for Qwen MoE models
class MoEProbeQwen(MoEProbe):
    
    def __init__(self, k=2):
        super(MoEProbe, self).__init__(k=2)
    
    # Function used for identifying router modules
    # Designed to work with Qwen MoE models
    def attach_fn(self, name, module):
        
        # In Qwen1.5-MoE, the router is usually a Linear layer named 'gate'
        # inside the MoE block.
        return name.endswith(".gate")
    
    # Function used for extracting router metrics
    # Designed to work with Qwen MoE models
    def hook_fn(self, module, inputs, outputs):
        
        # outputs are the raw logits [batch, seq_len, n_experts]
        router_logits = outputs
        
        # Calculate probabilities
        probs = torch.softmax(router_logits, dim=-1)
        
        # Metric: Router Entropy (Uncertainty)
        # High entropy = Router is unsure (or load balancing is forcing uniformity)
        # Low entropy = Strong specialization
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        
        # Metric: Expert Activation (Load)
        # Manually recalculate Top-K to see which experts won
        topk_weights, topk_indices = torch.topk(probs, self.k, dim=-1)
        
        # Store lightweight statistics (move to CPU to save VRAM)
        step_data = {
            "layer": self.layer_names[module],
            "entropy": entropy.item(),
            "active_experts": topk_indices.flatten().cpu().numpy().tolist()
        }
        self.logs.append(step_data)