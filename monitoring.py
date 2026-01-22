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
        
        
        # Access expert activations, topk
        #routing_weights = torch.softmax(router_logits, dim=-1) # [batch, seq_len, n_experts]
        routing_weights = module.routing_weights
        sparse_routing_weights = module.sparse_routing_weights
        topk_indices = module.topk_indices
        topk_vals = module.topk_vals
        
        # If padding mask is included in gating func module, use to to calculate statistics
        # only over non-padding tokens
        #if len(inputs) > 1:
        #    mask = inputs[1]
        #if mask is not None:
            
            # Expand mask to match expert dimension
            #mask = mask.unsqueeze(-1)
            
            # Zero out contributions from padding tokens
            #routing_weights = routing_weights * mask
            
            # Flatten along seq_len dimension
            #routing_weights = routing_weights.view(-1, routing_weights.size(-1)) # [batch*seq_len, n_experts]
            #mask = mask.view(-1, mask.size(-1)) # [batch*seq_len, 1]
            
            # Calculate top K over sequence using non-padding tokens only
            #topk_vals, topk_indices = torch.topk(routing_weights, self.k, dim=-1) # [batch*seq_len, k] (both)
            
            # Metric: expert activation (load)
            # Track active experts only for non-padding tokens
            #non_padding_indices = topk_indices[mask.expand(-1, self.k).bool()]
            #active_experts = non_padding_indices.flatten().cpu().numpy().tolist()
        
        #else:
            
            # Flatten along seq_len dimension
            #routing_weights = routing_weights.view(-1, routing_weights.size(-1)) # [batch*seq_len, n_experts]
            
            # Calculate statistics over all tokens
            #topk_vals, topk_indices = torch.topk(routing_weights, self.k, dim=-1) # [batch, seq_len, k] (both)
            
            # Metric: expert activation (load)
            #active_experts = topk_indices.flatten().cpu().numpy().tolist()
        
        # Metric: expert activation (load)
        active_experts = topk_indices.flatten().cpu().numpy().tolist()
        
        # Check active experts.
        counts = Counter(active_experts)
        expert_ids = np.array(sorted(counts.keys()))
        count_per_expert = np.array([counts[i] for i in expert_ids])
        top_indices = np.argpartition(count_per_expert, -2)[-2:]
        top_experts = expert_ids[top_indices]
        #print(f"top experts: {top_experts}")
        
        
        # f_i: fraction of tokens routed to each expert
        #expert_mask = torch.ceil(sparse_routing_weights) # [batch, n_experts]
        #tokens_per_expert = torch.mean(expert_mask, dim=0) # [n_experts]

        # P_i: mean router probability over tokens for each expert
        router_prob_per_expert = torch.mean(routing_weights, dim=0) # [n_experts]
        
        # Metric: router entropy (uncertainty)
        # High entropy = router is unsure (or load balancing is forcing uniformity)
        # Low entropy = strong specialization
        entropy = -torch.mean(router_prob_per_expert * torch.log(router_prob_per_expert + 1e-9))
        
        # Store lightweight statistics
        step_data = {
            "layer": self.layer_names[module],
            "entropy": entropy.item(),
            "active_experts": active_experts
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
        #print(expert_ids)
        count_per_expert = np.array([counts[i] for i in expert_ids])
        #print(count_per_expert)
        tokens = counts.total()
        freqs = count_per_expert / tokens
        ax.bar(expert_ids, freqs, \
                label='tokens: %d entropy: %2.4f' % (tokens, avg_entropy), \
                alpha=0.7)
        n_experts = max(expert_ids)

        ax.set_title("Expert Activation Frequency (Load)")
        ax.set_xlabel(f"Expert Index (0-{n_experts})")
        ax.set_ylabel("Activation Frequency")
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