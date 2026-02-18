###
# monitoring.py
#
# Classes for monitoring experiments.
# Dylan Everingham
# 22.01.2026
###

# Dependencies
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from transformers import AutoConfig

# MoE monitoring probe. Attaches a callback to routing modules which computes routing metrics
class MoEProbe:

    def __init__(self, model, n_experts=64, k=2):
        self.model = model
        self.n_experts = n_experts
        self.k = k
        self.logs = {}  # Metrics are logged here after each router activation
                        # indexed by layer name
        self.most_recent = {} # Holds per-router metrics for most recent activation
        self.routers = {} # Holds router names, hooks. Indexed by router module object
        
        self.register(model)
        
        print(f"MoEProbe: Model has {self.n_experts} experts and selects k={self.k} at each layer.")
    
    # Default function used for identifying router modules
    # Designed to work with the custom MoE implementation from moe.py
    def attach_fn(self, name, module):
        return name.endswith(".gating_func")
    
    # Default function used for extracting router metrics
    # Designed to work with the custom MoE implementation from moe.py
    def hook_fn(self, module, inputs, outputs):
        
        # outputs are the raw logits [batch, seq_len, num_experts]
        
        # Access expert activations, topk
        routing_weights = module.routing_weights
        sparse_routing_weights = module.sparse_routing_weights
        topk_indices = module.topk_indices
        topk_vals = module.topk_vals
        
        # Metric: expert activation (load)
        active_experts = topk_indices.flatten().cpu().numpy().tolist()
        
        # Format expert activation also as a matrix to track per-request / per-token
        # Include a binary mask as well as one containing actual probabilities (for top-k experts, zero elsewhere)
        eam_binary = torch.zeros_like(outputs, dtype=torch.int8)
        eam_binary.scatter_(2, topk_indices, 1)
        eam_probs = torch.zeros_like(outputs)
        eam_probs.scatter_(2, topk_indices, topk_values)
        
        # f_i: fraction of tokens routed to each expert
        #expert_mask = torch.ceil(sparse_routing_weights) # [batch, n_experts]
        #tokens_per_expert = torch.mean(expert_mask, dim=0) # [n_experts]

        # P_i: mean router probability over tokens for each expert
        router_prob_per_expert = torch.mean(routing_weights, dim=0) # [n_experts]
        
        # Metric: router entropy (uncertainty)
        # High entropy = router is unsure (or load balancing is forcing uniformity)
        # Low entropy = strong specialization
        entropy = -torch.mean(router_prob_per_expert * torch.log(router_prob_per_expert + 1e-9))
        
        # Store lightweight statistics and full activations
        name = self.routers[module]["name"]
        log = {
            "entropy": entropy.item().cpu(),
            "active_experts": active_experts,
            "eam_binary": eam_binary.cpu(),
            "eam_probs": eam_probs.cpu(),
            #"logits": logits.detach().cpu()
        }
        if name in self.logs:
            self.logs[name].append(log)
        else:
            self.logs[name] = [log]
        self.most_recent[name] = log
        
    def clear(self):
        self.logs = {}
        self.most_recent = {}
    
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def register(self, model):
        
        print(f"MoEProbe: Scanning model for routers...")
        for name, module in model.named_modules():
            if self.attach_fn(name, module):
                handle = module.register_forward_hook(self.hook_fn)
                self.routers[module] = {"name": name, "hook": handle}
                self.logs[name] = []
        self.n_routers = len(self.routers)
        print(f"MoEProbe: Attached probes to {self.n_routers} router layers.")

    def print_count(self):
        print(f"MoEProbe: Captured {len(self.logs)} routing events from {self.n_routers} router modules.")
    
    def plot_loadbalance(self, router_idx=0, wandb_run=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        # Get list of routers
        routers = sorted(self.logs.keys())
        
        # Select single router for which we will plot
        selected_layer = routers[router_idx]

        # Collate data
        all_indices = []
        all_entropies = []
        for log in self.logs[selected_layer]:
            all_indices.extend(log['active_experts'])
            all_entropies.append(log['entropy'])
            
        counts = Counter(all_indices)
        avg_entropy = np.mean(all_entropies)
        expert_ids = np.array(sorted(counts.keys()))
        count_per_expert = np.array([counts[i] for i in expert_ids])
        tokens = counts.total()
        freqs = count_per_expert / tokens
        ax.bar(expert_ids, freqs, \
                label='tokens: %d entropy: %2.4f' % (tokens, avg_entropy), \
                alpha=0.7)

        ax.set_title("Expert Activation Frequency (Load)")
        ax.set_xlabel(f"Expert Index (0-{self.n_experts})")
        ax.set_ylabel("Activation Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Log plot with wandb
        if wandb_run is not None:
            wandb_run.log({"Load Balance Plot": fig})
        
# MoE monitoring probe specialized for Qwen MoE models
class MoEProbeQwen(MoEProbe):
    
    def __init__(self, model):
        # Load the configuration (does not load the heavy model weights)
        config = AutoConfig.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")

        # Query expert information
        n_experts = config.num_experts
        k = config.num_experts_per_tok
        self.n_routers = config.num_hidden_layers
        
        # List of router layer names in sorted order
        self.router_names_sorted = [f"model.layers.{i}.mlp.gate" for i in range(self.n_routers)]

        super(MoEProbeQwen, self).__init__(model, n_experts=n_experts, k=k)
        
        if hasattr(config, "shared_expert_intermediate_size") and hasattr(config, "moe_intermediate_size"):
            shared_size = config.shared_expert_intermediate_size
            routed_size = config.moe_intermediate_size
            print(f"MoEProbe: Qwen1.5-MoE-A2.7B model also has shared expert with intermediate size: {shared_size} (Equivalent to ~{shared_size // routed_size} routed experts)")
    
    # Function used for identifying router modules
    # Designed to work with Qwen MoE models
    def attach_fn(self, name, module):
        
        # In Qwen1.5-MoE, the router is a linear layer named 'gate'
        return name.endswith(".gate")
    
    # Function used for extracting router metrics
    def hook_fn(self, module, inputs, outputs):
        
        # outputs are the raw logits [batch * seq_len, n_experts]
        router_logits = outputs
        
        # Calculate probabilities
        probs = torch.softmax(router_logits, dim=-1)
        
        # Metric: router entropy (uncertainty)
        # High entropy = Router is unsure (or load balancing is forcing uniformity)
        # Low entropy = Strong specialization
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
        
        # Metric: expert activation (Load)
        # Manually recalculate top-k
        topk_values, topk_indices = torch.topk(probs, self.k, dim=-1)
        
        # Format expert activation also as a matrix to track per-request / per-token
        # Include a binary mask as well as one containing actual probabilities (for top-k experts, zero elsewhere)
        eam_binary = torch.zeros_like(outputs, dtype=torch.int8)
        eam_binary.scatter_(1, topk_indices, 1)
        eam_probs = torch.zeros_like(outputs)
        eam_probs.scatter_(1, topk_indices, topk_values)
        
        # Store lightweight statistics (move to cpu to save vram)
        name = self.routers[module]["name"]
        log = {
            "entropy": entropy.item(),
            "active_experts": topk_indices.flatten().cpu().numpy().tolist(),
            "eam_binary": eam_binary.cpu(),
            "eam_probs": eam_probs.float().cpu(),
            #"logits": logits.detach().cpu()
        }
        if name in self.logs:
            self.logs[name].append(log)
        else:
            self.logs[name] = [log]
        self.most_recent[name] = log
    
    # get expert activation matrix (EAM)
    # EAM has dimensions [generated_tokens, n_experts, n_routers]
    def get_eam(self, binary=False):
        if binary:
            eam_type = 'eam_binary'
        else:
            eam_type = 'eam_probs'
        
        eam = torch.stack([torch.cat([l[eam_type] for l in self.logs[n]], dim=0) for n in self.router_names_sorted], dim=-1)
        
        return eam
        
    def plot_loadbalance(self, router_idx=0, wandb_run=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        # Select single router for which we will plot
        name = self.router_names_sorted[router_idx]

        # Collate data
        all_indices = []
        all_entropies = []
        for log in self.logs[name]:
            all_indices.extend(log['active_experts'])
            all_entropies.append(log['entropy'])
            
        counts = Counter(all_indices)
        avg_entropy = np.mean(all_entropies)
        expert_ids = np.array(sorted(counts.keys()))
        count_per_expert = np.array([counts[i] for i in expert_ids])
        tokens = counts.total()
        freqs = count_per_expert / tokens
        ax.bar(expert_ids, freqs, \
                label='tokens: %d entropy: %2.4f' % (tokens, avg_entropy), \
                alpha=0.7)

        ax.set_title("Expert Activation Frequency (Load)")
        ax.set_xlabel(f"Expert Index (0-{self.n_experts})")
        ax.set_ylabel("Activation Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Log plot with wandb
        if wandb_run is not None:
            wandb_run.log({"Load Balance Plot": fig})
            
    # plot of expert activation matrix in binary mask form
    def plot_eam_perrouter(self, router_idx=0, binary=False):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        name = self.router_names_sorted[router_idx]
        eam = self.get_eam(binary=binary)
        eam = eam[:,:,router_idx]
        eam = torch.transpose(eam, 0, 1)
        
        # Draw a line to separate prefill and decoding stages
        prompt_len = self.logs[name][0]['eam_binary'].shape[0]
        plt.axvline(x=prompt_len, color='red', linestyle='--', linewidth=2, label="Gen Start")
        
        sns.heatmap(eam, cmap="Blues", cbar=False, cbar_kws={'label': 'Active'})
        plt.title(f"Expert Activation Matrix (Layer {name})\nPrompt Prefill + Generated Tokens")
        plt.xlabel("Token Position (Time)")
        plt.ylabel("Expert ID")
        plt.legend()
   
    # plot of expert activation for a single token across all router layers
    def plot_eam_pertoken(self, token_idx=0, binary=False):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        eam = self.get_eam(binary=binary)
        eam = eam[token_idx,:,:]
        
        sns.heatmap(eam, cmap="Blues", cbar=False, cbar_kws={'label': 'Active'})
        plt.title(f"Expert Activation Matrix (Token Position {token_idx})")
        plt.xlabel("MoE Layer ID")
        plt.ylabel("Expert ID")