###
# router_injection.py
#
# Classes for injecting router behavior (i.e. modifying router logits).
# Dylan Everingham
# 20.03.2026
###

# Dependencies
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from moe_hooks import *
from config import *

# MoE injection hook: registers a hook which sets router outputs
class MoERouterInjector(MoEHook):
    
    def __init__(self, model, n_experts=64, k=2):
        
        super(MoERouterInjector, self).__init__(model, n_experts=n_experts, k=k)
    
    # Function to set outputs to be injected
    # layer_id in [0,n_layers-1] sorted in order by router_names_sorted
    def set_router_outputs(self, router_id, outputs):
        module = module = self._get_router_module(router_id)
        self.routers[module]["injection_outputs"] = outputs
        self.routers[module]["enable_injection"] = True
    
    # Function to enable / disable router output injection
    def set_router_output_enable(self, router_id, enable=True):
        module = self._get_router_module(router_id)
        self.routers[module]["enable_injection"] = enable
    
            
# Implentation for native models from moe.py
class MoERouterInjectorNative(MoERouterInjector, MoEHookNative):
    
    def __init__(self, model, n_experts=64, k=2):
        super(MoERouterInjectorNative, self).__init__(model, n_experts=n_experts, k=k)


# Implentation for Qwen MoE models ex. Qwen1.5-MoE-A2.7B-Chat
class MoERouterInjectorQwen(MoERouterInjector, MoEHookQwen):
    
    def __init__(self, model):

        # Query expert information
        n_experts = model.config.num_experts
        k = model.config.num_experts_per_tok
        self.n_routers = model.config.num_hidden_layers

        super(MoEProbeQwen, self).__init__(model, n_experts=n_experts, k=k)
        
        if hasattr(model.config, "shared_expert_intermediate_size") and hasattr(model.config, "moe_intermediate_size"):
            shared_size = model.config.shared_expert_intermediate_size
            routed_size = model.config.moe_intermediate_size
            print(f"MoERouterInjector: Qwen1.5-MoE-A2.7B model also has shared expert with intermediate size: {shared_size} (Equivalent to ~{shared_size // routed_size} routed experts)")
            

# Implentation for DeepSeek MoE models ex. DeepSeek-V2-Lite-Chat
class MoERouterInjectorDeepSeek(MoERouterInjector, MoEHookDeepSeek):
            
    def __init__(self, model):

        # Query expert information
        n_experts = model.config.n_routed_experts
        k = model.config.num_experts_per_tok
        self.n_routers = model.config.num_hidden_layers - model.config.first_k_dense_replace

        super(MoERouterInjectorDeepSeek, self).__init__(model, n_experts=n_experts, k=k)
        
        if hasattr(model.config, "shared_expert_intermediate_size") and hasattr(model.config, "moe_intermediate_size"):
            shared_size = model.config.shared_expert_intermediate_size
            n_shared_experts = model.config.n_share_experts
            routed_size = model.config.moe_intermediate_size
            print(f"MoERouterInjector: DeepSeek-V2-Lite-Chat model also has {n_shared_experts} shared experts with the same intermediate size: {shared_size} (Compared to {routed_size} for routed experts)")

        # Add router output injection enable flags and actual outputs to be injected
        # DeepSeek router outputs: tuple of tokp indices and logits
        # each of size [batch*seq_len, k]
        for r in self.routers.values():
            r["enable_injection"] = False
            r["injection_outputs"] = (
                torch.zeros(self.k, dtype=torch.float).unsqueeze(0).to(device), 
                torch.zeros(self.k, dtype=torch.int).unsqueeze(0).to(device))
    
    # Modifies router outputs
    def hook_fn(self, module, inputs, outputs):
        #print(f"outputs: {outputs[0].size()}, {outputs[1].size()}")
        if self.routers[module]["enable_injection"]:
            
            # Replicate outputs across batch dimension
            batch_size = outputs[0].size(0)
            new_topk_indices = self.routers[module]["injection_outputs"][0].repeat(batch_size, 1).to(device)
            new_logits = self.routers[module]["injection_outputs"][1].repeat(batch_size, 1).to(device)
            new_outputs = (new_topk_indices, new_logits)
            #print(f"new outputs: {new_outputs[0].size()}, {new_outputs[1].size()}")
            return new_outputs
        else:
            return outputs
            
# Implentation for Mistral MoE models ex. Mixtral-8x7B-Instruct-v0.1
class MoERouterInjectorMistral(MoERouterInjector, MoEHookMistral):
    
    def __init__(self, model):

        # Query expert information
        n_experts = 8 # Mixtral experts set at 8 (see model name)
        k = model.config.num_experts_per_tok
        self.n_routers = model.config.num_hidden_layers

        super(MoERouterInjectorMistral, self).__init__(model, n_experts=n_experts, k=k)
        
        # No shared experts in Mixtral