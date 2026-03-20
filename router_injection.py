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

# MoE injection hook
class MoERouterInjector(MoEHook):
    
    def __init__(self, model, n_experts=64, k=2):
        
        super(MoERouterInjector, self).__init__(model, n_experts=n_experts, k=k)
        
        # Add router output injection enable flags and actual outputs to be injected
        # router outputs for each router are of size [n_experts] and default to 0s
        for r in self.routers.values(): 
            r["enable_injection"] = False
            r["injection_outputs"] = torch.zeros((self.n_experts), dytpe=torch.float)
    
    # Function to set outputs to be injected
    # layer_id in [0,n_layers-1] sorted in order by router_names_sorted
    def set_router_outputs(self, layer_id, outputs):
        module = module = self._get_router_module(layer_id)
        self.routers[module]["injection_outputs"] = outputs
        self.routers[module]["enable_injection"] = True
    
    # Function to enable / disable router output injection
    def set_router_output_enable(self, layer_id, enable=True):
        module = self._get_router_module(layer_id)
        self.routers[module]["enable_injection"] = False
        
    # Default function for router injection
    # Modifies router outputs
    def hook_fn(self, module, inputs, outputs):
        if self.routers[module]["enable_injection"]:
            outputs = self.routers[module]["injection_outputs"]
    
    