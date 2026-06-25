###
# router_patching.py
#
# Classes for altering router weights.
# Dylan Everingham
# 10.06.2026
###

import os
from experiments_pretrained import *
from moe_hooks import *

# Custom router which returns manipulated logits
class StaticRouter(torch.nn.Module):
    
    def __init__(self, n_experts, k, selected_experts):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.selected_experts = selected_experts

    def forward(self, hidden_states):
        
        min_val = 1e2
        max_val = 1e-2
        
        batch_shape = hidden_states.shape[:-1]
        logits = torch.full(
            (*batch_shape, self.k),
            min_val,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        logits = torch.reshape(logits, [logits.shape[0]*logits.shape[1], self.k])
        
        for i in range(k):
            idx = self.selected_experts[i]
            logits[..., idx] = max_val + ((k-i)*10)
            
        active_experts = torch.tensor(self.selected_experts, dtype=torch.int32, device=hidden_states.device)
        active_experts = torch.tile(active_experts, [*batch_shape, 1])
        active_experts = torch.reshape(active_experts, [active_experts.shape[0]*active_experts.shape[1], self.k])
        return active_experts, logits

def patch_routers(model_choice, output_dir, selected_experts=None):
    
    # Get model, tokenizer, and function to locate router modules
    if model_choice == "qwen":
        model, tokenizer = load_model_qwen()
        identify_router = identify_router_qwen
        n_experts = model.config.num_experts
    elif model_choice == "qwen_bitsandbytes":
        model, tokenizer = load_model_qwen_bitsandbytes()
        identify_router = identify_router_qwen
        n_experts = model.config.num_experts
    elif model_choice == "qwen_gptq":
        model, tokenizer = load_model_qwen_gptq()
        identify_router = identify_router_qwen
        n_experts = model.config.num_experts
    elif model_choice == "deepseek":
        model, tokenizer = load_model_deepseek()
        identify_router = identify_router_deepseek
        n_experts = model.config.n_routed_experts
    elif model_choice == "deepseek_bitsandbytes":
        model, tokenizer = load_model_deepseek_bitsandbytes()
        identify_router = identify_router_deepseek
        n_experts = model.config.n_routed_experts
    elif model_choice == "mistral":
        model, tokenizer = load_model_mistral()
        identify_router = identify_router_mistral
        n_experts = 8
    elif model_choice == "mistral_bitsandbytes":
        model, tokenizer = load_model_mistral_bitsandbytes()
        identify_router = identify_router_mistral
        n_experts = 8
    else:
        raise ValueError("Invalid model_choice. Select 'qwen', 'qwen_bitsandbytes', 'qwen_gptq', 'deepseek', 'deepseek_bitsandbytes_bitsandbytes', 'mistral', or 'mistral_bitsandbytes'.")
    
    k = model.config.num_experts_per_tok
    if not selected_experts:
        selected_experts = list(range(k))

    print("Adjusting router weights...")
    for i, layer in enumerate(model.model.layers):    
        for name, module in layer.named_modules():
            if hasattr(module, "gate"):
                
                print(f"Found router in layer {i}.")
                module.gate =  StaticRouter(n_experts, k, selected_experts)
    
    print(f"Saving patched model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer