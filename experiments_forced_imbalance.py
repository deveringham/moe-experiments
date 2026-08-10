from transformers import MixtralConfig, MixtralForCausalLM, AutoModelForCausalLM, AutoTokenizer
import torch
from experiments_vllm import *

def generate_imbalanced_moe(imbalance_level, save_path, n_layers=1, n_local_experts=2, k=1, hidden_size=2048, intermediate_size=8192):
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    config = MixtralConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_hidden_layers=n_layers, num_attention_heads=32, num_key_value_heads=8,
        num_local_experts=n_local_experts, num_experts_per_tok=k,
    )
    model = MixtralForCausalLM(config)

    # If no imbalance, we're done.
    if imbalance_level > 0:
        with torch.no_grad():
            
            # Apply bias to embeddings such that sum is positive
            model.model.embed_tokens.weight.data += 1.0
    
            for name, module in model.named_modules():
                if "gate" in name:
                    
                    # Zero-center expert weights
                    row_means = module.weight.data.mean(dim=1, keepdim=True)
                    module.weight.data -= row_means
                    
                    # Apply imbalance
                    module.weight[0, :] += imbalance_level 
                
    model.save_pretrained(save_path)
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)