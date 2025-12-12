###
# pretrained.py
#
# Routines to fetch pretrained models for MoE experiments.
# Dylan Everingham
# 09.12.2025
###

@timing
def load_model_qwen():
    
    model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #device_map="auto",
        trust_remote_code=True,
        #torch_dtype=torch.float16 
    )
    print("Model loaded.")
    
    return model, tokenizer