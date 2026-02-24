###
# experiments_pretrained.py
#
# Routines for MoE experiments on pretrained models from Huggingface.
# Dylan Everingham
# 18.02.2026
###

import torch
import time
import datetime
import h5py
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from monitoring import *
from config import *
from data import *

eam_data_dir = "./eam_logs/"

def chat_generate(model, tokenizer, prompt="", max_new_tokens=100):
        
    # Set chat template and transfer to device
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate output
    generated_ids = model.generate(
        model_inputs['input_ids'],
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode and return
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def eam_single_generate(model, tokenizer, moe_probe, prompt="", max_new_tokens=100):

    # Clear monitoring probe
    moe_probe.clear()
        
    response = chat_generate(model, tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
    
    # Get EAM
    eam = moe_probe.get_eam()
    return response, eam

def save_eam_data(filename, run_id, eam_results):
    
    with h5py.File(filename, 'w') as f:
        
        # For each sample...
        count = 0
        for sample in eam_results:
            
            # Store EAM
            eam_dataset = f.create_dataset(
                f"eam_{count}", 
                data=sample['eam'].cpu().numpy(), 
                compression="gzip"
            )
            
            # Sore run id
            f.attrs['run_id'] = run_id

            # Store all other metrics
            for key, value in sample['metrics'].items():
                eam_dataset.attrs[key] = value
            
            count += 1
            
def run_experiment_qwen_mmlu_eam(n_samples, save_samples=100, max_new_tokens=100, shuffle_seed=100):
    
    # Get unique string id for the run
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    run_id = f"qwen2moe-{timestamp}-samples{n_samples}-tokens{max_new_tokens}"
    
    # Figure out intervals at which to save results to file
    n_saves = (n_samples // save_samples)
    if (n_samples % save_samples) != 0: n_saves += 1
    
    # Get model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        dtype="auto",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")
    
    # Attach MoE probe
    probe = MoEProbeQwen(model)
    
    # Get data
    dataset = get_data_mmlu(n_samples=n_samples, shuffle_seed=shuffle_seed)
    
    # Results will contain EAM for each sample plus prompt and response
    results = []
    
    # For each sample...
    count = 0
    for sample in dataset:
        
        count += 1
        print(f"Generating response {count}/{n_samples}...")
        
        prompt = sample['question']

        # Generate response and get EAM
        response, eam = eam_single_generate(model, tokenizer, probe,
                                            prompt=prompt, max_new_tokens=max_new_tokens)
        
        # Store results
        result = {}
        result['eam'] = eam
        result['metrics'] = {
            'prompt': prompt,
            'response': response,
            'prompt_tokenized': tokenizer.encode(prompt),
            'response_tokenized': tokenizer.encode(response),
            'subject': sample['subject'],
        }
        results.append(result)
        
        # Write results to file
        if (count % save_samples == 0):
            filename = eam_data_dir + run_id + '-n' + str((count//save_samples)-1) + '.h5'
            print("Saving outputs to file " + filename)
            save_eam_data(filename, run_id, results)
            results = []
            
    # Write final results
    if (count % save_samples) != 0:
        filename = eam_data_dir + run_id + '-n' + str(count//save_samples) + '.h5'
        print("Saving outputs to file " + filename)
        save_eam_data(filename, run_id, results)
        results = []

        
def run_experiment_qwen_mmlu(n_samples, max_new_tokens=100, shuffle_seed=100):
    
    # Get unique string id for the run
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    run_id = f"qwen2moe-{timestamp}-samples{n_samples}-tokens{max_new_tokens}"
    
    # Get model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        dtype="auto",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")
    
    # Get data
    dataset = get_data_mmlu(n_samples=n_samples, shuffle_seed=shuffle_seed)
    
    # Results will contain EAM for each sample plus prompt and response
    results = []
    
    # For each sample...
    count = 0
    for sample in dataset:
        
        count += 1
        print(f"Generating response {count}/{n_samples}...")
        
        # Time
        start_time = time.time()

        prompt = sample['question']

        # Generate response
        response = chat_generate(model, tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
        
        # Store results
        result = {
            'prompt': prompt,
            'response': response,
            'prompt_tokenized': tokenizer.encode(prompt),
            'response_tokenized': tokenizer.encode(response),
            'subject': sample['subject'],
        }
        results.append(result)
        
        # Stop timing
        end_time = time.time()
        print(f"{end_time - start_time:.3f}s")
   
    return results