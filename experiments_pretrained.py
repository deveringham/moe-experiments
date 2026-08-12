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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.distributed.configuration_utils import DistributedConfig
from monitoring import *
from data import *

routing_data_dir = "./routing_logs/"

# torch device
device = torch.device("cuda")

def load_model(model_id, max_memory=None, enable_bnb=True):
    
    # Configure 4-bit quantization
    if enable_bnb:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory=max_memory,
        dtype=torch.float16,
        trust_remote_code=False,
        quantization_config=quantization_config,
    )
    
    tokenizer = load_tokenizer(model_id)
    return model, tokenizer
    
def load_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id)


def chat_generate(model, tokenizer, prompt="", max_new_tokens=100, prompt_formatted=True):
        
    # Set chat template and transfer to device
    if prompt_formatted:
        messages = prompt
    else:
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


def chat_generate_batched(model, tokenizer, prompts, max_new_tokens=100, prompt_formatted=True):
    
    # Apply chat template to a list of prompts
    texts = []
    for prompt in prompts:
        if prompt_formatted:
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        
    # Tokenize with padding
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate output
    generated_ids = model.generate(
        input_ids=model_inputs['input_ids'],
        attention_mask=model_inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode and return
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


def single_generate(model, tokenizer, probe=None, prompt="", max_new_tokens=100, clear_probe=True):

    # Clear monitoring probe
    if probe and clear_probe:
        probe.clear()
        
    response = chat_generate(model, tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
    
    # Get metrics
    if probe:
        probs = probe.get_probs()
        active_experts = probe.get_active_experts()
        
        return response, probs, active_experts
    else:
        return response, _, _

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

def save_routing_data(filename, run_id, results):
    
    with h5py.File(filename, 'w') as f:
        
        # For each sample...
        count = 0
        for sample in results:
            
            # Store metrics
            dataset_probs = f.create_dataset(
                f"probs_{count}",
                data=sample['probs'].cpu().numpy(), 
                compression="gzip"
            )
            dataset_active_experts = f.create_dataset(
                f"active_experts_{count}",
                data=sample['active_experts'].cpu().numpy(), 
                compression="gzip"
            )
            
            # Store run id
            f.attrs['run_id'] = run_id

            # Store all other metrics as attributes of prob dataset
            for key, value in sample['metrics'].items():
                dataset_probs.attrs[key] = value
            
            count += 1
            
def run_experiment_mmlu_eam(model, tokenizer, n_samples, probe=None, start_sample=0, save_samples=100, 
                            max_new_tokens=100, shuffle_seed=100,
                            save_results=True):
    
    # Get unique string id for the run
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    run_id = f"{model_choice}-{timestamp}-samples{n_samples}-tokens{max_new_tokens}"
    
    # Get data
    dataset = get_data_mmlu(n_samples=n_samples, shuffle_seed=shuffle_seed)
    
    # Results will contain router logits for each sample plus prompt and response
    results = []
    
    # For each sample...
    count = 0
    for sample in dataset:
        
        count += 1
        
        # Skip to starting sample
        if count >= start_sample:
            print(f"Generating response {count}/{n_samples}...")

            prompt = sample['question']
            
            # Start timing
            start_time = time.perf_counter()

            # Generate response and get metrics
            response, probs, active_experts = single_generate(model, tokenizer, probe=probe,
                                                              prompt=prompt, max_new_tokens=max_new_tokens)
            
            # Stop timing
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            print(f"Inference took {inference_time:.3f}s.")

            # Store results
            result = {}
            if probe:
                result['probs'] = probs
                result['active_experts'] = active_experts
            result['metrics'] = {
                'prompt': prompt,
                'response': response,
                'prompt_tokenized': tokenizer.encode(prompt),
                'response_tokenized': tokenizer.encode(response),
                'subject': sample['subject'],
                'inference_time': inference_time,
            }
            results.append(result)

            # Write results to file
            if ((count % save_samples == 0) and save_results):
                filename = routing_data_dir + run_id + '-n' + str((count//save_samples)-1) + '.h5'
                print("Saving outputs to file " + filename)
                save_routing_data(filename, run_id, results)
                results = []
            
    # Write final results
    if ((count % save_samples) and save_results) != 0:
        filename = routing_data_dir + run_id + '-n' + str(count//save_samples) + '.h5'
        print("Saving outputs to file " + filename)
        save_routing_data(filename, run_id, results)
        results = []

    # If not recording results, simply return them
    if not save_results:
        return results