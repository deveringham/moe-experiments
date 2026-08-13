###
# experiments_pretrained.py
#
# Routines for MoE experiments on pretrained models from Huggingface.
# Dylan Everingham
# 18.02.2026
###

import torch
import time
import h5py
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.distributed.configuration_utils import DistributedConfig
from monitoring import *
from data import *

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
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"load_tokenizer: no pad token defined for {model_id}, using eos_token ({tokenizer.eos_token!r}) as pad_token.").
    tokenizer.padding_side = "left"
    return tokenizer


def chat_generate(model, tokenizer, probe=None, prompt="", max_new_tokens=100, clear_probe=True, prompt_formatted=True):

    # Clear monitoring probe
    if probe and clear_probe:
        probe.clear()
        
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
        max_new_tokens=max_new_tokens,
        attention_mask=model_inputs['attention_mask'],
        pad_token_id=tokenizer.pad_token_id,

    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Get metrics
    if probe:
        probs = probe.get_probs()
        active_experts = probe.get_active_experts()
        
        return response, probs, active_experts
    else:
        return response, None, None
        

def chat_generate_batched(model, tokenizer, probe=None, prompts=[""], max_new_tokens=100, clear_probe=True, prompt_formatted=True):
 
    # Clear monitoring probe
    if probe and clear_probe:
        probe.clear()
 
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
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode and return
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
 
    if probe:
        batch_size = len(prompts)
        probs = probe.get_probs(batch_size=batch_size)
        active_experts = probe.get_active_experts(batch_size=batch_size)
 
        return responses, probs, active_experts
    else:
        return responses, None, None


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
            
def get_activations_mmlu(model, tokenizer, dataset, probe=None, batch_size=1,
                         max_new_tokens=100):
    
    # Results will contain router logits for each sample plus prompt and response
    results = []

    # Formatted prompts
    prompts = format_prompts_mmlu(dataset, prompt_reps=1)
    n_samples = len(prompts)
    
    # For each sample...
    count = 0
    for d in dataset:
    
        print(f"Generating response {count+1}/{n_samples}...")
        
        # Start timing
        start_time = time.perf_counter()

        # Generate response and get metrics
        response, probs, active_experts = chat_generate(model, tokenizer, probe=probe,
                                                        prompt=prompts[count], max_new_tokens=max_new_tokens,
                                                        prompt_formatted=True)
        
        # Stop timing
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        print(f"Inference took {inference_time:.3f}s / prompt.")

        # Store results
        result = {}
        if probe:
            result['probs'] = probs
            result['active_experts'] = active_experts       
        result['prompt'] = d['question']
        result['response'] = response
        text = tokenizer.apply_chat_template(prompts[count], tokenize=False, add_generation_prompt=True)
        result['prompt_tokenized'] = tokenizer([text])
        result['response_tokenized'] = tokenizer.encode(response)
        result['subject'] = d['subject']
        result['inference_time'] = inference_time
        
        results.append(result)

        count += 1

        """
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
    """

    # Batched version TBD

    return results