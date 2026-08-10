###
# experiments_throughput_hf.py
#
# Routines for MoE experiments using HuggingFace Transformers interface.
# Dylan Everingham
# 06.08.2026
###

import torch
import asyncio
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers.generation.streamers import BaseStreamer

from experiments_pretrained import *

trace_dir = "./traces"

# Global profiler instance for start/stop profiling functions
_global_profiler = None

# Custom HuggingFace Streamer type which records timestamps for output tokens
# in order to calculate TTFT and TPOT
class TimingStreamer(BaseStreamer):
    def __init__(self):
        super().__init__()
        self.prompt_received = False
        self.first_token_time = None
        self.last_token_time = None
        self.token_count = 0
        self.generated_ids = []

    def put(self, value):
        now = time.perf_counter()
        
        if not self.prompt_received:
            self.prompt_received = True
            return

        if self.first_token_time is None:
            self.first_token_time = now

        self.last_token_time = now
        self.token_count += value.numel()

        if value.ndim > 1:
            self.generated_ids.extend(value[0].tolist())
        else:
            self.generated_ids.extend(value.tolist())

    def end(self):
        pass


def start_profiling(trace_dir=trace_dir):
    global _global_profiler
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        _global_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        _global_profiler.start()


def stop_profiling():
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.stop()
        _global_profiler = None


# Runs a batch of prompts concurrently/sequentially and calculates aggregate metrics
def run_batch(model, tokenizer, prompts, max_new_tokens=100, batch_size=256, print_output=False, prompt_formatted=True):

    print(f"Sending batch of {len(prompts)} requests...")

    results = []
    
    batch_start_time = time.perf_counter()

    # Process in chunks of batch_size
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        current_batch_size = len(batch_prompts)

        text_inputs = []
        for prompt in batch_prompts:
            if prompt_formatted and isinstance(prompt, list):
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    text_inputs.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
                else:
                    text_inputs.append(" ".join([m.get("content", "") for m in prompt]))
            elif not prompt_formatted:
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    text_inputs.append(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
                else:
                    text_inputs.append(prompt)
            else:
                text_inputs.append(prompt)

        inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to(model.device)
        
        # Calculate tokens per sequence
        num_input_tokens = inputs.input_ids.shape[1] 

        streamer = TimingStreamer()
        chunk_start = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id
            )
        
        responses = tokenizer.batch_decode(outputs[:, num_input_tokens:], skip_special_tokens=True)

        chunk_end = time.perf_counter()

        first_token_time = streamer.first_token_time if streamer.first_token_time is not None else chunk_end
        
        # Calculate timing metrics
        ttft = first_token_time - chunk_start
        generation_time = chunk_end - first_token_time
        
        # Total tokens across all sequences in the batch
        total_output_tokens = streamer.token_count
        avg_output_tokens_per_seq = total_output_tokens // current_batch_size
        
        tpot = 0
        if total_output_tokens > current_batch_size:
            # Average TPOT per generated token across the batch
            tpot = generation_time / (total_output_tokens - current_batch_size)

        # Append result for each prompt in the batch
        for j, prompt in enumerate(batch_prompts):
            results.append({
                "prompt": prompt,
                "prompt_id": i + j,
                "response": responses[j],
                "ttft": ttft, 
                "tpot": tpot,
                "num_output_tokens": avg_output_tokens_per_seq,
                "num_input_tokens": num_input_tokens,
                "total_time": chunk_end - chunk_start
            })

    total_batch_time = time.perf_counter() - batch_start_time
    
    total_tpot = 0
    total_tokens = 0
    valid_requests = 0

    for res in results:
        if print_output:
            print(f"Request {res['prompt_id']}: TTFT = {res['ttft']:.4f}s | "
                  f"TPOT = {res['tpot']*1000:.2f}ms | Tokens = {res['num_output_tokens']}")

        if res['num_output_tokens'] > 1:
            total_tpot += res['tpot']
            total_tokens += res['num_output_tokens']
            valid_requests += 1

    avg_tpot = total_tpot / valid_requests if valid_requests > 0 else 0
    throughput = total_tokens / total_batch_time if total_batch_time > 0 else 0

    if print_output:
        print("\n--- Batch Metrics ---")
        if valid_requests > 0:
            print(f"Average Per-Request TPOT: {avg_tpot * 1000:.2f} ms/token")
        print(f"Total Batch Time: {total_batch_time:.2f}s")
        print(f"Total Tokens Generated: {total_tokens}")
        print(f"Overall Throughput: {throughput:.2f} tokens/second")
    
    return results, avg_tpot


def run_experiment_throughput(model, tokenizer, prompts, seed=0, max_new_tokens=100, batch_size=256,
                              n_warmup_samples=5, print_output=False,
                              trace_dir=None):
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

    # Run warmup
    if n_warmup_samples > 0:
        run_batch(model, tokenizer, prompts[:n_warmup_samples],
                        print_output=False, max_new_tokens=max_new_tokens, batch_size=batch_size,
                        prompt_formatted=True)
    
    # Start profiling
    if trace_dir:
        start_profiling(trace_dir=trace_dir)
    
    # Run experiment
    results = run_batch(model, tokenizer, prompts[n_warmup_samples:],
                        print_output=print_output, max_new_tokens=max_new_tokens, batch_size=batch_size,
                        prompt_formatted=True)
    
    # Stop profiling
    if trace_dir:
        stop_profiling()

    return results


def plot_hist_ttfts(overall_results, subject_results=None, x_limits=(1000, 20000), title=None):
    plt.figure(figsize=(10, 8))
    
    overall_ttfts = [r['ttft'] * 1000 for r in overall_results]
    overall_total_requests = len(overall_ttfts)
    all_ttfts = [] + overall_ttfts

    if subject_results is not None:
        n_subjects = len(subject_results)
        subject_ttfts = {s: [r['ttft'] * 1000 for r in subject_results[s]] for s in subject_results}
        subject_total_requests = {s: len(subject_ttfts[s]) for s in subject_ttfts}
        for s in subject_ttfts:
            all_ttfts += subject_ttfts[s]
    
    n_bins = 100
    min_ttft = max(min(all_ttfts), x_limits[0]) if all_ttfts else x_limits[0]
    max_ttft = min(max(all_ttfts), x_limits[1]) if all_ttfts else x_limits[1]
    bin_width_ms = (max_ttft - min_ttft) / n_bins if max_ttft > min_ttft else 1
    fixed_bins = np.arange(min_ttft, max_ttft + bin_width_ms, bin_width_ms)
    
    if subject_results is not None:
        result_idx = 0
        blue_colors = plt.get_cmap('Blues')(np.linspace(0.4, 1.0, n_subjects))
        for s in subject_ttfts:
            plt.hist(list(subject_ttfts[s]), bins=fixed_bins,
                     color=blue_colors[result_idx],
                     label=f'\'{s}\' ({subject_total_requests[s]} requests)')
            result_idx += 1
    
    plt.hist(list(overall_ttfts), bins=fixed_bins,
             color='r',
             label=f'prompts drawn from all subjects ({overall_total_requests} requests)')
    
    plt.title(title if title else 'Distribution of per-request TTFT for MMLU Requests', fontsize=14, pad=15)
    plt.xlabel('TTFT (ms)', fontsize=12)
    plt.ylabel('Frequency (Number of Requests)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()


def plot_hist_tpots(overall_results, subject_results=None, x_limits=(0, 50), title=None):
    plt.figure(figsize=(10, 8))
    
    overall_tpots = [r['tpot'] * 1000 for r in overall_results]
    overall_total_requests = len(overall_tpots)
    all_tpots = [] + overall_tpots

    if subject_results is not None:
        n_subjects = len(subject_results)
        subject_tpots = {s: [r['tpot'] * 1000 for r in subject_results[s]] for s in subject_results}
        subject_total_requests = {s: len(subject_tpots[s]) for s in subject_tpots}
        for s in subject_tpots:
            all_tpots += subject_tpots[s]
    
    n_bins = 100
    min_tpot = max(min(all_tpots), x_limits[0]) if all_tpots else x_limits[0]
    max_tpot = min(max(all_tpots), x_limits[1]) if all_tpots else x_limits[1]
    bin_width_ms = min((max_tpot - min_tpot) / n_bins, 1) if max_tpot > min_tpot else 1
    fixed_bins = np.arange(min_tpot, max_tpot + bin_width_ms, bin_width_ms)
    
    if subject_results is not None:
        result_idx = 0
        blue_colors = plt.get_cmap('Blues')(np.linspace(0.4, 1.0, n_subjects))
        for s in subject_tpots:
            plt.hist(list(subject_tpots[s]), bins=fixed_bins,
                     color=blue_colors[result_idx],
                     label=f'\'{s}\' ({subject_total_requests[s]} requests)')
            result_idx += 1
    
    plt.hist(list(overall_tpots), bins=fixed_bins,
             color='r',
             label=f'prompts drawn from all subjects ({overall_total_requests} requests)')
    
    plt.title(title if title else 'Distribution of per-request TPOT for MMLU Requests', fontsize=14, pad=15)
    plt.xlabel('TPOT (ms)', fontsize=12)
    plt.ylabel('Frequency (Number of Requests)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()


def plot_hist_prefill_throughput(overall_results, subject_results=None, x_limits=(0, 20000), title=None):
    plt.figure(figsize=(10, 8))
    
    overall_throughputs = [r['num_input_tokens'] / r['ttft'] if r['ttft'] > 0 else 0 for r in overall_results]
    overall_total_requests = len(overall_throughputs)
    all_throughputs = [] + overall_throughputs

    if subject_results is not None:
        n_subjects = len(subject_results)
        subject_throughputs = {s: [r['num_input_tokens'] / r['ttft'] if r['ttft'] > 0 else 0 for r in subject_results[s]] for s in subject_results}
        subject_total_requests = {s: len(subject_throughputs[s]) for s in subject_throughputs}
        for s in subject_throughputs:
            all_throughputs += subject_throughputs[s]
    
    n_bins = 100
    min_throughput = max(min(all_throughputs), x_limits[0]) if all_throughputs else x_limits[0]
    max_throughput = min(max(all_throughputs), x_limits[1]) if all_throughputs else x_limits[1]
    bin_width_ms = (max_throughput - min_throughput) / n_bins if max_throughput > min_throughput else 1
    fixed_bins = np.arange(min_throughput, max_throughput + bin_width_ms, bin_width_ms)
    
    if subject_results is not None:
        result_idx = 0
        blue_colors = plt.get_cmap('Blues')(np.linspace(0.4, 1.0, n_subjects))
        for s in subject_throughputs:
            plt.hist(list(subject_throughputs[s]), bins=fixed_bins,
                     color=blue_colors[result_idx],
                     label=f'\'{s}\' ({subject_total_requests[s]} requests)')
            result_idx += 1
    
    plt.hist(list(overall_throughputs), bins=fixed_bins,
             color='r',
             label=f'prompts drawn from all subjects ({overall_total_requests} requests)')
    
    plt.title(title if title else 'Distribution of per-request Prefill Throughput for MMLU Requests', fontsize=14, pad=15)
    plt.xlabel('Throughput (tokens/s)', fontsize=12)
    plt.ylabel('Frequency (Number of Requests)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()