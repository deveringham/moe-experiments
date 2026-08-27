###
# experiments_vllm.py
#
# Routines for MoE experiments using vLLM serving interface.
# Dylan Everingham
# 06.08.2026
###

import torch
import asyncio
import time
import subprocess
import os
import json
import gzip
import collections
import glob
import urllib.request
import urllib.error
from openai import AsyncOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import vllm
from experiments_pretrained import *

# Spins up the vLLM server as a subprocess and blocks until ready.
def start_vllm_server(model_name, port=8000, seed=0, max_model_len=1024, batch_size=16, gpu_memory_utilization=0.85, n_gpus=1, enable_bnb=False, enable_expert_parallel=False, enable_prefix_caching=False, enable_eplb=False, trace_dir=None, trace_start_iteration=50, trace_active_iterations=10):
    print(f"Starting vLLM server for {model_name}...")
    
    cmd = [
        "vllm", "serve", model_name,
        "--port", str(port),
        "--dtype", "auto",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(batch_size),
        "--tensor-parallel-size", str(n_gpus),
        "--data-parallel-size", "1",
        "--seed", str(seed),
        "--override-generation-config", '{"temperature": 0.0}',
        "--enforce-eager",
        "--no-async-scheduling",
    ]
    
    if trace_dir:
        cmd.append("--profiler-config")
        cmd.append(f'{{"profiler": "torch", "torch_profiler_dir": "{trace_dir}", "active_iterations": {trace_active_iterations}, "delay_iterations": {trace_start_iteration}, "torch_profiler_with_stack": false}}')
    
    if enable_expert_parallel:
        cmd.append("--enable-expert-parallel")

    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")

    if enable_bnb:
         cmd.extend(["--quantization", "bitsandbytes"])

    if enable_eplb:
        cmd.append("--enable-eplb")
        
    server_process = subprocess.Popen(cmd)
    
    # Poll the endpoint for 200 OK
    print("Waiting for server to initialize ...")
    url = f"http://localhost:{port}/v1/models"
    
    while True:
        try:
            response = urllib.request.urlopen(url)
            if response.getcode() == 200:
                print("Server is ready!")
                break
        except urllib.error.URLError:
            time.sleep(5)
            
        if server_process.poll() is not None:
            raise RuntimeError("vLLM server process terminated unexpectedly.")
            
    return server_process

# Terminates the vLLM server subprocess
def stop_vllm_server(server_process):
    print("Shutting down vLLM server...")
    server_process.terminate()
    server_process.wait()
    print("Server successfully shut down.")

def start_profiling(port=8000):
    print("Starting vLLM PyTorch Profiler...")
    req = urllib.request.Request(f"http://localhost:{port}/start_profile", method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            print("Profiler started successfully.")
    except urllib.error.URLError as e:
        print(f"Failed to start profiler: {e}")

def stop_profiling(port=8000):
    print("Stopping vLLM PyTorch Profiler (Note: flushing traces to disk may take a few minutes)...")
    req = urllib.request.Request(f"http://localhost:{port}/stop_profile", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            print("Profiler stopped and traces flushed successfully.")
    except urllib.error.URLError as e:
        print(f"Failed to stop profiler: {e}")

# Sends a single streaming request and measures TTFT and TPOT
async def measure_request(client, model, prompt_idx, prompt, seed=0, max_new_tokens=100,
                          get_response=False, prompt_formatted=True):
    
    start_time = time.perf_counter()
    first_token_time = None
    if prompt_formatted:
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=max_new_tokens,
        seed=seed,
        temperature=0
    )

    response_str = ""
    num_output_tokens = 0
    
    async for chunk in response:
        # Get output tokens
        if get_response:
            if chunk.choices and chunk.choices[0].delta.content:
                response_str += chunk.choices[0].delta.content
        
        # Time of first token (in order to deduct decode fromm TPOT)
        if first_token_time is None and chunk.choices:
            first_token_time = time.perf_counter()
            
        # The last chunk when using include_usage=True contains the token stats
        if chunk.usage is not None:
            num_output_tokens = chunk.usage.completion_tokens
            num_input_tokens = chunk.usage.prompt_tokens

    end_time = time.perf_counter()

    # If generation failed or no tokens produced...
    if first_token_time is None:
        first_token_time = end_time

    # Calculate metrics
    ttft = first_token_time - start_time
    generation_time = end_time - first_token_time
    
    # Subtract 1 from output_tokens because the first token's time is captured in TTFT
    tpot = 0
    if num_output_tokens > 1:
        tpot = generation_time / (num_output_tokens - 1)
    
    result = {
        "prompt": prompt,
        "prompt_id": prompt_idx,
        "ttft": ttft,
        "tpot": tpot,
        "num_output_tokens": num_output_tokens,
        "num_input_tokens": num_input_tokens,
        "total_time": end_time - start_time
    }
    if get_response:
        result["response"] = response_str
    return result
    
# Runs a batch of prompts concurrently and calculates aggregate metrics
async def run_batch(client, model, prompts, seed=0, max_new_tokens=100, concurrency_limit=100, print_output=False, prompt_formatted=True):

    print(f"Sending batch of {len(prompts)} concurrent requests...")
    
    batch_start_time = time.perf_counter()

    # Use semaphore to limit request rate
    # pipeline should still saturate as long as concurrency_limit > batch_size
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Wrapper function that acquires the semaphore before making the request
    async def rate_limited_measure_request(i, prompt):
        async with semaphore:
            return await measure_request(
                client, 
                model, 
                i, 
                prompt, 
                seed=seed, 
                max_new_tokens=max_new_tokens, 
                get_response=False,
                prompt_formatted=prompt_formatted
            )
    
    # Fire all requests
    tasks = [rate_limited_measure_request(i, prompt) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    batch_end_time = time.perf_counter()
    total_batch_time = batch_end_time - batch_start_time
    
    if print_output:
        print("\n--- Per-Request Metrics ---")
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
            
        if print_output:
            print("\n--- Batch Metrics ---")
        if valid_requests > 0:
            avg_tpot = total_tpot / valid_requests
            if print_output:
                print(f"Average Per-Request TPOT: {avg_tpot * 1000:.2f} ms/token")

        throughput = total_tokens / total_batch_time
        if print_output:
            print(f"Total Batch Time: {total_batch_time:.2f}s")
            print(f"Total Tokens Generated: {total_tokens}")
            print(f"Overall Server Throughput: {throughput:.2f} tokens/second")
    
    return results 

# Run full experiment:
# - Start vLLM server
# - Start vLLM client
# - Run inference
# - Return timing measurements
async def measure_vllm_throughput(model, prompts, seed=0, max_new_tokens=100, concurrency_limit=1024,
                                  max_model_len=1024, batch_size=256, gpu_memory_utilization=0.85,
                                  n_gpus=1, n_warmup_samples=5,
                                  print_output=False, enable_bnb=False, enable_expert_parallel=False,
                                  enable_prefix_caching=False, enable_eplb=False,
                                  trace_dir=None, trace_active_iterations=2):
    server_process = None
    port = 8000
    results = None
    n_samples = len(prompts)
    try:
        # Start server
        server_process = start_vllm_server(model, port=port, seed=seed,
                                           max_model_len=max_model_len,
                                           batch_size=batch_size,
                                           gpu_memory_utilization=gpu_memory_utilization,
                                           n_gpus=n_gpus, enable_expert_parallel=enable_expert_parallel,
                                           enable_prefix_caching=enable_prefix_caching, enable_bnb=enable_bnb,
                                           enable_eplb=enable_eplb,
                                           trace_dir=trace_dir, trace_start_iteration=n_samples//2,
                                           trace_active_iterations=trace_active_iterations)

        # Start client
        client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    
        # Run warmup
        await run_batch(client, model, prompts[:n_warmup_samples],
                        seed=seed, print_output=False, max_new_tokens=max_new_tokens,
                        concurrency_limit=concurrency_limit,
                        prompt_formatted=True)
        
        # Start profiling
        start_profiling(port=port)
        
        # Run experiment
        results = await run_batch(client, model, prompts,
                                  seed=seed, print_output=print_output, max_new_tokens=max_new_tokens,
                                  prompt_formatted=True)
        
        # Stop profiling
        stop_profiling(port=port)

    except Exception as e:
        print(f"An error occurred during inference: {e}")

    finally:
        # Tear down server
        if server_process is not None:
            stop_vllm_server(server_process)

    return results

def plot_forcedimbalance_results(results_balanced, results_imbalanced):

    # Get metrics: TTFT, TPOT
    ttfts_balanced = [r['ttft']*1000 for r in results_balanced]
    ttfts_imbalanced = [r['ttft']*1000 for r in results_imbalanced]
    tpots_balanced = [r['tpot']*1000 for r in results_balanced]
    tpots_imbalanced = [r['tpot']*1000 for r in results_imbalanced]
    all_ttfts = ttfts_balanced + ttfts_imbalanced
    all_tpots = tpots_balanced + tpos_imbalanced
    min_ttft = min(all_ttfts)
    min_tpot = min(all_tpots)
    max_ttft = max(all_ttfts)
    max_tpot = max(all_tpots)
    avg_ttfts_balanced = np.mean(ttfts_balanced)
    avg_ttfts_imbalanced = np.mean(ttfts_imbalanced)
    avg_tpots_balanced = np.mean(tpots_balanced)
    avg_tpots_imbalanced = np.mean(tpots_imbalanced)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: TTFT
    ax = axes[0][0]
    # Create uniform bins
    n_bins = 100
    bin_width_ttft = (max_ttft - min_ttft) / n_bins
    bins_ttft = np.arange(min_ttft, max_ttft + bin_width_ttft, bin_width_ttft)
    
    # Hist for each
    ax.hist(ttfts_balanced[k], bins=bins_ttft,
                  label='baseline model')
    ax.hist(ttfts_imbalanced[k], bins=bins_ttft,
                  label='imbalanced model')

    ax.set_title('TTFT vs. Forced Imbalance')
    ax.set_xlabel('TTFT(ms)')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Plot 2: TPOT
    ax = axes[0][1]
    # Create uniform bins
    n_bins = 100
    bin_width_tpot = (max_tpot - min_tpot) / n_bins
    bins_tpot = np.arange(min_tpot, max_tpot + bin_width_tpot, bin_width_tpot)
    
    # Hist for each
    ax.hist(tpots_balanced[k], bins=bins_tpot,
                  label='baseline model')
    ax.hist(tpots_imbalanced[k], bins=bins_tpot,
                  label='imbalanced model')

    ax.set_title('TPOT vs. Forced Imbalance')
    ax.set_xlabel('TPOT(ms)')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.suptitle('Performance Metrics from vLLM')
    plt.tight_layout()
    plt.show()

def plot_eplb_results(results_eplb, results_noeplb):

    # Get metrics: TTFT, TPOT
    ttfts_eplb = [r['ttft']*1000 for r in results_eplb]
    ttfts_noeplb = [r['ttft']*1000 for r in results_noeplb]
    tpots_eplb = [r['tpot']*1000 for r in results_eplb]
    tpots_noeplb = [r['tpot']*1000 for r in results_noeplb]
    all_ttfts = ttfts_ebpl + ttfts_noeplb
    all_tpots = tpots_eplb + tpos_noeplb
    min_ttft = min(all_ttfts)
    min_tpot = min(all_tpots)
    max_ttft = max(all_ttfts)
    max_tpot = max(all_tpots)
    avg_ttfts_eplb = np.mean(ttfts_eplb)
    avg_ttfts_noeplb = np.mean(ttfts_noeplb)
    avg_tpots_eplb = np.mean(tpots_eplb)
    avg_tpots_noeplb = np.mean(tpots_noeplb)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: TTFT
    ax = axes[0][0]
    # Create uniform bins
    n_bins = 100
    bin_width_ttft = (max_ttft - min_ttft) / n_bins
    bins_ttft = np.arange(min_ttft, max_ttft + bin_width_ttft, bin_width_ttft)
    
    # Hist for each
    ax.hist(ttfts_eplb[k], bins=bins_ttft,
                  label='eplb enabled')
    ax.hist(ttfts_noeplb[k], bins=bins_ttft,
                  label='eplb disabled')

    ax.set_title('TTFT vs. EPLB')
    ax.set_xlabel('TTFT(ms)')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Plot 2: TPOT
    ax = axes[0][1]
    # Create uniform bins
    n_bins = 100
    bin_width_tpot = (max_tpot - min_tpot) / n_bins
    bins_tpot = np.arange(min_tpot, max_tpot + bin_width_tpot, bin_width_tpot)
    
    # Hist for each
    ax.hist(tpots_eplb[k], bins=bins_tpot,
                  label='eplb enabled')
    ax.hist(tpots_noeplb[k], bins=bins_tpot,
                  label='eplb disabled')

    ax.set_title('TPOT vs. EPLB')
    ax.set_xlabel('TPOT(ms)')
    ax.set_ylabel('Frequency')
    ax.legend()
    fig.suptitle('Performance Metrics from vLLM')
    plt.tight_layout()
    plt.show()

def plot_result_metrics(results_dict):
    
    # Remove any failed runs
    results_keys = list(results_dict.keys()) # e.g. alphas for synthetic workloads
    results = {k: results_dict[k] for k in results_keys if results_dict[k] is not None}
    results_keys = list(results.keys()) # update
            
    # Get metrics: TTFT, TPOT
    ttfts = {k: [r['ttft']*1000 for r in results[k]] for k in results_keys}
    tpots = {k: [r['tpot']*1000 for r in results[k]] for k in results_keys}
    all_ttfts = sum(list(ttfts.values()), [])
    all_tpots = sum(list(tpots.values()), [])
    min_ttft = min(all_ttfts)
    min_tpot = min(all_tpots)
    max_ttft = max(all_ttfts)
    max_tpot = max(all_tpots)
    avg_ttfts = {k: np.mean(ttfts[k]) for k in results_keys}
    avg_tpots = {k: np.mean(tpots[k]) for k in results_keys}
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    alpha_colors = plt.cm.plasma(torch.linspace(0, 1, len(results_keys)))

    # Plot 1: TTFT
    ax = axes[0][0]
    # Create uniform bins
    n_bins = 100
    bin_width_ttft = (max_ttft - min_ttft) / n_bins
    bins_ttft = np.arange(min_ttft, max_ttft + bin_width_ttft, bin_width_ttft)
    
    # Hist for each alpha
    for i, k in enumerate(results_keys):
        ax.hist(ttfts[k], bins=bins_ttft,
                      color=alpha_colors[i],
                      label=f'alpha = {k}')

    ax.set_title('TTFT vs. Alpha (Imbalance)')
    ax.set_xlabel('TTFT(ms)')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Plot 2: TPOT
    ax = axes[0][1]
    # Create uniform bins
    n_bins = 100
    bin_width_tpot = (max_tpot - min_tpot) / n_bins
    bins_tpot = np.arange(min_tpot, max_tpot + bin_width_tpot, bin_width_tpot)
    
    # Hist for each alpha
    for i, k in enumerate(results_keys):
        ax.hist(tpots[k], bins=bins_tpot,
                      color=alpha_colors[i],
                      label=f'alpha = {k}')

    ax.set_title('TPOT vs. Alpha (Imbalance)')
    ax.set_xlabel('TPOT(ms)')
    ax.set_ylabel('Frequency')
    ax.legend()

    # Plot 3: Avg TPOT vs. Alpha
    ax = axes[1][0]
    x = results_keys
    y = [avg_tpots[k] for k in results_keys]
    ax.plot(x, y, 'o-', color='b')
    ax.set_title('Average TPOT vs. Alpha (Imbalance)')
    ax.set_xlabel('Load Imbalance (Parameterized by Alpha)')
    ax.set_ylabel('TPOT(ms)')

    # Plot 4: Avg TTFT vs. Alpha
    ax = axes[1][1]
    x = results_keys
    y = [avg_ttfts[k] for k in results_keys]
    ax.plot(x, y, 'o-', color='b')
    ax.set_title('Average TTFT vs. Alpha (Imbalance)')
    ax.set_xlabel('Load Imbalance (Parameterized by Alpha)')
    ax.set_ylabel('TTFT(ms)')

    fig.suptitle('Performance Metrics from vLLM')
    plt.tight_layout()
    plt.show()
    

def plot_hist_prefill_throughput(overall_results, subject_results=None, x_limits=(0,20000), title=None):
    
    plt.figure(figsize=(10,8))
    
    overall_throughputs = [r['num_input_tokens']/r['ttft'] for r in overall_results]
    overall_avg_throughput = np.mean(overall_throughputs)
    overall_total_requests = len(overall_throughputs)
    all_throughputs = []
    all_throughputs += overall_throughputs
    if subject_results is not None:
        n_subjects = len(subject_results)
        subject_throughputs = {s:[r['num_input_tokens']/r['ttft'] for r in subject_results[s]] for s in subject_results}
        subject_avg_throughputs = {s:np.mean(subject_throughputs[s]) for s in subject_throughputs}
        subject_total_requests = {s:len(subject_throughputs[s]) for s in subject_throughputs}
        for s in subject_throughputs:
            all_throughputs += subject_throughputs[s]
    
    # Create uniform bins for histograms
    n_bins = 100
    min_throughput = max(min(all_throughputs), x_limits[0])
    max_throughput = min(max(all_throughputs), x_limits[1])
    bin_width_ms = (max_throughput - min_throughput) / n_bins
    fixed_bins = np.arange(min_throughput, max_throughput + bin_width_ms, bin_width_ms)
    
    # Plot subjects in blue...
    if subject_results is not None:
        result_idx = 0
        blue_colors = plt.get_cmap('Blues')(np.linspace(0.4, 1.0, n_subjects))
        for s in subject_throughputs:
            plt.hist(list(subject_throughputs[s]), bins=fixed_bins,
                     color=blue_colors[result_idx],
                     label=f'\'{s}\' ({subject_total_requests[s]} requests)')
            result_idx += 1
    
    # ...then superimpose overall in red
    plt.hist(list(overall_throughputs), bins=fixed_bins,
             color='r',
             label=f'prompts drawn from all subjects ({overall_total_requests} requests)')
    
    if title:
        plt.title(title, fontsize=14, pad=15)
    else:
        plt.title(f'Distribution of per-request Prefill Throughput for MMLU Requests', fontsize=14, pad=15)
    plt.xlabel('Throughput (tokens/s)', fontsize=12)
    plt.ylabel('Frequency (Number of Requests)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()