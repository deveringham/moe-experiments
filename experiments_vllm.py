###
# experiments_vllm.py
#
# Routines for MoE experiments using vLLM serving interface.
# Dylan Everingham
# 18.02.2026
###

import torch
import asyncio
import time
import subprocess
import urllib.request
import urllib.error
from openai import AsyncOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Spins up the vLLM server as a subprocess and blocks until ready.
def start_vllm_server(model_name, port=8000, seed=0, max_model_len=1024, gpu_memory_utilization=0.85, n_gpus=1, enable_expert_parallel=False):
    print(f"Starting vLLM server for {model_name}...")
    
    # Command array based on your notebook
    cmd = [
        "vllm", "serve", model_name,
        "--port", str(port),
        "--quantization", "gptq_marlin",
        "--dtype", "auto",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", "16",
        "--tensor-parallel-size", str(n_gpus),
        "--data-parallel-size", "1",
        "--seed", str(seed),
        "--override-generation-config", "{\"temperature\": 0.0}"
        #"--enable-eplb"
    ]
    
    if enable_expert_parallel:
        cmd.append("--enable-expert-parallel")
    
    # Start the process, output to current console (stdout/stderr)
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

# Terminates the vLLM server subprocess.
def stop_vllm_server(server_process):
    print("Shutting down vLLM server...")
    server_process.terminate()
    server_process.wait()
    print("Server successfully shut down.")

# Sends a single streaming request and measures TTFT and TPOT.
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
        messages= messages,
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
        
        # Time of first token (in order to deduct decode form TPOT)
        if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_time = time.perf_counter()
            
        # The last chunk when using include_usage=True contains the token stats
        if chunk.usage is not None:
            num_output_tokens = chunk.usage.completion_tokens

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
        "total_time": end_time - start_time
    }
    if get_response:
        result["response"] = response_str
    return result
    
# Runs a batch of prompts concurrently and calculates aggregate metrics
async def run_batch(client, model, prompts, seed=0, max_new_tokens=100, print_output=False, prompt_formatted=True):

    print(f"Sending batch of {len(prompts)} concurrent requests...")
    
    batch_start_time = time.perf_counter()
    
    # Fire all requests concurrently
    tasks = [measure_request(client, model, i, prompt, seed=seed, max_new_tokens=max_new_tokens, prompt_formatted=prompt_formatted) for i, prompt in enumerate(prompts)]
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
    
    return results, avg_tpot

# Run full experiment:
# - Start vLLM server
# - Start vLLM client
# - Run inference
# - Return timing measurements
async def run_experiment_vllm_throughput(model, prompts, seed=0, max_new_tokens=100,
                                         max_model_len=1024, gpu_memory_utilization=0.85,
                                         n_gpus=1, n_warmup_samples=5,
                                         print_output=False, enable_expert_parallel=False):
    server_process = None
    port = 8000
    results = None
    try:
        # Start server
        server_process = start_vllm_server(model, port=port, seed=seed,
                                           max_model_len=max_model_len, 
                                           gpu_memory_utilization=gpu_memory_utilization,
                                           n_gpus=n_gpus, enable_expert_parallel=enable_expert_parallel)

        # Start client
        client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    
        # Run warmup
        await run_batch(client, model, prompts[:n_warmup_samples],
                        seed=seed, print_output=False, max_new_tokens=max_new_tokens,
                        prompt_formatted=True)
        
        # Run experiment
        results = await run_batch(client, model, prompts,
                                  seed=seed, print_output=print_output, max_new_tokens=max_new_tokens,
                                  prompt_formatted=True)

    except Exception as e:
        print(f"An error occurred during inference: {e}")

    finally:
        # Tear down server
        if server_process is not None:
            stop_vllm_server(server_process)

    return results

def plot_hist_tpots(overall_results, subject_results=None, x_limit=50, title=None):
    
    plt.figure(figsize=(10,8))
    
    # Convert all TPOT to ms
    overall_tpots = [r['tpot']*1000 for r in overall_results]
    overall_avg_tpot = np.mean(overall_tpots)
    overall_total_requests = len(overall_tpots)
    all_tpots = []
    all_tpots += overall_tpots
    if subject_results is not None:
        n_subjects = len(subject_results)
        subject_tpots = {s:[r['tpot']*1000 for r in subject_results[s]] for s in subject_results}
        subject_avg_tpots = {s:np.mean(subject_tpots[s]) for s in subject_tpots}
        subject_total_requests = {s:len(subject_tpots[s]) for s in subject_tpots}
        for s in subject_tpots:
            all_tpots += subject_tpots[s]
    
    # Create uniform bins for histograms
    n_bins = 100
    min_tpot = min(all_tpots)
    max_tpot = min(max(all_tpots), x_limit) # Limit right side of plot
    bin_width_ms = (max_tpot - min_tpot) / n_bins
    fixed_bins = np.arange(min_tpot, max_tpot + bin_width_ms, bin_width_ms)
    
    # Plot subjects in blue...
    result_idx = 0
    blue_colors = plt.get_cmap('Blues')(np.linspace(0.4, 1.0, n_subjects))
    for s in subject_tpots:
        plt.hist(list(subject_tpots[s]), bins=fixed_bins,
                 color=blue_colors[result_idx],
                 label=f'\'{s}\' ({subject_total_requests[s]} requests)')
        result_idx += 1
    
    # ...then superimpose overall in red
    plt.hist(list(overall_tpots), bins=fixed_bins,
             color='r',
             label=f'prompts drawn from all subjects ({overall_total_requests} requests)')
    
    if title:
        plt.title(title, fontsize=14, pad=15)
    else:
        plt.title(f'Distribution of per-request TPOT for MMLU Requests', fontsize=14, pad=15)
    plt.xlabel('TPOT (ms)', fontsize=12)
    plt.ylabel('Frequency (Number of Requests)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()