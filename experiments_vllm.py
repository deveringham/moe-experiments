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

# Spins up the vLLM server as a subprocess and blocks until ready.
def start_vllm_server(model_name, port=8000, seed=0, max_model_len=1024, gpu_memory_utilization=0.85, n_gpus=1):
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
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "1",
        "--enable-expert-parallel",
        "--seed", str(seed),
        #"--enable-eplb"
    ]
    
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
async def measure_request(client, model, prompt_idx, prompt, seed=0, max_new_tokens=100, get_response=True):
    
    start_time = time.perf_counter()
    first_token_time = None
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
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
async def run_batch(client, model, prompts, seed=0, print_output=False, max_new_tokens=100):

    print(f"Sending batch of {len(prompts)} concurrent requests...")
    
    batch_start_time = time.perf_counter()
    
    # Fire all requests concurrently
    tasks = [measure_request(client, model, i, prompt, seed=seed, max_new_tokens=max_new_tokens) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    batch_end_time = time.perf_counter()
    total_batch_time = batch_end_time - batch_start_time
    
    if print_output:
        print("\n--- Per-Request Metrics ---")
        total_tpot = 0
        total_tokens = 0
        valid_requests = 0

        for res in results:
            print(f"Request {res['prompt_id']}: TTFT = {res['ttft']:.4f}s | "
                  f"TPOT = {res['tpot']*1000:.2f}ms | Tokens = {res['num_output_tokens']}")

            if res['num_output_tokens'] > 1:
                total_tpot += res['tpot']
                total_tokens += res['num_output_tokens']
                valid_requests += 1

        print("\n--- Batch Metrics ---")
        if valid_requests > 0:
            avg_tpot = total_tpot / valid_requests
            print(f"Average Per-Request TPOT: {avg_tpot * 1000:.2f} ms/token")

        throughput = total_tokens / total_batch_time
        print(f"Total Batch Time: {total_batch_time:.2f}s")
        print(f"Total Tokens Generated: {total_tokens}")
        print(f"Overall Server Throughput: {throughput:.2f} tokens/second")
    
    return results

# Run full experiment:
# - Start vLLM server
# - Start vLLM client
# - Run inference
# - Return timing measurements
async def run_experiment_vllm_throughput(model, prompts, seed=0, max_new_tokens=100,
                                         max_model_len=1024, gpu_memory_utilization=0.85,
                                         n_gpus=1, print_output=True):
    server_process = None
    port = 8000
    results = None
    try:
        # Start server
        server_process = start_vllm_server(model, port=port, seed=seed,
                                           max_model_len=max_model_len, 
                                           gpu_memory_utilization=gpu_memory_utilization, n_gpus=n_gpus)

        # Start client
        client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")

        # Run experiment
        results = await run_batch(client, model, prompts, seed=seed, print_output=print_output, max_new_tokens=max_new_tokens)

    except Exception as e:
        print(f"An error occurred during inference: {e}")

    finally:
        # Tear down server
        if server_process is not None:
            stop_vllm_server(server_process)

    return results