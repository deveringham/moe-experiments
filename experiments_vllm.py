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
from openai import AsyncOpenAI

# Sends a single streaming request and measures TTFT and TPOT.
async def measure_request(prompt_idx, prompt):
    
    start_time = time.perf_counter()
    first_token_time = None
    
    # We use stream_options={"include_usage": True} to get the exact token count 
    # in the final chunk of the stream.
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=256
    )

    output_tokens = 0
    
    async for chunk in response:
        # Time of first token (in order to deduct decode form TPOT)
        if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
            first_token_time = time.perf_counter()
            
        # The last chunk when using include_usage=True contains the token stats
        if chunk.usage is not None:
            output_tokens = chunk.usage.completion_tokens

    end_time = time.perf_counter()

    # If generation failed or no tokens produced...
    if first_token_time is None:
        first_token_time = end_time

    # Calculate metrics
    ttft = first_token_time - start_time
    generation_time = end_time - first_token_time
    
    # Subtract 1 from output_tokens because the first token's time is captured in TTFT
    tpot = 0
    if output_tokens > 1:
        tpot = generation_time / (output_tokens - 1)
        
    return {
        "prompt_id": prompt_id,
        "ttft": ttft,
        "tpot": tpot,
        "output_tokens": output_tokens,
        "total_time": end_time - start_time
    }
    
# Runs a batch of prompts concurrently and calculates aggregate metrics.
async def run_batch(prompts, print_output=False):

    print(f"Sending batch of {len(prompts)} concurrent requests...")
    
    batch_start_time = time.per_counter()
    
    # Fire all requests concurrently
    tasks = [measure_request(i, prompt) for i, prompt in enumerate(prompts)]
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
                  f"TPOT = {res['tpot']*1000:.2f}ms | Tokens = {res['output_tokens']}")

            if res['output_tokens'] > 1:
                total_tpot += res['tpot']
                total_tokens += res['output_tokens']
                valid_requests += 1

        print("\n--- Batch Metrics ---")
        if valid_requests > 0:
            avg_tpot = total_tpot / valid_requests
            print(f"Average Per-Request TPOT: {avg_tpot * 1000:.2f} ms/token")

        throughput = total_tokens / total_batch_time
        print(f"Total Batch Time: {total_batch_time:.2f}s")
        print(f"Total Tokens Generated: {total_tokens}")
        print(f"Overall Server Throughput: {throughput:.2f} tokens/second")
