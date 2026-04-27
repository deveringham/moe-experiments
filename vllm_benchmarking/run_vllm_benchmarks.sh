#!/bin/bash

# Configuration
# For full-scale tests use something like: n_prompts 1000, req_rate 20
#MODEL="Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
#MODEL="Qwen/Qwen1.5-MoE-A2.7B-Chat"
#MODEL="ModelCloud/DeepSeek-V2-Lite-gptq-4bit"
#MODEL="TechxGenus/DeepSeek-V2-Lite-Chat-AWQ"
#MODEL="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
MODEL="microsoft/Phi-mini-MoE-instruct"
NUM_PROMPTS=10
NUM_WARMUPS=2
REQUEST_RATE=10
PORT=8000
SEED=43

# Get data with
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"

# Start server, run benchmark, teardown
run_benchmark() {
    local num_npus=$1
    local scheme_name=$2
    local serve_args=$3
    
    local metrics_file="metrics_${num_npus}npu_${scheme_name}.json"
    local server_log="server_${num_npus}npu_${NUM_PROMPTS}prompts_${REQUEST_RATE}reqrate_${scheme_name}.log"

    echo "==========================================================="
    echo "Starting Run: ${num_npus} NPUs | Parallel Scheme: ${scheme_name}"
    echo "Metrics will be saved to: ${metrics_file}"
    echo "==========================================================="

    # Start the vLLM server in the background
    echo "-> Booting up vLLM server..."
    vllm serve $MODEL \
        --dtype auto \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 16 \
        $serve_args > "$server_log" 2>&1 &
    
    SERVER_PID=$!

    # Wait for the server to be ready
    echo "-> Waiting for server to initialize..."
    while true; do
        # Look for 200 OK
        echo "-> Pinging..."
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)
        if [ "$HTTP_STATUS" -eq 200 ]; then
            echo "-> Server is online and accepting requests."
            break
        fi
        
        # Check if the server crashed during startup
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "-> ERROR: Server crashed. Check $server_log for details."
            return 1
        fi
        sleep 5
    done

    # Run the benchmark client
    echo "-> Running benchmark..."
    vllm bench serve \
        --backend vllm \
        --model $MODEL \
        --dataset-name random \
        --num-prompts $NUM_PROMPTS \
        --request-rate $REQUEST_RATE \
        --result-filename "$metrics_file" \
        --save-result \
        --seed $SEED \
        --num-warmups $NUM_WARMUPS \
        --dataset-path $DATASET \

    # Teardown and VRAM flush
    echo "-> Tearing down server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    sleep 2
    kill $SERVER_PID 2>/dev/null
    
    # Give the OS a moment to fully release the GPU VRAM
    echo "-> Flushing VRAM..."
    sleep 10
    echo "Run complete."
    echo ""
}

# Benchmark matrix
 Format: run_benchmark <GPUs/NPUs> <Scheme_Name> "<vLLM_Args>"

# 1 GPU/NPU (Baseline - Parallel schemes don't apply to a single device)
run_benchmark 1 "baseline" "--tensor-parallel-size 1 --data-parallel-size 1"

# 2 GPUs/NPUs
run_benchmark 2 "tensor"    "--tensor-parallel-size 2 --data-parallel-size 1"
run_benchmark 2 "data"      "--tensor-parallel-size 1 --data-parallel-size 2"
run_benchmark 2 "expert"    "--tensor-parallel-size 2 --data-parallel-size 1 --enable-expert-parallel"
run_benchmark 2 "expert_lb" "--tensor-parallel-size 2 --data-parallel-size 1 --enable-expert-parallel --enable-eplb"

# 4 GPUs/NPUs
run_benchmark 4 "tensor"    "--tensor-parallel-size 4 --data-parallel-size 1"
run_benchmark 4 "data"      "--tensor-parallel-size 1 --data-parallel-size 4"
run_benchmark 4 "expert"    "--tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel"
run_benchmark 4 "expert_lb" "--tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --enable-eplb"

echo "==========================================================="
echo "All benchmark runs complete."
echo "Check the generated metrics_*.json files."