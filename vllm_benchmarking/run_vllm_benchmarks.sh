#!/bin/bash

# Configuration
MODEL="Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS=50
REQUEST_RATE=10
PORT=8000

# Function to start server, run benchmark, and teardown
run_benchmark() {
    local num_gpus=$1
    local scheme_name=$2
    local serve_args=$3
    
    local metrics_file="metrics_${num_gpus}gpu_${scheme_name}.json"
    local server_log="server_${num_gpus}gpu_${NUM_PROMPTS}prompts_${scheme_name}.log"

    echo "==========================================================="
    echo "Starting Run: ${num_gpus} GPUs | Scheme: ${scheme_name}"
    echo "Metrics will be saved to: ${metrics_file}"
    echo "==========================================================="

    # Start the vLLM server in the background
    echo "-> Booting up vLLM server..."
    VLLM_USE_V1=0 vllm serve $MODEL \
        --port $PORT \
        --quantization gptq_marlin \
        --dtype auto \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 16 \
        $serve_args > "$server_log" 2>&1 &
    
    SERVER_PID=$!

    # Wait for the server to be ready
    echo "-> Waiting for server to initialize..."
    while true; do
        # Ping the /v1/models endpoint, looking for 200 OK
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
        --result-filename "$metrics_file"
        #--dataset-path $DATASET \

    # Teardown and VRAM flush
    echo "-> Tearing down server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    # Give the OS a moment to fully release the GPU VRAM
    echo "-> Flushing VRAM..."
    sleep 10
    echo "Run complete."
    echo ""
}

# Benchmark matrix
# Format: run_benchmark <GPUs> <Scheme_Name> "<vLLM_Args>"

# 1 GPU (Baseline - Parallel schemes don't apply to a single GPU)
#run_benchmark 1 "baseline" "--tensor-parallel-size 1 --data-parallel-size 1"

# 2 GPUs
#run_benchmark 2 "tensor"    "--tensor-parallel-size 2 --data-parallel-size 1"
#run_benchmark 2 "data"      "--tensor-parallel-size 1 --data-parallel-size 2"
#run_benchmark 2 "expert"    "--tensor-parallel-size 2 --data-parallel-size 1 --enable-expert-parallel"
run_benchmark 2 "expert_lb" "--tensor-parallel-size 2 --data-parallel-size 1 --enable-expert-parallel --enable-eplb"

# 4 GPUs
#run_benchmark 4 "tensor"    "--tensor-parallel-size 4 --data-parallel-size 1"
#run_benchmark 4 "data"      "--tensor-parallel-size 1 --data-parallel-size 4"
#run_benchmark 4 "expert"    "--tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel"
#run_benchmark 4 "expert_lb" "--tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --enable-eplb"

echo "==========================================================="
echo "All benchmark runs completed successfully."
echo "Check the generated metrics_*.json files for offline analysis."