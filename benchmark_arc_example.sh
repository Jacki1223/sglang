#!/bin/bash
# ARC Cache Benchmark Example Script
# This script demonstrates how to run benchmark_offline_throughput with different eviction policies

set -e

MODEL_PATH=${1:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
NUM_PROMPTS=${2:-100}
OUTPUT_FILE=${3:-"arc_benchmark_results.jsonl"}

echo "=========================================="
echo "ARC Cache Benchmark Comparison"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Number of prompts: $NUM_PROMPTS"
echo "Output file: $OUTPUT_FILE"
echo ""

# Clean up previous results
rm -f $OUTPUT_FILE

echo "=========================================="
echo "Test 1: LRU Policy (Baseline)"
echo "=========================================="
python -m sglang.bench_offline_throughput \
    --model-path $MODEL_PATH \
    --num-prompts $NUM_PROMPTS \
    --radix-eviction-policy lru \
    --result-filename $OUTPUT_FILE \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 16 \
    --gsp-prompts-per-group 8 \
    --gsp-system-prompt-len 2048 \
    --gsp-question-len 128 \
    --gsp-output-len 256

echo ""
echo "=========================================="
echo "Test 2: LFU Policy"
echo "=========================================="
python -m sglang.bench_offline_throughput \
    --model-path $MODEL_PATH \
    --num-prompts $NUM_PROMPTS \
    --radix-eviction-policy lfu \
    --result-filename $OUTPUT_FILE \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 16 \
    --gsp-prompts-per-group 8 \
    --gsp-system-prompt-len 2048 \
    --gsp-question-len 128 \
    --gsp-output-len 256

echo ""
echo "=========================================="
echo "Test 3: ARC Policy (Adaptive)"
echo "=========================================="
python -m sglang.bench_offline_throughput \
    --model-path $MODEL_PATH \
    --num-prompts $NUM_PROMPTS \
    --radix-eviction-policy arc \
    --result-filename $OUTPUT_FILE \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 16 \
    --gsp-prompts-per-group 8 \
    --gsp-system-prompt-len 2048 \
    --gsp-question-len 128 \
    --gsp-output-len 256

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Analyzing results..."
python -c "
import json

results = []
with open('$OUTPUT_FILE', 'r') as f:
    for line in f:
        results.append(json.loads(line))

# Find the policy name based on order
policies = ['LRU', 'LFU', 'ARC']

print('')
print('=' * 80)
print('Performance Comparison Summary')
print('=' * 80)
print(f\"{'Policy':<10} {'Total Throughput (tok/s)':<30} {'Output Throughput (tok/s)':<30}\")
print('-' * 80)

for i, result in enumerate(results):
    policy = policies[i] if i < len(policies) else f'Test {i+1}'
    print(f\"{policy:<10} {result['total_throughput']:<30.2f} {result['output_throughput']:<30.2f}\")

print('=' * 80)

# Find best policy
best_idx = max(range(len(results)), key=lambda i: results[i]['total_throughput'])
best_policy = policies[best_idx] if best_idx < len(policies) else f'Test {best_idx+1}'
improvement = (results[best_idx]['total_throughput'] / results[0]['total_throughput'] - 1) * 100

print(f\"\\nBest policy: {best_policy}\")
print(f\"Improvement over LRU: {improvement:.2f}%\")
print('')
"

echo "Done!"
