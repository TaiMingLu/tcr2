#!/bin/bash
# Run baseline (no intervention) evaluation.
#
# This runs the LLM with standard CoT prompting, no TCR intervention.
# Used as a reference point to compare against the paper's method.
#
# Usage:
#   bash scripts/baseline.sh

set -e

cd /home/user

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"
RESULTS_DIR="/home/user/results"

mkdir -p "$RESULTS_DIR"

echo "=== Baseline Evaluation ==="
echo "Model: $MODEL_NAME"
echo "Samples per task: $NUM_SAMPLES"

# Run baseline for all tasks
for TASK_CONFIG in "parity_nl:50" "llc:6" "mdm:6" "moas:50" "clf:30" "objc:30" "nums:10"; do
    TASK="${TASK_CONFIG%%:*}"
    HOPS="${TASK_CONFIG##*:}"

    echo "Running baseline for $TASK (hops: $HOPS)..."
    python -m method.inference \
        --task "$TASK" \
        --hop_count "$HOPS" \
        --num_samples $NUM_SAMPLES \
        --model_name "$MODEL_NAME" \
        --method baseline \
        --output_dir "$RESULTS_DIR" \
        --seed $SEED
done

echo ""
echo "=== Baseline complete ==="
bash scripts/evaluate.sh "$RESULTS_DIR" "$MODEL_NAME"
