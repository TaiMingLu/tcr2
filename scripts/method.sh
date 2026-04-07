#!/bin/bash
# Run the paper's TCR method.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
SEED="${SEED:-42}"
RESULTS_DIR="/home/user/results"

mkdir -p "$RESULTS_DIR"

echo "=== TCR Method Pipeline ==="
echo "Model: $MODEL_NAME"

# Generate and run core tasks (parity_nl and mdm are the paper's main examples)
python3 -m method.inference \
    --task parity_nl \
    --hop_count 50 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method tcr \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED &

python3 -m method.inference \
    --task mdm \
    --hop_count 6 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method tcr \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED &

wait

echo ""
echo "=== TCR complete ==="
bash scripts/evaluate.sh "$RESULTS_DIR" "$MODEL_NAME"
