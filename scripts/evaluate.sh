#!/bin/bash
# Evaluation script — standard way to evaluate work in this environment.
#
# This script evaluates results in the results directory and writes
# scoring/scores.json with the computed metrics.
#
# OUTPUT CONTRACT: Writes scoring/scores.json with per-task and
# aggregate accuracy numbers.
#
# Usage:
#   bash scripts/evaluate.sh                        # use defaults
#   bash scripts/evaluate.sh --results_dir /path    # custom results dir
#   bash scripts/evaluate.sh --model_name Qwen2.5-1.5B-Instruct

set -e

cd /home/user

# Parse arguments
RESULTS_DIR="${1:-/home/user/results}"
MODEL_NAME="${2:-Qwen2.5-1.5B-Instruct}"
OUTPUT_PATH="/home/user/scoring/scores.json"

mkdir -p /home/user/scoring

echo "=== Evaluating results from $RESULTS_DIR ==="

python -m eval.evaluate \
    --results_dir "$RESULTS_DIR" \
    --output "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME"

echo ""
echo "=== Scores written to $OUTPUT_PATH ==="
