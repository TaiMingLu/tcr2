#!/bin/bash
# End-to-end reproduction of the paper's results.
#
# This script runs inside the compute container and reproduces the paper's
# main claim: TCR improves reasoning hop generalization accuracy.
#
# Assumes:
#   - Container is built
#   - Data and models are downloaded via scripts/download.sh
#   - Working directory is /home/user
#
# Usage:
#   bash scripts/reproduce.sh

set -e

cd /home/user

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"

echo "============================================"
echo "TCR Paper Reproduction"
echo "Model: $MODEL_NAME"
echo "Samples per task: $NUM_SAMPLES"
echo "============================================"

# Step 1: Baseline evaluation
echo ""
echo "=== Step 1: Baseline (no intervention) ==="
bash scripts/baseline.sh

# Step 2: TCR method
echo ""
echo "=== Step 2: TCR Method ==="
bash scripts/method.sh

# Step 3: Evaluate and compare
echo ""
echo "=== Step 3: Final Evaluation ==="
bash scripts/evaluate.sh

echo ""
echo "============================================"
echo "Reproduction complete!"
echo "See scoring/scores.json for results"
echo "See scoring/reference.json for paper's numbers"
echo "============================================"
