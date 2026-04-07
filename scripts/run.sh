#!/bin/bash
# GPU job script for TCR evaluation.
#
# This runs inside the compute container with GPU access.
# PYTHONPATH must include the pylibs directory.

set -e

cd /home/user

export PYTHONPATH="/home/user/pylibs:/home/user:$PYTHONPATH"
export HF_HOME="/home/user/shared/hf_cache"
export TRANSFORMERS_CACHE="/home/user/shared/hf_cache"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
SEED="${SEED:-42}"
RESULTS_DIR="/home/user/results"

mkdir -p "$RESULTS_DIR"
mkdir -p "/home/user/data"

echo "=== TCR Evaluation ==="
echo "Model: $MODEL_NAME"
echo "Samples: $NUM_SAMPLES"
echo "Seed: $SEED"

# ============================================================
# Generate test data
# ============================================================
echo ""
echo "=== Generating test data ==="
python3 -c "
import sys
sys.path.insert(0, '/home/user')
from data.task_generator import generate_dataset, save_dataset

tasks = [
    ('parity_nl', {'hop_count': 50}, 'parity_nl_50'),
    ('llc', {'word_count': 6}, 'llc_6'),
    ('mdm', {'digits_a': 3, 'digits_b': 6}, 'mdm_3x6'),
    ('moas', {'operand_count': 50}, 'moas_50'),
    ('clf', {'seq_length': 30}, 'clf_30'),
    ('objc', {'object_count': 30}, 'objc_30'),
    ('nums', {'student_count': 10}, 'nums_10'),
]

for task_name, params, fname in tasks:
    instances = generate_dataset(task_name, $NUM_SAMPLES, seed=$SEED, **params)
    save_dataset(instances, f'/home/user/data/test_{fname}.jsonl')
    print(f'Generated {len(instances)} for {task_name}')
"

# ============================================================
# Run baseline for all tasks
# ============================================================
echo ""
echo "=== Running baseline evaluation ==="
for TASK_CONFIG in "parity_nl:50:parity_nl_50" "llc:6:llc_6" "mdm:6:mdm_3x6" "moas:50:moas_50" "clf:30:clf_30" "objc:30:objc_30" "nums:10:nums_10"; do
    TASK="${TASK_CONFIG%%:*}"
    REMAIN="${TASK_CONFIG#*:}"
    HOPS="${REMAIN%%:*}"
    FNAME="${REMAIN#*:}"

    echo "  Baseline: $TASK..."
    python3 -m method.inference \
        --task "$TASK" \
        --hop_count "$HOPS" \
        --num_samples $NUM_SAMPLES \
        --model_name "$MODEL_NAME" \
        --method baseline \
        --output_dir "$RESULTS_DIR" \
        --seed $SEED
done

# ============================================================
# Run TCR for core tasks (parity_nl and mdm - main paper results)
# ============================================================
echo ""
echo "=== Running TCR method ==="
python3 -m method.inference \
    --task parity_nl \
    --hop_count 50 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method tcr \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

python3 -m method.inference \
    --task mdm \
    --hop_count 6 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method tcr \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

# ============================================================
# Evaluate and write scores
# ============================================================
echo ""
echo "=== Evaluating results ==="
python3 -m eval.evaluate \
    --results_dir "$RESULTS_DIR" \
    --output /home/user/scoring/scores.json \
    --model_name "$MODEL_NAME"

echo ""
echo "=== Done! ==="
cat /home/user/scoring/scores.json
