#!/bin/bash
# CPU test job - tests the pipeline with tiny samples to verify everything works.
# This avoids GPU for quick validation.
# GPU jobs can be submitted separately once the container builds.

set -e

cd /home/user

export PYTHONPATH="/home/user/pylibs:/home/user:$PYTHONPATH"
export HF_HOME="/home/user/shared/hf_cache"
export TRANSFORMERS_CACHE="/home/user/shared/hf_cache"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SEED="${SEED:-42}"
RESULTS_DIR="/home/user/results"

mkdir -p "$RESULTS_DIR"

echo "=== TCR Pipeline Test (CPU, $NUM_SAMPLES samples) ==="
echo "Model: $MODEL_NAME"

# Test 1: Data generation
echo ""
echo "=== Test 1: Data generation ==="
python3 -c "
import sys
sys.path.insert(0, '/home/user')
from data.task_generator import generate_dataset, save_dataset

tasks = [
    ('parity_nl', {'hop_count': 10}, 'parity_nl_10'),
    ('llc', {'word_count': 4}, 'llc_4'),
    ('mdm', {'digits_a': 2, 'digits_b': 2}, 'mdm_2x2'),
]
for task_name, params, fname in tasks:
    instances = generate_dataset(task_name, 3, seed=42, **params)
    save_dataset(instances, f'/home/user/data/test_{fname}.jsonl')
    print(f'  Generated {len(instances)} for {task_name}')
print('Data generation: OK')
"

# Test 2: Import and basic model loading
echo ""
echo "=== Test 2: Model loading ==="
python3 -c "
import sys, torch
sys.path.insert(0, '/home/user')
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f'  PyTorch CUDA: {torch.cuda.is_available()}')
print(f'  PyTorch device count: {torch.cuda.device_count()}')
model_path = '/home/user/shared/models/Qwen2.5-1.5B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map='cpu')
print(f'  Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params')
print('Model loading: OK')
" 2>&1 | tail -5

# Test 3: Baseline generation (tiny sample)
echo ""
echo "=== Test 3: Baseline generation (1 sample) ==="
python3 -m method.inference \
    --task parity_nl \
    --hop_count 10 \
    --num_samples 1 \
    --model_name "$MODEL_NAME" \
    --method baseline \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED 2>&1 | tail -10

# Test 4: Evaluation
echo ""
echo "=== Test 4: Evaluation ==="
python3 -m eval.evaluate \
    --results_dir "$RESULTS_DIR" \
    --output /home/user/scoring/scores.json \
    --model_name "$MODEL_NAME" 2>&1 | tail -5

echo ""
echo "=== Pipeline test complete ==="
cat /home/user/scoring/scores.json 2>/dev/null || echo "No scores yet"
