#!/bin/bash
# GPU reproduction job for TCR paper (2601.21214).
# Runs baseline and TCR on key reasoning hop generalization tasks.
# Focus: Parity-NL at 50 hops (paper's main result) + 2 additional tasks.

set -e

cd /home/user

export PYTHONPATH="/home/user/pylibs:/home/user:$PYTHONPATH"
export HF_HOME="/home/user/shared/hf_cache"
export TRANSFORMERS_CACHE="/home/user/shared/hf_cache"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
SEED="${SEED:-42}"
RESULTS_DIR="/home/user/results"

echo "============================================"
echo "TCR Paper Reproduction (GPU)"
echo "Model: $MODEL_NAME"
echo "Samples per task: $NUM_SAMPLES"
echo "============================================"

# Check GPU availability
echo ""
echo "=== GPU Smoke Test ==="
python3 -c "
import sys
sys.path.insert(0, '/home/user/pylibs')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
"

# Smoke test: load model on GPU
echo ""
echo "=== Model Loading Test ==="
python3 -c "
import sys
sys.path.insert(0, '/home/user/pylibs')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = '/home/user/shared/models/Qwen2.5-1.5B-Instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f'Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')
print(f'Device map: {model.hf_device_map}')
del model, tokenizer
import gc; gc.collect(); torch.cuda.empty_cache()
print('Model test: PASSED')
"

mkdir -p "$RESULTS_DIR"

# ============================================================
# Key tasks from the paper:
# 1. Parity-NL 50 hops (paper's main result)
# 2. Parity-NL 10 hops (lower complexity)
# 3. LLC 6 words (moderate complexity)
# 4. MDM 3x6 digits (mathematical reasoning)
# ============================================================

echo ""
echo "=== Baseline: parity_nl (50 hops) ==="
python3 -m method.inference \
    --task parity_nl \
    --hop_count 50 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method baseline \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

echo ""
echo "=== TCR-gold: parity_nl (50 hops) ==="
python3 -m method.inference \
    --task parity_nl \
    --hop_count 50 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method tcr_gold \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

echo ""
echo "=== Baseline: parity_nl (10 hops) ==="
python3 -m method.inference \
    --task parity_nl \
    --hop_count 10 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method baseline \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

echo ""
echo "=== Baseline: LLC (6 words) ==="
python3 -m method.inference \
    --task llc \
    --hop_count 6 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method baseline \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

echo ""
echo "=== TCR-gold: LLC (6 words) ==="
python3 -m method.inference \
    --task llc \
    --hop_count 6 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method tcr_gold \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

echo ""
echo "=== Baseline: MDM (3x6 digits) ==="
python3 -m method.inference \
    --task mdm \
    --hop_count 6 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method baseline \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

echo ""
echo "=== TCR-gold: MDM (3x6 digits) ==="
python3 -m method.inference \
    --task mdm \
    --hop_count 6 \
    --num_samples $NUM_SAMPLES \
    --model_name "$MODEL_NAME" \
    --method tcr_gold \
    --output_dir "$RESULTS_DIR" \
    --seed $SEED

# Evaluate and produce scores
echo ""
echo "=== Final Evaluation ==="
python3 -m eval.evaluate \
    --results_dir "$RESULTS_DIR" \
    --output /home/user/scoring/scores.json \
    --model_name "$MODEL_NAME"

echo ""
echo "============================================"
echo "Reproduction complete!"
echo "Results saved to: $RESULTS_DIR"
echo "Scores written to: /home/user/scoring/scores.json"
echo "============================================"
cat /home/user/scoring/scores.json
echo "============================================"
