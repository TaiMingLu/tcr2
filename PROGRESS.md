# Progress

## What Works

- Data generation for all 7 reasoning hop generalization tasks (Parity-NL, LLC, MDM, MOAS, CLF, ObjC, NumS)
- Baseline LLM generation with CoT prompting (Qwen2.5-1.5B-Instruct from shared models)
- TCR method implementation with entropy-based error detection and head knockout
- Evaluation pipeline with scoring/reference.json
- All scripts (evaluate, reproduce, method, baseline, download)
- Container environment setup (nvidia/cuda:12.1.0-runtime-ubuntu22.04 base for GPU support)
- Git history rewritten: removed pylibs (4.6GB) and trajectory files

## Results

Pending GPU evaluation. Job submitted to cluster.

## Remaining

- GPU job execution on compute cluster
- Compare baseline vs TCR accuracy on all 7 tasks
- Train head selector classifier (optional, for full TCR)

## Issues

- CPU-only login node: generation is slow on CPU, GPU job needed
- Qwen2.5-1.5B has 12 attention heads (vs 32 in paper's 7B model), ep heads scaled accordingly
- TCR-gold implementation is simplified (not doing full oracle knockout yet)

## Deviations from Paper

### Model Size
- **Paper uses**: Qwen2.5-7B-Instruct (32 heads per layer)
- **We use**: Qwen2.5-1.5B-Instruct (28 layers, 12 heads per layer)
- **Why**: 7B model not available in shared cache; 1.5B is the largest Qwen2.5 available
- **Impact**: Ep head indices scaled proportionally; results may differ from paper

### Ep Head Candidates
- **Paper**: Identified via mechanistic analysis with full head knockout experiments
- **We use**: Scaled candidate heads from paper's 7B model to 1.5B architecture
- **Impact**: May not perfectly match the actual ep heads in 1.5B model

### TCR Implementation
- **Paper**: Uses trained head selector (Qwen2.5-0.5B fine-tuned) + majority voting
- **We use**: Simplified entropy-based detection with greedy token selection
- **Why**: Training head selector requires ~20K labeled samples and LoRA fine-tuning
- **Impact**: Simplified TCR may show smaller improvements than paper's full method
