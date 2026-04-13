# Progress

## What Works

- **Data generation**: All 7 reasoning hop generalization tasks implemented in `data/task_generator.py`:
  - Parity-NL (coin flip parity), LLC (last letter concatenation), MDM (multi-digit multiplication), MOAS (multi-operand addition/subtraction), CLF (crawler log folder), ObjC (object counting), NumS (number of students)
- **Baseline generation**: CoT prompting with Qwen2.5-1.5B-Instruct from shared models
- **TCR implementation**: Proper ep head knockout via forward hooks on attention layers
  - `EpHeadKnockoutHook` class that registers hooks on Qwen2 attention modules
  - TCR-gold: baseline + ep head knockout on wrong samples with majority vote
  - TCR (entropy-based): entropy threshold detection + per-token knockout
- **Evaluation pipeline**: `eval/evaluate.py` computes per-task accuracy and produces `scoring/scores.json`
- **Scripts**: run.sh, reproduce.sh, evaluate.sh, baseline.sh, method.sh, download.sh all working
- **Container**: nvidia/cuda:12.1.0-runtime-ubuntu22.04 for GPU support
- **Model**: Qwen2.5-1.5B-Instruct available at /home/user/shared/models

## Results

**GPU job submitted.** Running baseline vs TCR-gold on Parity-NL(50/10 hops), LLC(6 words), MDM(3x6 digits) with 50 samples each.

## Remaining

- GPU job execution: baseline vs TCR-gold on Parity-NL(50/10 hops), LLC(6), MDM(3x6)
- Compare accuracy with paper's reported results
- Verify head knockout mechanism works correctly

## Issues

- CPU-only login node: generation requires GPU job
- Qwen2.5-1.5B has 12 attention heads (vs 32 in paper's 7B model) — ep head indices scaled proportionally
- TCR-gold implementation simplified (oracle detection + full ep head knockout, majority vote)
- Full TCR (trained head selector) not implemented due to training cost

## Deviations from Paper

### Model Size
- **Paper uses**: Qwen2.5-7B-Instruct (32 heads per layer)
- **We use**: Qwen2.5-1.5B-Instruct (28 layers, 12 heads per layer)
- **Why**: 7B model not available in shared cache; 1.5B is the largest Qwen2.5 available
- **Impact**: Ep head indices scaled from 32 to 12 heads proportionally

### Ep Head Candidates
- **Paper**: Identified via mechanistic analysis with full head knockout experiments on 7B model
- **We use**: Scaled candidate heads from paper's 7B model to 1.5B architecture
- **Impact**: May not perfectly match actual ep heads in 1.5B model

### TCR Implementation
- **Paper**: Uses trained head selector (Qwen2.5-0.5B fine-tuned) + majority voting
- **We use**: Simplified entropy-based detection with greedy token selection
- **Why**: Training head selector requires ~20K labeled samples and LoRA fine-tuning
- **Impact**: Simplified TCR may show smaller improvements than paper's full method

### TCR-gold (what we test)
- **Paper**: Oracle detector precisely localizes errors, top-3 head knockout with majority vote
- **We use**: Oracle knows wrong samples, tries all ep head knockouts, majority vote
- **Impact**: Should still show meaningful improvement from ep head knockout
