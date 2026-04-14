# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-14] - method_runs
- Workspace audit: implementation is solid and ready for GPU execution
- Fixed scoring/TARGETS.md (was template), scoring/CONSTRAINTS.md (was template), scoring/DIRECTION.md (was template)
- TCR implementation: ep head knockout via forward hooks, correct generation params (do_sample=True, temp=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
- TCR-gold: oracle error detection + majority vote over ep head knockout
- All 7 tasks implemented in data/task_generator.py
- Evaluation pipeline: eval/evaluate.py computes accuracy from result JSONL files
- Scripts: run.sh, evaluate.sh, reproduce.sh, method.sh, baseline.sh all present
- Container: nvidia/cuda:12.1.0-runtime-ubuntu22.04 with setup.sh for packages
- Scoring reference.json filled with paper's Table 2 results
- Briefing files: problem.md, evaluation.md, method.md, overview.md all filled
- Resubmitting GPU job for baseline vs TCR-gold evaluation

### [2026-04-13] - method_runs
- Fixed generation parameters to match paper: do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
- Simplified TCR-gold to use majority vote over ep head knockout responses

### [2026-04-11] - method_runs
- Rewrote inference.py with proper ep head knockout via forward hooks
- Cleaned up tcr_model.py (removed duplicate code)
- Fixed evaluate.py (added missing `import re`)
- Updated container.def to use nvidia/cuda:12.1.0-runtime-ubuntu22.04
- Updated run.sh for comprehensive GPU evaluation
- Updated scoring/reference.json with paper results
- Updated eval/evaluate.py for correct scores.json format

### [2026-04-07] - previous work
- Data generation for all 7 reasoning hop generalization tasks (Parity-NL, LLC, MDM, MOAS, CLF, ObjC, NumS)
- Baseline LLM generation with CoT prompting (Qwen2.5-1.5B-Instruct from shared models)
- Initial TCR method implementation with entropy-based error detection and head knockout
- Evaluation pipeline with scoring/reference.json
- All scripts (evaluate, reproduce, baseline, download)
- Container environment setup
