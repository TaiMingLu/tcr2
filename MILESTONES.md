# Reproduction Milestones

**Current: method_runs**

## Progress Log

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
- All scripts (evaluate, reproduce, method, baseline, download)
- Container environment setup

### [2026-04-07] - none
- Just started reproduction work
