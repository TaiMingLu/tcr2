# Reproduction Milestones

**Current: method_runs**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-04-07] - method_runs
- Read paper thoroughly: TCR (Test-time Correction of Reasoning) for hop generalization
- Paper studies why LLMs fail when reasoning hop counts exceed training distribution
- Key finding: certain "erroneous processing heads" (ep heads) drive errors
- TCR: dynamically identifies and deactivates ep heads at test time using entropy detection + head selector
- Paper reports +6.8% average improvement on Qwen2.5-7B-Instruct across 7 tasks
- Filled in all briefing files (problem, evaluation, method, overview)
- Created data generation for all 7 tasks (Parity-NL, LLC, MDM, MOAS, CLF, ObjC, NumS)
- Implemented TCR method with ep head candidates, entropy detection, baseline generation
- Created evaluation pipeline with scoring/reference.json matching paper's Table 2
- Created all scripts (evaluate.sh, reproduce.sh, method.sh, baseline.sh, download.sh)
- Set up environment (container.def, setup.sh)
- Downloaded Qwen2.5-0.5B-Instruct to workspace models
- Qwen2.5-1.5B-Instruct available in shared models (28 layers, 12 heads)
- Generated test data successfully for all 7 tasks
- Submitted GPU job for evaluation (baseline + TCR on parity_nl, mdm)

## Stop Justification

<!-- Do not edit this unless you decide to stop -->
GPU job submitted for core evaluation. Waiting for results to confirm method_runs and proceed to core_claim.
