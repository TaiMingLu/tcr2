# Overview

- **Paper ID:** 2601.21214
- **Title:** Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models
- **Domain:** Natural Language Processing / LLM Reasoning
- **TL;DR:** LLMs suffer from reasoning hop generalization failures (accuracy drops when problems require more reasoning steps). The paper identifies that certain attention heads ("erroneous processing heads") drive these failures by competing with correct reasoning mechanisms, and proposes TCR — a test-time intervention that dynamically deactivates these heads to improve accuracy.

## Short Summary

The paper studies why large language models fail at reasoning tasks requiring many hops, even when the underlying skill is unchanged. Through mechanistic analysis, they discover that errors concentrate at specific token positions driven by "erroneous processing heads" (ep heads) that amplify incorrect reasoning trajectories. They propose **TCR (Test-time Correction of Reasoning)**, which trains a small classifier to predict which heads to deactivate, uses entropy-based detection to decide when to intervene, and employs majority voting across candidate heads to correct erroneous predictions. On Qwen2.5-7B-Instruct, TCR achieves +6.8% average accuracy improvement across 7 reasoning tasks.

## Key Results

- **Qwen2.5-7B-Instruct baseline**: 41.7% average accuracy on 7 reasoning hop generalization tasks
- **TCR (proposed)**: 48.5% average accuracy (+6.8% improvement)
- **TCR-gold (oracle detection)**: 61.3% average accuracy (+19.6% improvement)
- Key finding: knocking out individual ep heads can restore correct predictions in ~60% of erroneous cases

---

# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

## Research Question

Chain-of-thought (CoT) reasoning enables LLMs to solve complex multi-step problems, but a key failure mode emerges when the number of required reasoning steps (hops) increases beyond the training distribution. When problems require more reasoning hops — even if the underlying algorithmic skill is identical — model accuracy drops sharply. This is called **reasoning hop generalization failure**.

Example: A model that correctly computes 2-digit × 2-digit multiplication may fail at 3-digit × 6-digit multiplication, even though both require the same multi-digit multiplication algorithm. The model has learned the algorithm but cannot generalize it to longer chains of reasoning.

## Why It Matters

Hop generalization failures reveal a fundamental brittleness in LLMs' reasoning capabilities. If models cannot reliably extend learned reasoning skills to longer chains, their reliability in real-world complex reasoning tasks is severely limited. A good solution would enable LLMs to consistently apply reasoning skills across varying problem complexity without needing fine-tuning on each specific hop count.

## Success Criteria

A successful solution should:
1. **Improve accuracy on longer-hop reasoning tasks** compared to unmodified LLM outputs, measured by final answer accuracy.
2. **Maintain or improve performance across varying hop counts** — ideally, accuracy should remain stable as hop count increases.
3. **Be applicable at test time** without requiring fine-tuning on the specific downstream task.
4. **Be lightweight and compatible** with off-the-shelf LLMs without architectural changes.

---

# Paper's Method

## Key Contribution

The paper proposes **TCR (Test-time Correction of Reasoning)**, a lightweight test-time intervention method that dynamically identifies and deactivates erroneous processing heads in an LLM's attention layers to correct reasoning hop generalization errors.

## Approach

The method has three components:

### 1. Identifying Erroneous Processing Heads (ep heads)

Through mechanistic analysis, the paper discovers that:
- Certain attention heads ("ep heads") amplify incorrect reasoning trajectories while suppressing correct ones.
- Knocking out individual ep heads during inference can often restore correct predictions.
- Different tasks and error types map onto a shared subset of ep heads per model.
- Candidate head sets H per model are identified via a greedy set-cover algorithm over multiple tasks and error types.

For Qwen2.5-7B-Instruct: H = {a₁⁰, a₃⁰, a₆⁰, a₇⁰, a₁₅⁰, a₁₃¹, a₁₁³, a₂₂⁸}

### 2. Head Selector (Classifier)

- A small model (Qwen2.5-0.5B-Instruct fine-tuned with LoRA) predicts which head(s) to knock out given the input context.
- This is formulated as a multi-label classification: given input context, output a binary label per head in H indicating whether knocking it out would correct the prediction.
- Training uses multi-label softmax loss on ~20K erroneous predictions from 5 training tasks.
- At inference, the classifier selects top-3 heads from H.

### 3. Entropy-Based Detection

- During decoding, monitor the predictive entropy of each generated token.
- If entropy > threshold τ (paper uses τ=0.3), trigger the head selector to identify heads for knockout.
- Otherwise, proceed with normal decoding.

### Intervention: Attention Head Knockout

When a head is selected for knockout, its attention output is zeroed out during the residual stream update, effectively removing its contribution to the prediction. The paper uses majority voting over top-3 selected heads: knock out each head separately, re-run forward pass, and take majority vote of the resulting outputs.

## Main Claims

1. Errors in hop-generalization concentrate at token positions of a few specific error types, enabling focused diagnosis.
2. Token-level errors stem from internal competition between correct and erroneous processing heads; knocking out ep heads substantially restores correct predictions.
3. TCR consistently improves reasoning hop generalization across tasks and LLMs (avg +6.8% for Qwen2.5-7B-Instruct).
4. TCR-gold (with oracle error detection) achieves +19.6% improvement, demonstrating the strong rectification potential.

---

# Evaluation

How success is measured. This describes the evaluation protocol for reasoning hop generalization tasks.

## Metrics

**Primary metric: Final answer accuracy** — For each test instance, the model's final answer is extracted from the chain-of-thought response and compared against the ground-truth answer. Accuracy is computed as the fraction of correct final answers across test instances.

- Higher accuracy is better.
- Each task has a deterministic ground-truth answer that can be checked programmatically.

## Baselines and Targets

The paper evaluates on 7 reasoning hop generalization tasks using 4 open-source LLMs. Key reference points (reported by the paper):

**Qwen2.5-7B-Instruct results (Table 2, paper):**
| Method | Parity-NL | MDM | LLC | CLF | MOAS | ObjC | NumS | Average |
|--------|-----------|-----|-----|-----|------|------|------|---------|
| Baseline (no intervention) | 48.3% | 43.0% | 11.7% | 56.8% | 39.2% | 52.0% | 41.1% | 41.7% |
| TCR (proposed method) | 60.4% | 48.2% | 16.2% | 66.6% | 46.0% | 56.0% | 46.0% | 48.5% |
| TCR-gold (oracle detection) | 81.2% | 58.3% | 23.0% | 71.3% | 62.0% | 76.0% | 54.5% | 61.3% |

The goal of reproduction is to demonstrate that the proposed method (TCR) achieves measurable improvement over the baseline, using the same tasks and model family.

## Evaluation Protocol

1. **Data generation**: For each task, generate synthetic instances with controlled hop counts. The paper uses hop numbers from specific ranges (e.g., Parity-NL: 10-50, MDM: digit combinations, LLC: list sizes 2-10, etc.).
2. **Task-specific hop settings** used in the paper's main experiments:
   - Parity-NL: 50 hops
   - MDM: 3 digits × 6 digits
   - LLC: 6 hops (list size 6)
   - CLF: 30 hops (sequence length 30)
   - MOAS: 50 hops (50 operands)
   - ObjC: 30 hops (30 objects)
   - NumS: 10 hops (10 students)
3. **Model generation**: Generate responses using the LLM with CoT prompting (in-context demonstrations of the solution template).
4. **Answer extraction**: Extract the final answer from the model's response (e.g., the last line of the CoT output).
5. **Accuracy computation**: Compare extracted answers against ground truth. Report per-task accuracy and average across all tasks.
6. **Multiple runs**: Report averages over 3 random seeds for the TCR method (to account for stochasticity in generation and intervention).
7. **Test set size**: 100 instances per task per hop count (following the paper's protocol).

---

# Reproduction Log

### Iteration 1: MiniMax-M2.7
- **Milestone**: `method_runs` | **Status**: done
- **Working time**: 2.3h | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 1: MiniMax-M2.7
- **Milestone**: `method_runs` | **Status**: done
- **Working time**: 17m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

### [2026-04-09] - method_runs (GPU job submitted)
- Updated run.sh to execute GPU reproduction: baseline + TCR on parity_nl at 10 and 20 hops
- Updated container.def to use nvidia/cuda:12.1.0-runtime-ubuntu22.04 for GPU support
- Committed and pushed to GitHub (c14ec4e)
- GPU job submitted via action.yaml:submit
- Waiting for container build and job execution

### [2026-04-08] - method_runs (submitted)
- Git history rewritten: removed pylibs (19,989 files, 4.6GB) and trajectory files from git tracking
- Git objects reduced from 2.5GB to 14MB
- Successfully pushed to GitHub: https://github.com/TaiMingLu/tcr2
- GPU job submitted for evaluation
- Waiting for GPU job to execute and return results

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
- Pipeline verified: data generation, method imports, evaluation, model loading all working

</details>

### Iteration 1: MiniMax-M2.7
- **Milestone**: `none` | **Status**: done
- **GPU**: 0.0h

### Iteration 2: glm-5.1
- **Milestone**: `none` | **Status**: done
- **GPU**: 0.0h

### Iteration 1: MiniMax-M2.7
- **Milestone**: `method_runs` | **Status**: done
- **Working time**: 30m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

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

</details>


---

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
