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
