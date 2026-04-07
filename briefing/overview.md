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
