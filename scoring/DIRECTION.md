# Research Direction

## Research Problem
LLMs fail at reasoning tasks requiring more hops than seen during training, even when the underlying algorithmic skill is unchanged. The core question is: how can we correct erroneous reasoning inside LLMs at test time, without requiring task-specific fine-tuning?

## Core Contribution
The paper's contribution is a mechanistic understanding that certain attention heads ("erroneous processing heads" or ep heads) amplify incorrect reasoning trajectories. The method — TCR — dynamically identifies and deactivates these heads at test time to correct reasoning. The scientist must work within this mechanistic understanding: improving test-time intervention on internal attention mechanisms.

## Approach Scope
The scientist should focus on:
- Improving ep head identification (better detection, different selection criteria)
- Enhancing error localization (entropy-based or alternative detection methods)
- Better intervention strategies (alternative knockout methods, ensemble strategies)
- Cross-task generalization of ep heads

The scientist may also explore:
- Alternative attention manipulation techniques (activation patching, route-through modification)
- Different uncertainty measures for error detection
- Combining TCR with other test-time methods

## Out of Bounds
- Replacing test-time intervention with fine-tuning or training on downstream tasks
- Using prompting strategies (changing the prompt, adding demonstrations) to improve accuracy — this is a prompt engineering solution, not a mechanistic intervention
- Switching to a non-Qwen model family
- Using external datasets beyond the provided synthetic tasks
- Modifying the model architecture (e.g., changing attention mechanisms)
