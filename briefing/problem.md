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
