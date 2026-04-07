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
