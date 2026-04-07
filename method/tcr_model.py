"""
TCR (Test-time Correction of Reasoning) implementation.

Core components:
1. Attention head knockout mechanism
2. Ep head candidate sets per model
3. Entropy-based detection
4. Head selector classifier (multi-label)
5. TCR inference with majority voting
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


# ============================================================
# Candidate ep head sets per model (from paper Appendix G.1)
# Format: (layer, head_idx) for Qwen-style attention
# ============================================================
EP_HEAD_CANDIDATES = {
    # Qwen2.5-7B-Instruct: 28 layers, 32 heads each
    "Qwen2.5-7B-Instruct": [
        (0, 1), (0, 3), (0, 6), (0, 7), (0, 15),
        (1, 13),
        (3, 11),
        (8, 22),
    ],
    # Qwen2.5-1.5B-Instruct: 28 layers, 12 heads each (GQA: 2 KV heads)
    # Scaled from 7B model (32 heads) to 12 heads
    "Qwen2.5-1.5B-Instruct": [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 6),
        (1, 5),
        (3, 4),
        (8, 8),
    ],
    # Qwen2.5-0.5B-Instruct (for head selector): 24 layers, 14 heads
    "Qwen2.5-0.5B-Instruct": [
        (0, 0), (0, 1), (0, 3), (0, 3), (0, 7),
        (1, 6),
        (3, 5),
        (8, 10),
    ],
    # Phi-3-Instruct
    "Phi-3-mini-4k-instruct": [
        (1, 9), (2, 27), (4, 4), (7, 19), (12, 5), (12, 8),
        (13, 21), (15, 6), (21, 30),
    ],
    # LLaMA3-8B-Instruct
    "Meta-Llama-3-8B-Instruct": [
        (0, 3), (0, 23), (4, 25), (6, 21), (10, 17), (11, 20),
        (11, 12), (13, 19), (13, 18), (17, 24),
    ],
    # Qwen3-8B-Instruct
    "Qwen3-8B": [
        (0, 1), (8, 21), (14, 20), (15, 9), (18, 12), (21, 10),
        (25, 28), (28, 0), (29, 8),
    ],
}


@dataclass
class HeadSelectorOutput:
    """Output from the head selector classifier."""
    # Softmax logits over candidate heads
    logits: torch.Tensor  # shape: [num_heads]
    # Probabilities from softmax
    probs: torch.Tensor   # shape: [num_heads]
    # Selected head indices (top-k)
    selected_heads: List[int]


def compute_predictive_entropy(logits: torch.Tensor) -> float:
    """Compute predictive entropy of token distribution.

    Args:
        logits: Tensor of shape [vocab_size] or [batch, vocab_size]

    Returns:
        Entropy value (scalar)
    """
    probs = F.softmax(logits, dim=-1)
    # Avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean().item()


def get_entropy_threshold(task: str) -> float:
    """Get entropy threshold for error detection.

    Paper uses tau=0.3 for all experiments (Appendix G.3).
    """
    return 0.3


# ============================================================
# Attention Head Knockout
# ============================================================
def knock_out_head_hook(
    layer_idx: int,
    head_idx: int,
    zero_scale: float = 0.0,
) -> callable:
    """Create a hook function to zero out an attention head's output.

    This hook modifies the attention output during the forward pass,
    effectively knocking out the specified attention head.

    Args:
        layer_idx: Layer number
        head_idx: Head index within the layer
        zero_scale: Scale to multiply the head output by (0 = full knockout)

    Returns:
        A hook function
    """
    def hook_fn(module, input, output):
        # output is (attn_output, attn_weights) for standard attention
        # or just attn_output for some implementations
        if isinstance(output, tuple):
            attn_output = output[0]
        else:
            attn_output = output

        if attn_output is None:
            return output

        # attn_output shape: [batch, num_heads, seq_len, head_dim]
        try:
            attn_output[:, head_idx, :, :] *= zero_scale
        except Exception:
            pass

        return output

    return hook_fn


def knock_out_heads_in_forward(
    model,
    device: torch.device,
    head_list: List[Tuple[int, int]],
    input_ids: torch.Tensor,
    zero_scale: float = 0.0,
) -> torch.Tensor:
    """Run forward pass with specified heads knocked out.

    Args:
        model: The transformer model
        device: Device to run on
        head_list: List of (layer_idx, head_idx) tuples to knock out
        input_ids: Input token IDs
        zero_scale: Scale for knockout (0 = full knockout)

    Returns:
        Logits from the model
    """
    handles = []

    try:
        # Register hooks for each head to knock out
        for layer_idx, head_idx in head_list:
            # Find the attention layer
            for name, module in model.named_modules():
                if "attn" in name.lower() or "attention" in name.lower():
                    try:
                        handle = module.register_forward_hook(
                            knock_out_head_hook(layer_idx, head_idx, zero_scale)
                        )
                        handles.append(handle)
                    except Exception:
                        pass

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()

    return logits


# ============================================================
# TCR Inference
# ============================================================
def extract_final_answer(response: str, task: str) -> str:
    """Extract the final answer from a model response.

    Args:
        response: The model's text response
        task: Task name

    Returns:
        Extracted answer string
    """
    response = response.strip()

    if task == "parity_nl":
        # Look for "heads up" or "tails up" in the final answer
        for line in response.split("\n")[::-1]:
            line = line.strip().lower()
            if "heads up" in line:
                return "heads up"
            if "tails up" in line:
                return "tails up"
        # Fallback
        if "heads up" in response.lower():
            return "heads up"
        return "tails up"

    elif task in ("llc", "mdm", "moas", "clf", "nums"):
        # Extract the last number from the response
        numbers = re.findall(r'-?\d+', response)
        if numbers:
            return numbers[-1]
        return ""

    elif task == "objc":
        # Extract a number from the response
        numbers = re.findall(r'-?\d+', response)
        if numbers:
            return numbers[-1]
        return ""

    return response.strip()


def check_answer_correct(response: str, ground_truth: str, task: str) -> bool:
    """Check if the model's response contains the correct answer.

    Args:
        response: Model's text response
        ground_truth: Expected answer
        task: Task name

    Returns:
        True if answer is correct
    """
    extracted = extract_final_answer(response, task)

    if task == "parity_nl":
        return extracted.strip().lower() == ground_truth.strip().lower()

    elif task in ("llc", "mdm", "moas", "clf", "nums", "objc"):
        # Compare as strings (strip spaces)
        return extracted.strip() == ground_truth.strip()

    return extracted.strip() == ground_truth.strip()


# ============================================================
# Majority Voting for TCR
# ============================================================
def tcr_majority_vote(
    base_logits: torch.Tensor,
    knockout_logits_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    """Apply majority voting over multiple knockout configurations.

    For each candidate head, we knock it out individually and compare
    the resulting prediction against the base prediction. We use majority
    voting over top-k selected heads.

    Args:
        base_logits: Logits from the base model (no intervention)
        knockout_logits_list: List of logits from models with different heads knocked out

    Returns:
        (final_token_id, vote_count) - the majority vote result
    """
    all_votes = []

    # Get base prediction
    base_token = base_logits.argmax(dim=-1).item()
    all_votes.append(base_token)

    # Get predictions from each knockout
    for klogits in knockout_logits_list:
        k_token = klogits.argmax(dim=-1).item()
        all_votes.append(k_token)

    # Majority vote
    from collections import Counter
    vote_counts = Counter(all_votes)
    majority_token, count = vote_counts.most_common(1)[0]

    return majority_token, count


# ============================================================
# Head Selector (Classifier)
# ============================================================
class HeadSelectorClassifier:
    """Classifier that predicts which heads to knock out.

    The selector takes the input context and predicts a multi-label
    probability over candidate heads — whether knocking out each head
    would correct the prediction.
    """

    def __init__(
        self,
        model,
        tokenizer,
        candidate_heads: List[Tuple[int, int]],
        device: torch.device,
    ):
        """
        Args:
            model: The selector model (Qwen2.5-0.5B-Instruct fine-tuned)
            tokenizer: Tokenizer for the model
            candidate_heads: List of (layer, head) tuples
            device: Device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_heads = candidate_heads
        self.device = device

    def predict(
        self,
        input_text: str,
        top_k: int = 3,
    ) -> HeadSelectorOutput:
        """Predict which heads to knock out for the given input.

        Args:
            input_text: The input context (problem text)
            top_k: Number of top heads to return

        Returns:
            HeadSelectorOutput with logits, probs, and selected heads
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # For multi-label classification, we use the [CLS] or first token output
            if logits.dim() == 3:
                # Sequence classification: use first token
                logits = logits[:, 0, :]

            # Take first num_heads logits
            num_heads = len(self.candidate_heads)
            head_logits = logits[0, :num_heads].float()

            probs = F.softmax(head_logits, dim=-1)
            topk_indices = head_logits.topk(top_k).indices.tolist()

        return HeadSelectorOutput(
            logits=head_logits,
            probs=probs,
            selected_heads=topk_indices,
        )

    @staticmethod
    def load(path: str, model_name: str, device: torch.device):
        """Load a trained head selector from checkpoint."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model.to(device)
        model.eval()

        # Get candidate heads for this model
        candidate_heads = EP_HEAD_CANDIDATES.get(model_name, EP_HEAD_CANDIDATES["Qwen2.5-7B-Instruct"])

        return HeadSelectorClassifier(model, tokenizer, candidate_heads, device)
