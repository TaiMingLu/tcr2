"""
TCR (Test-time Correction of Reasoning) - reusable model utilities.

Core components:
1. Attention head knockout mechanism (EpHeadKnockoutHook)
2. Ep head candidate sets per model
3. Entropy-based error detection
4. Head selector classifier (multi-label)
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
        (0, 0), (0, 1), (0, 6), (0, 7), (0, 15),
        (1, 13),
        (3, 11),
        (8, 22),
    ],
    # Qwen2.5-1.5B-Instruct: 28 layers, 12 heads each (GQA: 2 KV heads)
    "Qwen2.5-1.5B-Instruct": [
        (0, 0), (0, 2), (0, 5),
        (0, 4),
        (1, 4),
        (3, 8),
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
    logits: torch.Tensor      # shape: [num_heads]
    probs: torch.Tensor       # shape: [num_heads]
    selected_heads: List[int] # top-k selected head indices


def compute_predictive_entropy(logits: torch.Tensor) -> float:
    """Compute predictive entropy of token distribution.

    Args:
        logits: Tensor of shape [vocab_size] or [batch, vocab_size]

    Returns:
        Entropy value (scalar)
    """
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    if entropy.numel() > 1:
        return entropy.mean().item()
    return entropy.item()


def get_entropy_threshold(task: str = "default") -> float:
    """Get entropy threshold for error detection.

    Paper uses tau=0.3 for all experiments (Appendix G.3).
    """
    return 0.3


# ============================================================
# Attention Head Knockout
# ============================================================
class EpHeadKnockoutHook:
    """Manages hooks for zeroing out specific attention heads during forward passes.

    This is the core mechanism of TCR: certain attention heads (ep heads) amplify
    incorrect reasoning trajectories. By zeroing their output during the forward pass,
    we allow correct trajectories to dominate.
    """

    def __init__(self, model, ep_heads: List[Tuple[int, int]], num_heads: int):
        """
        Args:
            model: The transformer model
            ep_heads: List of (layer_idx, head_idx) to knockout
            num_heads: Total number of attention heads per layer
        """
        self.model = model
        self.ep_heads = set(ep_heads)
        self.num_heads = num_heads
        self.hooks = []
        self.active = False

    def _make_hook(self, layer_idx: int, head_idx: int):
        """Create a hook that zeros out head_idx in layer_idx."""
        def hook_fn(module, input, output):
            if not self.active:
                return output

            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            if attn_output is None:
                return output

            # attn_output shape: [batch, num_heads, seq_len, head_dim]
            attn_output[:, head_idx, :, :] = 0.0

            if isinstance(output, tuple):
                new_output = list(output)
                new_output[0] = attn_output
                return tuple(new_output)
            return attn_output

        return hook_fn

    def register(self):
        """Register hooks on attention layers for target ep heads."""
        if self.hooks:
            self.remove()

        self.hooks = []
        for name, module in self.model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                layer_idx = None
                for part in name.split("."):
                    if part.isdigit():
                        layer_idx = int(part)
                        break

                if layer_idx is not None:
                    layer_heads = [h for l, h in self.ep_heads if l == layer_idx]
                    for head_idx in layer_heads:
                        try:
                            handle = module.register_forward_hook(
                                self._make_hook(layer_idx, head_idx)
                            )
                            self.hooks.append(handle)
                        except Exception:
                            pass

    def remove(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def enable(self):
        """Enable knockout during forward pass."""
        self.active = True

    def disable(self):
        """Disable knockout during forward pass."""
        self.active = False


# ============================================================
# Majority Voting for TCR
# ============================================================
def tcr_majority_vote(
    base_logits: torch.Tensor,
    knockout_logits_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    """Apply majority voting over multiple knockout configurations.

    Args:
        base_logits: Logits from the base model (no intervention)
        knockout_logits_list: List of logits from models with different heads knocked out

    Returns:
        (final_token_id, vote_count) - the majority vote result
    """
    all_votes = []

    base_token = base_logits.argmax(dim=-1).item()
    all_votes.append(base_token)

    for klogits in knockout_logits_list:
        k_token = klogits.argmax(dim=-1).item()
        all_votes.append(k_token)

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
    probability over candidate heads - whether knocking out each head
    would correct the prediction.
    """

    def __init__(
        self,
        model,
        tokenizer,
        candidate_heads: List[Tuple[int, int]],
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_heads = candidate_heads
        self.device = device

    def predict(
        self,
        input_text: str,
        top_k: int = 3,
    ) -> HeadSelectorOutput:
        """Predict which heads to knock out for the given input."""
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

            if logits.dim() == 3:
                logits = logits[:, 0, :]

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

        candidate_heads = EP_HEAD_CANDIDATES.get(
            model_name, EP_HEAD_CANDIDATES["Qwen2.5-7B-Instruct"]
        )

        return HeadSelectorClassifier(model, tokenizer, candidate_heads, device)
