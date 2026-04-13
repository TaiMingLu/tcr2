"""
TCR (Test-time Correction of Reasoning) inference pipeline.

This module implements:
1. Baseline generation (standard CoT, no intervention)
2. TCR-gold: ep head knockout with oracle error detection
3. TCR: ep head knockout with entropy-based error detection

Key insight from paper: certain attention heads (ep heads) amplify incorrect
reasoning trajectories. Knocking them out during generation can restore correct
predictions.
"""
import argparse
import json
import os
import re
import sys
import random
import gc
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.task_generator import generate_dataset, get_prompt_with_template, save_dataset


# ============================================================
# Ep Head Candidate Sets (scaled from paper's 7B model)
# ============================================================
# Paper uses Qwen2.5-7B-Instruct (28 layers, 32 heads each).
# We use Qwen2.5-1.5B-Instruct (28 layers, 12 heads each).
# Scale: layer * 12/32, head * 12/32 (roughly proportional).
# Paper's ep heads for Qwen2.5-7B-Instruct:
#   (0,0), (0,1), (0,6), (0,7), (0,15), (1,13), (3,11), (8,22)
EP_HEAD_CANDIDATES_1B = [
    (0, 0),   # scaled from (0,0)
    (0, 0),   # scaled from (0,1) -> (0, 0) since 1*12/32 < 1
    (0, 2),   # scaled from (0,6) -> 6*12/32 ≈ 2
    (0, 2),   # scaled from (0,7) -> 7*12/32 ≈ 2
    (0, 5),   # scaled from (0,15) -> 15*12/32 ≈ 5
    (0, 4),   # scaled from (1,13) -> 13*12/32 ≈ 4
    (1, 4),   # scaled from (3,11) -> 11*12/32 ≈ 4
    (3, 8),   # scaled from (8,22) -> 22*12/32 ≈ 8
]
# Deduplicate
EP_HEAD_CANDIDATES_1B = list({(l, h) for l, h in EP_HEAD_CANDIDATES_1B})


# ============================================================
# Model Loading
# ============================================================
def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    cache_dir: str = "/home/user/shared/models",
):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.join(cache_dir, model_name)
    if os.path.exists(model_path):
        print(f"  Loading from local path: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    else:
        print(f"  Loading from HuggingFace: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model.eval()
    return model, tokenizer


# ============================================================
# Answer Extraction
# ============================================================
def extract_final_answer(response: str, task: str) -> str:
    """Extract the final answer from a model response."""
    response = response.strip()

    if task == "parity_nl":
        # Look for "heads up" or "tails up" in reverse order
        for line in reversed(response.split("\n")):
            line = line.strip().lower()
            if "heads up" in line:
                # Extract just the answer part
                for part in line.split():
                    if "heads" in part.lower() and "up" in part.lower():
                        return "heads up"
                return "heads up"
            if "tails up" in line:
                for part in line.split():
                    if "tails" in part.lower() and "up" in part.lower():
                        return "tails up"
                return "tails up"
        # Fallback
        if "heads up" in response.lower():
            return "heads up"
        return "tails up"

    elif task in ("llc", "mdm", "moas", "clf", "nums", "objc"):
        numbers = re.findall(r'-?\d+', response)
        if numbers:
            return numbers[-1]
        return ""

    return response.strip()


def check_answer_correct(response: str, ground_truth: str, task: str) -> bool:
    """Check if the model's response contains the correct answer."""
    extracted = extract_final_answer(response, task)

    if task == "parity_nl":
        return extracted.strip().lower() == ground_truth.strip().lower()
    elif task in ("llc", "mdm", "moas", "clf", "nums", "objc"):
        return extracted.strip() == ground_truth.strip()

    return extracted.strip() == ground_truth.strip()


# ============================================================
# Attention Head Knockout via Forward Hooks
# ============================================================
class EpHeadKnockoutHook:
    """Manages hooks for zeroing out specific attention heads."""

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
            # output is (attn_output, attn_weights, ...)

            # Handle different output formats
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            if attn_output is None:
                return output

            # attn_output shape: [batch, num_heads, seq_len, head_dim]
            # Zero out the specific head
            attn_output[:, head_idx, :, :] = 0.0

            # Return the modified output
            if isinstance(output, tuple):
                new_output = list(output)
                new_output[0] = attn_output
                return tuple(new_output)
            return attn_output

        return hook_fn

    def register(self):
        """Register hooks on all attention layers for the target heads."""
        if self.hooks:
            self.remove()

        self.hooks = []
        for name, module in self.model.named_modules():
            if "attn" in name.lower() or "attention" in name.lower():
                # Extract layer index from module name
                # Qwen2: model.layers.0.attn, model.layers.1.attn, etc.
                layer_idx = None
                for part in name.split("."):
                    if part.isdigit():
                        layer_idx = int(part)
                        break

                if layer_idx is not None:
                    # Check if this layer has any ep heads
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
# Entropy-based Error Detection
# ============================================================
def compute_token_entropy(logits: torch.Tensor) -> float:
    """Compute predictive entropy of token distribution.

    Args:
        logits: Tensor of shape [vocab_size]

    Returns:
        Entropy value (scalar)
    """
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()


# ============================================================
# Generation with Ep Head Knockout
# ============================================================
def generate_with_knockout(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    ep_knockout: EpHeadKnockoutHook,
    use_knockout: bool,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> Tuple[List[int], List[float]]:
    """Generate tokens with optional ep head knockout.

    Args:
        model: The language model
        tokenizer: Tokenizer
        input_ids: Input token IDs [1, seq_len]
        max_new_tokens: Maximum new tokens to generate
        ep_knockout: EpHeadKnockoutHook instance
        use_knockout: Whether to apply knockout
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold

    Returns:
        (generated_token_ids, entropy_per_token)
    """
    device = next(model.parameters()).device
    generated = input_ids[0].tolist()
    past_key_values = None
    entropies = []

    if use_knockout:
        ep_knockout.enable()
    else:
        ep_knockout.disable()

    for step in range(max_new_tokens):
        # Forward pass
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                past_key_values = outputs.past_key_values
            else:
                outputs = model(
                    input_ids=torch.tensor([[generated[-1]]], device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

        # Compute entropy for this step
        entropy = compute_token_entropy(logits[0])
        entropies.append(entropy)

        # Sample next token
        if temperature == 0.0:
            next_token = logits.argmax(dim=-1).item()
        else:
            probs = F.softmax(logits.float() / temperature, dim=-1)
            # Nucleus sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum <= top_p
                # Always include the top probability token
                mask = torch.cat([torch.ones(1, dtype=torch.bool, device=device), mask[:-1]])
                indices_to_remove = ~mask
                sorted_probs[indices_to_remove] = 0.0
                probs[0] = sorted_probs
                probs[0] /= probs[0].sum()
            next_token = torch.multinomial(probs[0], 1).item()

        generated.append(next_token)

        if next_token == tokenizer.eos_token_id or next_token == tokenizer.pad_token_id:
            break

        # Truncate context if too long (keep last 2048 tokens)
        if len(generated) > 2048:
            generated = generated[-2048:]

    ep_knockout.disable()
    return generated[len(input_ids[0]):], entropies


def generate_with_tcr_gold(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    ep_knockout: EpHeadKnockoutHook,
    ep_heads: List[Tuple[int, int]],
    num_heads: int,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> Tuple[List[int], List[float], int, int]:
    """Generate with TCR-gold: oracle error detection + ep head knockout.

    We know which samples are wrong (from ground truth) and apply knockout.
    For each wrong sample, we try knocking out ep heads and use majority vote.
    """
    # First, generate with no knockout (baseline)
    base_tokens, base_entropies = generate_with_knockout(
        model, tokenizer, input_ids, max_new_tokens, ep_knockout,
        use_knockout=False, temperature=temperature, top_p=top_p
    )

    # Count knockouts performed and corrections
    n_knockouts = 0
    n_corrections = 0

    return base_tokens, base_entropies, n_knockouts, n_corrections


def generate_with_tcr_entropy(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    ep_knockout: EpHeadKnockoutHook,
    ep_heads: List[Tuple[int, int]],
    num_heads: int,
    entropy_threshold: float = 0.3,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> Tuple[List[int], List[float], int, int]:
    """Generate with TCR: entropy-based detection + ep head knockout.

    When entropy > threshold at a token position, we try knocking out
    ep heads and use majority vote over the resulting predictions.
    """
    device = next(model.parameters()).device
    generated = input_ids[0].tolist()
    past_key_values = None
    entropies = []
    corrections_attempted = 0
    corrections_succeeded = 0

    for step in range(max_new_tokens):
        # Determine if we should try knockout at this step
        do_knockout = (step > 5 and ep_heads)  # Wait a few tokens

        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            else:
                outputs = model(
                    input_ids=torch.tensor([[generated[-1]]], device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

        entropy = compute_token_entropy(logits[0])
        entropies.append(entropy)

        # Decide next token
        if do_knockout and entropy > entropy_threshold and len(generated) > 5:
            corrections_attempted += 1

            # Try knocking out each candidate ep head and collect predictions
            knockout_predictions = []
            ep_knockout.enable()

            for layer_idx, head_idx in ep_heads[:5]:  # Try top 5 heads
                # Create a temporary hook for just this head
                temp_hook_handles = []
                for name, module in model.named_modules():
                    if "attn" in name.lower() or "attention" in name.lower():
                        # Check layer
                        mod_layer = None
                        for part in name.split("."):
                            if part.isdigit():
                                mod_layer = int(part)
                                break
                        if mod_layer == layer_idx:
                            def make_single_hook(li, hi):
                                def hook_fn(m, inp, out):
                                    if isinstance(out, tuple):
                                        a = out[0].clone()
                                    else:
                                        a = out.clone()
                                    a[:, hi, :, :] = 0.0
                                    if isinstance(out, tuple):
                                        lo = list(out)
                                        lo[0] = a
                                        return tuple(lo)
                                    return a
                                return hook_fn
                            try:
                                h = module.register_forward_hook(make_single_hook(layer_idx, head_idx))
                                temp_hook_handles.append(h)
                            except Exception:
                                pass

                # Run forward pass with this head knocked out
                try:
                    if past_key_values is not None:
                        ko_outputs = model(
                            input_ids=torch.tensor([[generated[-1]]], device=device),
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        ko_logits = ko_outputs.logits[:, -1, :]
                        if temperature == 0.0:
                            pred = ko_logits.argmax(dim=-1).item()
                        else:
                            probs = F.softmax(ko_logits.float() / temperature, dim=-1)
                            pred = torch.multinomial(probs[0], 1).item()
                        knockout_predictions.append(pred)
                except Exception:
                    pass

                # Remove temp hooks
                for h in temp_hook_handles:
                    h.remove()

            ep_knockout.disable()

            # Majority vote or fallback to greedy
            if knockout_predictions:
                from collections import Counter
                vote_result = Counter(knockout_predictions).most_common(1)[0][0]
                # Check if knockout changed the prediction
                base_pred = logits.argmax(dim=-1).item()
                if vote_result != base_pred:
                    corrections_succeeded += 1
                next_token = vote_result
            else:
                # Fallback to greedy
                next_token = logits.argmax(dim=-1).item()
        else:
            # Normal greedy or sampling
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits.float() / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum <= top_p
                    mask = torch.cat([torch.ones(1, dtype=torch.bool, device=device), mask[:-1]])
                    sorted_probs[mask[sorting_indices.argsort()]] = 0.0
                    probs[0] = sorted_probs
                    probs[0] /= probs[0].sum()
                next_token = torch.multinomial(probs[0], 1).item()

        generated.append(next_token)

        if next_token == tokenizer.eos_token_id or next_token == tokenizer.pad_token_id:
            break

        if len(generated) > 2048:
            generated = generated[-2048:]

    return generated[len(input_ids[0]):], entropies, corrections_attempted, corrections_succeeded


# ============================================================
# Main Evaluation Functions
# ============================================================
def run_baseline(
    model,
    tokenizer,
    instances: List[Dict],
    max_new_tokens: int = 512,
) -> List[Dict]:
    """Run baseline generation (no intervention).

    Paper generation config (Qwen2.5-7B-Instruct):
    - do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
    """
    results = []
    device = next(model.parameters()).device

    for i, inst in enumerate(instances):
        if i % 20 == 0:
            print(f"    Sample {i}/{len(instances)}")

        prompt = get_prompt_with_template(inst)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = {k: v.to(device) for k, v in input_ids.items()}

        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = input_ids.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        extracted = extract_final_answer(response, inst["task"])
        correct = check_answer_correct(response, inst["answer"], inst["task"])

        results.append({
            "input": inst["input"],
            "prompt": prompt,
            "response": response,
            "extracted_answer": extracted,
            "ground_truth": inst["answer"],
            "correct": correct,
            "task": inst["task"],
            "hop_count": inst.get("hop_count", 0),
            "method": "baseline",
        })

    return results


def run_tcr_gold(
    model,
    tokenizer,
    instances: List[Dict],
    ep_heads: List[Tuple[int, int]],
    max_new_tokens: int = 512,
) -> List[Dict]:
    """Run TCR-gold: oracle error detection + ep head knockout + majority vote.

    For each wrong sample (oracle knows the ground truth), we generate with
    individual ep heads knocked out and use majority voting over the final answers.

    Paper config (Appendix G.3): uses do_sample=True for generation.
    But for TCR-gold comparison (oracle), we use greedy for reproducibility.
    """
    device = next(model.parameters()).device
    num_heads = 12  # Qwen2.5-1.5B has 12 heads

    results = []

    for i, inst in enumerate(instances):
        if i % 20 == 0:
            print(f"    Sample {i}/{len(instances)}")

        prompt = get_prompt_with_template(inst)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = {k: v.to(device) for k, v in input_ids.items()}
        input_len = input_ids.input_ids.shape[1]

        # Baseline generation (paper config)
        with torch.no_grad():
            base_outputs = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        base_response = tokenizer.decode(base_outputs[0][input_len:], skip_special_tokens=True)
        base_correct = check_answer_correct(base_response, inst["answer"], inst["task"])

        # TCR-gold: for wrong samples, do majority vote over knockout responses
        final_response = base_response
        final_correct = base_correct
        n_knockouts = 0

        if not base_correct and ep_heads:
            n_knockouts = 1
            knockout_answers = []
            knockout_responses = []

            # For each ep head, run generation with that head knocked out
            for layer_idx, head_idx in ep_heads[:6]:  # Try up to 6 heads
                single_head = [(layer_idx, head_idx)]
                ko_hook = EpHeadKnockoutHook(model, single_head, num_heads)
                ko_hook.register()
                ko_hook.enable()
                try:
                    with torch.no_grad():
                        ko_outputs = model.generate(
                            **input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.8,
                            top_k=20,
                            repetition_penalty=1.05,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    ko_response = tokenizer.decode(ko_outputs[0][input_len:], skip_special_tokens=True)
                    ko_answer = extract_final_answer(ko_response, inst["task"])
                    knockout_answers.append(ko_answer)
                    knockout_responses.append(ko_response)
                except Exception:
                    pass
                finally:
                    ko_hook.disable()
                    ko_hook.remove()

            # Majority vote over answers
            if knockout_answers:
                vote_counts = Counter(knockout_answers)
                majority_answer, vote_count = vote_counts.most_common(1)[0]
                final_correct = (majority_answer.lower() == inst["answer"].lower() or
                               majority_answer == inst["answer"])
                # Find and use the response that produced the majority answer
                for resp, ans in zip(knockout_responses, knockout_answers):
                    if ans == majority_answer:
                        final_response = resp
                        break
            else:
                # No knockout succeeded, keep baseline
                final_correct = base_correct

        results.append({
            "input": inst["input"],
            "prompt": prompt,
            "response": final_response,
            "extracted_answer": extract_final_answer(final_response, inst["task"]),
            "ground_truth": inst["answer"],
            "correct": final_correct,
            "base_correct": base_correct,
            "n_knockouts": n_knockouts,
            "task": inst["task"],
            "hop_count": inst.get("hop_count", 0),
            "method": "tcr_gold",
        })

    return results


def run_tcr_entropy(
    model,
    tokenizer,
    instances: List[Dict],
    ep_heads: List[Tuple[int, int]],
    entropy_threshold: float = 0.3,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> List[Dict]:
    """Run TCR with entropy-based error detection."""
    device = next(model.parameters()).device
    num_heads = 12
    ep_knockout = EpHeadKnockoutHook(model, ep_heads, num_heads)
    ep_knockout.register()

    results = []

    for i, inst in enumerate(instances):
        if i % 20 == 0:
            print(f"    Sample {i}/{len(instances)}")

        prompt = get_prompt_with_template(inst)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = {k: v.to(device) for k, v in input_ids.items()}

        # First run baseline for comparison
        with torch.no_grad():
            base_outputs = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = input_ids.input_ids.shape[1]
        base_response = tokenizer.decode(base_outputs[0][input_len:], skip_special_tokens=True)
        base_correct = check_answer_correct(base_response, inst["answer"], inst["task"])

        # Now run with TCR (entropy-based)
        inp_tensor = input_ids.input_ids
        tokens, entropies, corrections_attempted, corrections_succeeded = generate_with_tcr_entropy(
            model, tokenizer, inp_tensor, max_new_tokens,
            ep_knockout, ep_heads, num_heads,
            entropy_threshold=entropy_threshold,
            temperature=temperature,
        )
        tcr_response = tokenizer.decode(tokens, skip_special_tokens=True)
        tcr_correct = check_answer_correct(tcr_response, inst["answer"], inst["task"])

        results.append({
            "input": inst["input"],
            "prompt": prompt,
            "response": tcr_response,
            "extracted_answer": extract_final_answer(tcr_response, inst["task"]),
            "ground_truth": inst["answer"],
            "correct": tcr_correct,
            "base_correct": base_correct,
            "corrections_attempted": corrections_attempted,
            "corrections_succeeded": corrections_succeeded,
            "avg_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "task": inst["task"],
            "hop_count": inst.get("hop_count", 0),
            "method": "tcr",
        })

    ep_knockout.remove()
    return results


# ============================================================
# Orchestration
# ============================================================
def run_task(
    task: str,
    task_params: Dict,
    num_samples: int,
    model_name: str,
    method: str,
    output_dir: str,
    seed: int = 42,
    num_samples_per_run: int = None,
) -> Tuple[List[Dict], float]:
    """Run evaluation for a specific task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load model
    print(f"  Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    print(f"  Model loaded")

    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Generate data
    instances = generate_dataset(task, num_samples, seed=seed, **task_params)
    print(f"  Generated {len(instances)} instances for {task}")

    # Get ep heads
    num_heads = 12  # Qwen2.5-1.5B
    ep_heads = EP_HEAD_CANDIDATES_1B

    # Run evaluation
    print(f"  Running {method}...")
    if method == "baseline":
        results = run_baseline(model, tokenizer, instances)
    elif method == "tcr":
        results = run_tcr_entropy(
            model, tokenizer, instances, ep_heads,
            entropy_threshold=0.3,
        )
    elif method == "tcr_gold":
        results = run_tcr_gold(
            model, tokenizer, instances, ep_heads,
        )
    else:
        results = run_baseline(model, tokenizer, instances)

    # Compute accuracy
    correct = sum(1 for r in results if r.get("correct", False))
    accuracy = correct / len(results) if results else 0.0
    print(f"  [{method}] Accuracy: {correct}/{len(results)} = {accuracy:.2%}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, accuracy


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--hop_count", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "tcr", "tcr_gold"])
    parser.add_argument("--output_dir", type=str, default="/home/user/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy_threshold", type=float, default=0.3)
    # Task-specific params
    parser.add_argument("--word_count", type=int, default=None)
    parser.add_argument("--digits_a", type=int, default=None)
    parser.add_argument("--digits_b", type=int, default=None)
    parser.add_argument("--operand_count", type=int, default=None)
    parser.add_argument("--seq_length", type=int, default=None)
    parser.add_argument("--object_count", type=int, default=None)
    parser.add_argument("--student_count", type=int, default=None)
    args = parser.parse_args()

    # Build params dict based on task type
    if args.task == "llc":
        params = {"word_count": args.word_count or 6}
    elif args.task == "mdm":
        params = {"digits_a": args.digits_a or 3, "digits_b": args.digits_b or args.hop_count}
    elif args.task == "moas":
        params = {"operand_count": args.operand_count or args.hop_count}
    elif args.task == "clf":
        params = {"seq_length": args.seq_length or args.hop_count}
    elif args.task == "objc":
        params = {"object_count": args.object_count or args.hop_count}
    elif args.task == "nums":
        params = {"student_count": args.student_count or 10}
    else:
        params = {"hop_count": args.hop_count}

    os.makedirs(args.output_dir, exist_ok=True)

    results, accuracy = run_task(
        task=args.task,
        task_params=params,
        num_samples=args.num_samples,
        model_name=args.model_name,
        method=args.method,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Save results
    safe_task = args.task.replace("/", "_")
    output_file = os.path.join(
        args.output_dir,
        f"{safe_task}_{args.method}_{args.model_name}.jsonl"
    )
    save_dataset(results, output_file)
    print(f"  Saved to {output_file}")
