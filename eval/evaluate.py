"""
Evaluation script for TCR reasoning hop generalization tasks.

This script evaluates model outputs against ground truth and produces
accuracy metrics per task and across all tasks.

Usage:
    python -m eval.evaluate --results_dir /home/user/results \
        --output /home/user/scoring/scores.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.task_generator import load_dataset, generate_dataset


def evaluate_results(
    results: List[Dict[str, Any]],
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate results and compute accuracy.

    Args:
        results: List of result dicts with 'correct', 'task', etc.
        task: Optional task name for filtering

    Returns:
        Dict with accuracy and per-task breakdown
    """
    if task:
        results = [r for r in results if r.get("task") == task]

    if not results:
        return {"accuracy": 0.0, "num_samples": 0, "num_correct": 0}

    num_samples = len(results)
    num_correct = sum(1 for r in results if r.get("correct", False))
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0

    return {
        "accuracy": accuracy,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "accuracy_percent": accuracy * 100.0,
    }


def evaluate_results_dir(
    results_dir: str,
    model_name: str,
    method: str = "baseline",
) -> Dict[str, Any]:
    """Evaluate all results files in a directory.

    Args:
        results_dir: Directory containing result JSONL files
        model_name: Model name (for filtering)
        method: Method name (for filtering)

    Returns:
        Dict with per-task and aggregate accuracy
    """
    all_results = defaultdict(list)
    task_accuracies = {}

    if not os.path.exists(results_dir):
        return {"error": f"Results directory not found: {results_dir}"}

    for filename in os.listdir(results_dir):
        if not filename.endswith(".jsonl"):
            continue

        # Parse filename: task_method_model.jsonl
        parts = filename.replace(".jsonl", "").split("_")
        if len(parts) < 2:
            continue

        file_task = parts[0]
        file_method = parts[1]

        # Filter by model and method
        if method and file_method != method:
            continue

        filepath = os.path.join(results_dir, filename)
        results = load_dataset(filepath)

        # Evaluate
        metrics = evaluate_results(results)
        task_accuracies[file_task] = metrics
        all_results[file_task].extend(results)

    # Compute aggregate
    all_correct = sum(1 for r_list in all_results.values() for r in r_list if r.get("correct", False))
    all_samples = sum(len(r_list) for r_list in all_results.values())
    aggregate_accuracy = all_correct / all_samples if all_samples > 0 else 0.0

    return {
        "aggregate": {
            "accuracy": aggregate_accuracy,
            "accuracy_percent": aggregate_accuracy * 100.0,
            "num_samples": all_samples,
            "num_correct": all_correct,
        },
        "per_task": task_accuracies,
    }


def compute_all_tasks_accuracy(
    results_dir: str,
) -> Dict[str, float]:
    """Compute accuracy across all task-specific result files.

    Handles filenames like:
        parity_nl_baseline_Qwen2.5-1.5B-Instruct.jsonl
        mdm_tcr_Qwen2.5-1.5B-Instruct.jsonl
    """
    task_metrics = {}

    if not os.path.exists(results_dir):
        return task_metrics

    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(results_dir, filename)

        # Parse task from filename
        # Format: {task}_{method}_{model}.jsonl
        parts = filename.replace(".jsonl", "").split("_")
        if len(parts) < 3:
            continue

        task = parts[0]
        method = parts[1]
        model = "_".join(parts[2:])

        results = load_dataset(filepath)
        num_correct = sum(1 for r in results if r.get("correct", False))
        num_total = len(results)
        acc = num_correct / num_total if num_total > 0 else 0.0

        key = f"{task}_{method}"
        task_metrics[key] = {
            "accuracy": acc,
            "accuracy_percent": acc * 100.0,
            "num_samples": num_total,
            "num_correct": num_correct,
            "method": method,
            "task": task,
            "model": model,
        }

    return task_metrics


def build_scores_json(
    results_dir: str,
    output_path: str,
    model_name: str,
):
    """Build scores.json from evaluation results.

    Format matches scoring/reference.json structure.
    """
    metrics = compute_all_tasks_accuracy(results_dir)

    # Group by experiment
    # Map task names to paper's experiment names
    task_to_exp = {
        "parity_nl": "tcr_hop_generalization",
        "mdm": "tcr_hop_generalization",
        "llc": "tcr_hop_generalization",
        "clf": "tcr_hop_generalization",
        "moas": "tcr_hop_generalization",
        "objc": "tcr_hop_generalization",
        "nums": "tcr_hop_generalization",
    }

    scores = {
        "experiments": {
            "tcr_hop_generalization": {
                "description": "TCR on reasoning hop generalization tasks",
                "results": {}
            }
        }
    }

    for key, m in metrics.items():
        task = m["task"]
        method = m["method"]

        # Normalize method name
        if method == "baseline":
            display_name = f"baseline_{model_name}"
        elif method == "tcr":
            display_name = f"tcr_{model_name}"
        elif method == "tcr_gold":
            display_name = f"tcr_gold_{model_name}"
        else:
            display_name = key

        scores["experiments"]["tcr_hop_generalization"]["results"][display_name] = {
            "accuracy_percent": m["accuracy_percent"],
            "num_samples": m["num_samples"],
        }

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"Wrote scores to {output_path}")
    print(json.dumps(scores, indent=2))

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/home/user/results")
    parser.add_argument("--output", type=str, default="/home/user/scoring/scores.json")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    scores = build_scores_json(args.results_dir, args.output, args.model_name)


if __name__ == "__main__":
    main()
