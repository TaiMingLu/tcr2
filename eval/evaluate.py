"""
Evaluation script for TCR reasoning hop generalization tasks.

Computes accuracy metrics per task and produces scoring/scores.json.

Usage:
    python -m eval.evaluate --results_dir /home/user/results \
        --output /home/user/scoring/scores.json \
        --model_name Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.task_generator import load_dataset


def extract_final_answer(response: str, task: str) -> str:
    """Extract the final answer from a model response."""
    response = response.strip()

    if task == "parity_nl":
        for line in reversed(response.split("\n")):
            line = line.strip().lower()
            if "heads up" in line:
                return "heads up"
            if "tails up" in line:
                return "tails up"
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


def evaluate_file(filepath: str) -> Dict[str, Any]:
    """Evaluate a single result JSONL file."""
    results = load_dataset(filepath)

    if not results:
        return {"num_samples": 0, "num_correct": 0, "accuracy": 0.0}

    num_samples = len(results)
    num_correct = sum(
        1 for r in results
        if check_answer_correct(r["response"], r["ground_truth"], r["task"])
    )
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0

    return {
        "num_samples": num_samples,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
    }


def build_scores_json(
    results_dir: str,
    output_path: str,
    model_name: str,
) -> Dict[str, Any]:
    """Build scores.json from evaluation results.

    Matches the structure of scoring/reference.json.
    """
    task_metrics = defaultdict(list)

    if not os.path.exists(results_dir):
        print(f"Warning: Results directory not found: {results_dir}")
        return {}

    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith(".jsonl"):
            continue

        filepath = os.path.join(results_dir, filename)

        # Parse filename: {task}_{method}_{model_name}.jsonl
        # Handle model names with dashes
        parts = filename.replace(".jsonl", "").split("_")
        if len(parts) < 3:
            continue

        task = parts[0]
        method = parts[1]
        # Model name is everything after parts[1]
        model_from_file = "_".join(parts[2:])

        metrics = evaluate_file(filepath)

        key = f"{task}_{method}"
        task_metrics[key].append({
            "task": task,
            "method": method,
            "model": model_from_file,
            **metrics
        })

    # Build the scores structure
    # Group by task and method (average across any runs)
    task_method_metrics = {}
    for key, runs in task_metrics.items():
        if not runs:
            continue
        avg_acc = sum(r["accuracy"] for r in runs) / len(runs)
        task_method_metrics[key] = {
            "task": runs[0]["task"],
            "method": runs[0]["method"],
            "accuracy_percent": avg_acc * 100.0,
            "num_samples": sum(r["num_samples"] for r in runs),
            "num_correct": sum(r["num_correct"] for r in runs),
        }

    # Build the experiment results
    experiment_results = {}

    for key, metrics in task_method_metrics.items():
        task = metrics["task"]
        method = metrics["method"]

        # Build method display name
        if method == "baseline":
            method_key = "baseline"
        elif method == "tcr":
            method_key = "tcr"
        elif method == "tcr_gold":
            method_key = "tcr_gold"
        else:
            method_key = method

        display_name = f"{model_name}_{method_key}"
        experiment_results[display_name] = {
            "type": "proposed" if method != "baseline" else "baseline",
            f"{task}_accuracy": metrics["accuracy_percent"],
            "num_samples": metrics["num_samples"],
        }

    scores = {
        "experiments": {
            "tcr_hop_generalization": {
                "description": "Reasoning hop generalization: TCR vs baseline on reasoning tasks",
                "results": experiment_results
            }
        }
    }

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nScores written to {output_path}")

    # Print summary
    print("\nPer-task accuracy summary:")
    print(f"{'Task':<15} {'Method':<15} {'Accuracy':<10} {'Samples':<8}")
    print("-" * 50)
    for key, m in sorted(task_method_metrics.items()):
        print(f"{m['task']:<15} {m['method']:<15} {m['accuracy_percent']:.1f}%{'':4} {m['num_samples']:<8}")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/home/user/results")
    parser.add_argument("--output", type=str, default="/home/user/scoring/scores.json")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    scores = build_scores_json(args.results_dir, args.output, args.model_name)
    print("\n" + "=" * 50)
    print("Full scores:")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
