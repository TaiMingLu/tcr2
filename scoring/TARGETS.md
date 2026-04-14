# Evaluation Targets

## Primary: task_accuracy
Improvement in final-answer accuracy on reasoning hop generalization tasks.
- **Metric**: per-task accuracy (higher is better)
- **Experiments**: All tasks in `tcr_hop_generalization`
- **Evaluation**: Run `scripts/evaluate.sh` to compute accuracy from result JSONL files

## Constraint: model_size
The method must work with the same model family without increasing compute.
- **Metric**: parameter count should not increase
- **Note**: TCR is a test-time intervention, so model size is unchanged

## Ablation: ep_head_knockout_effect
Knocking out erroneous processing heads (ep heads) should improve accuracy over baseline.
- **Metric**: accuracy_improvement = TCR_accuracy - baseline_accuracy (higher is better)
- **Evaluation**: Compare baseline vs tcr_gold accuracy on the same task

Every target above corresponds to an entry in both `reference.json` and `scores.json`.
