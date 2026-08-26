from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_json_if_exists, project_relative, resolve_path, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown status report for a run directory.")
    parser.add_argument("--run_dir", required=True, help="Run directory created by scripts.train.")
    parser.add_argument("--output", help="Optional explicit output Markdown path.")
    return parser.parse_args(argv)


def render_report(run_dir: Path, metadata: dict, train_plan: dict | None, evaluation: dict | None) -> str:
    dataset = metadata.get("dataset", {})
    training_runtime = metadata.get("training_runtime", {})
    dataset_artifacts = [
        dataset.get("split_index_snapshot_path", ""),
        dataset.get("sample_ids_path", ""),
        dataset.get("task_queue_path", ""),
        dataset.get("env_input_cache_path", ""),
    ]
    dataset_artifacts = [path for path in dataset_artifacts if path]
    sample_preview = ", ".join(dataset.get("sample_preview", [])) or "unspecified"

    lines = [
        "# Experiment Report",
        "",
        "## Overview",
        f"- Run directory: `{project_relative(run_dir)}`",
        f"- Job name: `{metadata.get('job_name', 'unknown')}`",
        f"- Status: `{metadata.get('status', 'unknown')}`",
        f"- Created at: `{metadata.get('created_at', 'unknown')}`",
        "",
        "## Training Setup",
        f"- Policy: `{metadata.get('policy', {}).get('algorithm', 'unspecified')}`",
        f"- Environment: `{metadata.get('environment', {}).get('id', 'unspecified')}`",
        f"- Semantic threshold: `{metadata.get('environment', {}).get('semantic_threshold', 'unspecified')}`",
        f"- Total timesteps: `{metadata.get('training', {}).get('total_timesteps', 'unspecified')}`",
        f"- Training runtime status: `{training_runtime.get('status', 'unspecified')}`",
        f"- Checkpoint selection: `{training_runtime.get('callback_summary', {}).get('selection_metric', 'unspecified')}`",
        "",
        "## Dataset",
        f"- Split index: `{dataset.get('split_index_path', 'unspecified')}`",
        f"- Split: `{dataset.get('split', 'unspecified')}`",
        f"- Sample count: `{dataset.get('sample_count', 'unspecified')}`",
        f"- Manifest job: `{dataset.get('manifest_job_name', 'unspecified')}`",
        f"- Sample preview: `{sample_preview}`",
        "",
        "## Artifacts",
        "- `metadata.json`",
        "- `train_plan.json`",
        "- `logs/reward_breakdown.csv`",
        "- `logs/action_mask_stats.json`",
        "- `logs/seed.txt`",
    ]

    lines.extend([f"- `{path}`" for path in dataset_artifacts])
    if training_runtime.get("runtime_summary_path"):
        lines.append(f"- `{training_runtime['runtime_summary_path']}`")
    if training_runtime.get("model_path"):
        lines.append(f"- `{training_runtime['model_path']}`")
    if training_runtime.get("best_model_path"):
        lines.append(f"- `{training_runtime['best_model_path']}`")
    if training_runtime.get("callback_summary_path"):
        lines.append(f"- `{training_runtime['callback_summary_path']}`")
    periodic_info = training_runtime.get("callback_summary", {}).get("periodic_evaluation", {})
    if periodic_info.get("latest_path"):
        lines.append(f"- `{periodic_info['latest_path']}`")
    if periodic_info.get("index_path"):
        lines.append(f"- `{periodic_info['index_path']}`")

    if evaluation:
        semantic = evaluation.get("semantic_verification", {})
        derived = evaluation.get("derived_flags", {})
        test_summary = semantic.get("tests", {})
        llvm_ir_metrics = evaluation.get("llvm_ir_metrics", {})
        obfuscation_metrics = evaluation.get("obfuscation_metrics", {})
        resilience_metrics = evaluation.get("resilience_metrics", {})
        lines.extend(
            [
                "- `reports/evaluation_summary.json`",
                "",
                "## Evaluation Snapshot",
                f"- Benchmarks: `{', '.join(evaluation.get('benchmark', {}).get('datasets', [])) or 'unspecified'}`",
                f"- Metrics: `{', '.join(evaluation.get('metrics', [])) or 'unspecified'}`",
                f"- Semantic score: `{semantic.get('semantic_score', 'unspecified')}`",
                f"- Semantic gate pass: `{derived.get('passes_semantic_gate', 'unspecified')}`",
                f"- Verifier coverage: `{semantic.get('verifier_coverage', 'unspecified')}`",
                f"- Available signals: `{', '.join(semantic.get('available_signals', [])) or 'unspecified'}`",
                f"- Passed test cases: `{test_summary.get('passed_cases', 'unspecified')}`",
                f"- Failed test cases: `{test_summary.get('failed_cases', 'unspecified')}`",
                f"- LLVM metric episodes: `{llvm_ir_metrics.get('llvm_metric_episode_count', 'unspecified')}`",
                f"- Mean LLVM block coverage: `{llvm_ir_metrics.get('mean_llvm_block_coverage', 'unspecified')}`",
                f"- Mean branch rewrite ratio: `{llvm_ir_metrics.get('mean_branch_rewrite_ratio', 'unspecified')}`",
                f"- Mean IR structural delta: `{llvm_ir_metrics.get('mean_ir_structural_delta', 'unspecified')}`",
                f"- Obfuscation metric episodes: `{obfuscation_metrics.get('episode_count', 'unspecified')}`",
                f"- Semantic gate pass rate: `{obfuscation_metrics.get('semantic_gate_pass_rate', 'unspecified')}`",
                f"- Mean potency score: `{obfuscation_metrics.get('mean_potency_score', 'unspecified')}`",
                f"- Mean cost penalty: `{obfuscation_metrics.get('mean_cost_penalty', 'unspecified')}`",
                f"- Mean applied action count: `{obfuscation_metrics.get('mean_applied_action_count', 'unspecified')}`",
                f"- Resilience metric episodes: `{resilience_metrics.get('episode_count', 'unspecified')}`",
                f"- Proxy resilience score: `{resilience_metrics.get('mean_proxy_resilience_score', 'unspecified')}`",
                f"- Flatten-CFG episode rate: `{resilience_metrics.get('flatten_cfg_episode_rate', 'unspecified')}`",
                f"- Operator diversity ratio: `{resilience_metrics.get('mean_operator_diversity_ratio', 'unspecified')}`",
            ]
        )
    else:
        periodic_snapshot_path = periodic_info.get("latest_path", "")
        lines.extend(
            [
                "",
                "## Evaluation Snapshot",
                f"- No evaluation summary found yet. Latest periodic snapshot: `{periodic_snapshot_path or 'unavailable'}`",
            ]
        )

    if train_plan:
        lines.extend(
            [
                "",
                "## Next Steps",
                *[f"- {step}" for step in train_plan.get("steps", [])],
            ]
        )

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = resolve_path(args.run_dir)
    metadata = load_json_if_exists(run_dir / "metadata.json")
    if metadata is None:
        raise FileNotFoundError(f"Run metadata not found in: {run_dir}")

    train_plan = load_json_if_exists(run_dir / "train_plan.json")
    evaluation = load_json_if_exists(run_dir / "reports" / "evaluation_summary.json")
    output_path = resolve_path(args.output, run_dir) if args.output else run_dir / "reports" / "report.md"

    write_text(output_path, render_report(run_dir, metadata, train_plan, evaluation))
    print(f"Wrote report: {output_path}")


if __name__ == "__main__":
    main()
