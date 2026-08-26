from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline runs against a Maskable PPO training run.")
    parser.add_argument("--baseline-summary", required=True, help="Path to reports/baseline_summary.json.")
    parser.add_argument("--ppo-run-dir", required=True, help="Path to a Maskable PPO run directory.")
    parser.add_argument("--output-dir", help="Optional directory for comparison_summary outputs.")
    return parser.parse_args(argv)


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _completed_episodes(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(episode) for episode in result.get("episodes", []) if episode.get("status") == "completed"]


def summarize_baseline_result(result: dict[str, Any]) -> dict[str, Any]:
    episodes = _completed_episodes(result)
    llvm_episodes = [episode for episode in episodes if bool(episode.get("llvm_metric_available", False))]
    return {
        "method": str(result.get("baseline_name", "unknown")),
        "status": str(result.get("status", "unknown")),
        "sample_count": int(result.get("episode_count", len(result.get("episodes", [])))),
        "completed_count": len(episodes),
        "mean_reward": round(float(result.get("mean_reward", _safe_mean([float(item.get("reward", 0.0)) for item in episodes]))), 6),
        "mean_semantic_score": round(float(result.get("mean_semantic_score", _safe_mean([float(item.get("semantic_score", 0.0)) for item in episodes]))), 6),
        "mean_verifier_coverage": round(float(result.get("mean_verifier_coverage", _safe_mean([float(item.get("verifier_coverage", 0.0)) for item in episodes]))), 6),
        "mean_potency_score": round(_safe_mean([float(item.get("potency_score", 0.0)) for item in episodes]), 6),
        "mean_cost_penalty": round(_safe_mean([float(item.get("cost_penalty", 0.0)) for item in episodes]), 6),
        "llvm_metric_episode_count": len(llvm_episodes),
        "mean_llvm_block_coverage": round(float(result.get("mean_llvm_block_coverage", 0.0)), 6),
        "mean_branch_rewrite_ratio": round(float(result.get("mean_branch_rewrite_ratio", 0.0)), 6),
        "mean_ir_structural_delta": round(float(result.get("mean_ir_structural_delta", 0.0)), 6),
    }


def summarize_ppo_run(run_dir: Path) -> dict[str, Any]:
    runtime_path = run_dir / "training_runtime.json"
    runtime = load_structured_file(runtime_path)
    callback_summary = dict(runtime.get("callback_summary", {}))
    periodic = dict(callback_summary.get("periodic_evaluation", {}))
    latest_eval_path = periodic.get("latest_path", "")
    latest_eval = {}
    if latest_eval_path:
        latest_candidate = resolve_path(str(latest_eval_path))
        if latest_candidate.exists():
            latest_eval = load_structured_file(latest_candidate)
    obfuscation = dict(latest_eval.get("obfuscation_metrics", {}))
    resilience = dict(latest_eval.get("resilience_metrics", {}))
    llvm = dict(latest_eval.get("llvm_ir_metrics", {}))
    return {
        "method": "maskable_ppo",
        "status": str(runtime.get("status", "unknown")),
        "sample_count": int(runtime.get("sample_count", 0)),
        "completed_count": int(callback_summary.get("episode_count", 0)),
        "mean_reward": round(float(callback_summary.get("mean_episode_reward", 0.0)), 6),
        "mean_semantic_score": round(float(callback_summary.get("mean_semantic_score", 0.0)), 6),
        "mean_verifier_coverage": round(float(callback_summary.get("mean_verifier_coverage", 0.0)), 6),
        "mean_potency_score": round(float(callback_summary.get("mean_potency_score", obfuscation.get("mean_potency_score", 0.0))), 6),
        "mean_cost_penalty": round(float(obfuscation.get("mean_cost_penalty", 0.0)), 6),
        "llvm_metric_episode_count": int(callback_summary.get("llvm_metric_episode_count", llvm.get("llvm_metric_episode_count", 0))),
        "mean_llvm_block_coverage": round(float(callback_summary.get("mean_llvm_block_coverage", llvm.get("mean_llvm_block_coverage", 0.0))), 6),
        "mean_branch_rewrite_ratio": round(float(callback_summary.get("mean_branch_rewrite_ratio", llvm.get("mean_branch_rewrite_ratio", 0.0))), 6),
        "mean_ir_structural_delta": round(float(callback_summary.get("mean_ir_structural_delta", llvm.get("mean_ir_structural_delta", 0.0))), 6),
        "mean_proxy_resilience_score": round(float(resilience.get("mean_proxy_resilience_score", 0.0)), 6),
        "model_path": str(runtime.get("model_path", "")),
        "best_model_path": str(runtime.get("best_model_path", "")),
        "latest_periodic_evaluation": str(latest_eval_path),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = summary["methods"]
    headers = [
        "method",
        "status",
        "sample_count",
        "completed_count",
        "mean_semantic_score",
        "mean_potency_score",
        "mean_cost_penalty",
        "mean_verifier_coverage",
        "llvm_metric_episode_count",
        "mean_llvm_block_coverage",
        "mean_branch_rewrite_ratio",
        "mean_ir_structural_delta",
        "mean_proxy_resilience_score",
    ]
    lines = [
        "# Baseline Comparison Summary",
        "",
        f"- Created at: `{summary['created_at']}`",
        f"- Baseline run: `{summary['baseline_summary_path']}`",
        f"- PPO run: `{summary['ppo_run_dir']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(f"`{row.get(header, '')}`" for header in headers) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "- `mean_proxy_resilience_score` is a structural proxy, not a full deobfuscation-resistance experiment.",
            "- LLVM metric counts can be zero when clang/LLVM block extraction does not produce usable block ids for a run.",
            "- PPO `completed_count` is episode count observed during training, not necessarily one pass over every sample.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    baseline_summary_path = resolve_path(args.baseline_summary)
    ppo_run_dir = resolve_path(args.ppo_run_dir)
    output_dir = ensure_dir(resolve_path(args.output_dir) if args.output_dir else ppo_run_dir / "reports" / "comparison")

    baseline_summary = load_structured_file(baseline_summary_path)
    if not isinstance(baseline_summary, dict):
        raise ValueError(f"Baseline summary must decode to an object: {baseline_summary_path}")

    methods = [
        summarize_baseline_result(dict(result))
        for _, result in sorted(dict(baseline_summary.get("results", {})).items())
        if dict(result).get("status") not in {"skipped_tool_missing", "skipped_not_configured", "skipped_unknown_baseline"}
    ]
    methods.append(summarize_ppo_run(ppo_run_dir))

    summary = {
        "created_at": utc_timestamp(),
        "baseline_summary_path": project_relative(baseline_summary_path),
        "ppo_run_dir": project_relative(ppo_run_dir),
        "methods": methods,
    }
    json_path = output_dir / "comparison_summary.json"
    md_path = output_dir / "comparison_summary.md"
    dump_json(json_path, summary)
    write_text(md_path, render_markdown(summary))
    print(f"Wrote comparison summary JSON: {json_path}")
    print(f"Wrote comparison summary Markdown: {md_path}")


if __name__ == "__main__":
    main()
