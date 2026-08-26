from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


CSV_FIELDS = [
    "method",
    "base_method",
    "variant",
    "dataset",
    "seed",
    "total_timesteps",
    "source_kind",
    "status",
    "sample_count",
    "completed_count",
    "semantic_pass_count",
    "semantic_pass_rate",
    "mean_reward",
    "mean_semantic_score",
    "mean_verifier_coverage",
    "mean_potency_score",
    "mean_cost_penalty",
    "llvm_metric_episode_count",
    "mean_llvm_block_coverage",
    "mean_branch_rewrite_ratio",
    "mean_ir_structural_delta",
    "failure_count",
    "compile_failed_count",
    "semantic_failed_count",
    "tool_parse_failure_count",
    "tool_compile_failure_count",
    "tool_timeout_count",
    "tool_incompatible_count",
    "failed_samples",
    "notes",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize baseline, PPO, verifier failure, and Tigress diagnosis outputs."
    )
    parser.add_argument("--baseline-summary", action="append", default=[], help="Path to reports/baseline_summary.json.")
    parser.add_argument("--ppo-run-dir", action="append", default=[], help="Path to a Maskable PPO run directory.")
    parser.add_argument("--failure-audit", action="append", default=[], help="Path to failure_audit.json.")
    parser.add_argument("--tigress-diagnosis", action="append", default=[], help="Path to tigress_failure_diagnosis.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for experiment_summary outputs.")
    parser.add_argument("--job-name", default="experiment_summary", help="Name stored in generated reports.")
    parser.add_argument("--semantic-threshold", type=float, default=0.97, help="Minimum semantic score counted as pass.")
    parser.add_argument(
        "--group-by",
        default="",
        help="Comma-separated row fields for grouped mean/std output, e.g. method,variant or dataset,method,variant.",
    )
    parser.add_argument(
        "--aggregate",
        default="seed",
        help="Comma-separated fields treated as replicate dimensions in grouped summaries. Stored for provenance.",
    )
    return parser.parse_args(argv)


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _load_dict(path: Path) -> dict[str, Any]:
    data = load_structured_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def _load_dict_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = load_structured_file(path)
    return dict(data) if isinstance(data, dict) else {}


def _split_csv_fields(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _strip_duplicate_suffix(method: str) -> str:
    return str(method).split("#", 1)[0]


def _infer_dataset(metadata: dict[str, Any], runtime: dict[str, Any], run_dir: Path) -> str:
    dataset = dict(metadata.get("dataset", {}))
    periodic = dict(dict(metadata.get("training", {})).get("periodic_evaluation", {}))
    benchmark = dict(periodic.get("benchmark", {}))
    datasets = benchmark.get("datasets", [])
    if isinstance(datasets, list) and datasets:
        return str(datasets[0])
    split_index_path = str(dataset.get("split_index_path", ""))
    if split_index_path:
        name = Path(split_index_path).name
        return name.removesuffix(".index.json").removesuffix(".json")
    runtime_path = str(runtime.get("runtime_summary_path", ""))
    if "formal_eval_30_v2" in runtime_path or "formal_eval_30_v2" in run_dir.as_posix():
        return "formal_eval_30_v2"
    return ""


def _infer_variant(metadata: dict[str, Any], runtime: dict[str, Any], run_dir: Path) -> str:
    text = " ".join(
        [
            str(metadata.get("job_name", "")),
            str(metadata.get("config_path", "")),
            str(run_dir.as_posix()),
        ]
    ).lower()
    if "no_action_safety" in text:
        return "no_action_safety"
    if "no_fuzz_verifier" in text:
        return "no_fuzz_verifier"
    if "no_llvm_reward" in text:
        return "no_llvm_reward"
    timesteps = int(runtime.get("total_timesteps", dict(metadata.get("training", {})).get("total_timesteps", 0)) or 0)
    if "long" in text or timesteps >= 1000:
        return f"full_long_{timesteps}" if timesteps else "full_long"
    if "formal_eval_30" in text or "mppo_seed" in text:
        return "full_short"
    return "default"


def _row_group_value(row: dict[str, Any], field: str) -> str:
    if field == "method":
        return str(row.get("base_method") or _strip_duplicate_suffix(str(row.get("method", ""))))
    return str(row.get(field, ""))


def _sort_replicate_values(values: set[str]) -> list[str]:
    def key(value: str) -> tuple[int, int | str]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(values, key=key)


def _episodes(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in result.get("episodes", []) if isinstance(item, dict)]


def _completed_episodes(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in _episodes(result) if item.get("status") == "completed"]


def _sample_id(episode: dict[str, Any]) -> str:
    return str(episode.get("sample_id", "unknown"))


def _classify_episode_failure(episode: dict[str, Any], semantic_threshold: float) -> str | None:
    if float(episode.get("semantic_score", 0.0)) >= semantic_threshold:
        return None
    status = str(episode.get("status", ""))
    stderr = str(episode.get("transform_stderr", ""))
    stdout = str(episode.get("transform_stdout", ""))
    compile_succeeded = bool(episode.get("compile_succeeded", False))
    tests_passed = bool(episode.get("tests_passed", False))

    if "timeout" in status.lower() or "timed out" in stderr.lower() or "timed out" in stdout.lower():
        return "tool_timeout"
    if status in {"transform_failed", "skipped_tool_missing"} or "Parsing error" in stderr:
        return "tool_parse_failure"
    if not compile_succeeded:
        return "tool_compile_failure"
    if compile_succeeded and not tests_passed:
        return "semantic_failure"
    return "semantic_failure"


def _empty_failure_counts() -> dict[str, int]:
    return {
        "failure_count": 0,
        "compile_failed_count": 0,
        "semantic_failed_count": 0,
        "tool_parse_failure_count": 0,
        "tool_compile_failure_count": 0,
        "tool_timeout_count": 0,
        "tool_incompatible_count": 0,
    }


def _episode_failure_counts(episodes: list[dict[str, Any]], semantic_threshold: float) -> tuple[dict[str, int], list[str]]:
    counts = _empty_failure_counts()
    failed_samples: list[str] = []
    for episode in episodes:
        semantic_score = float(episode.get("semantic_score", 0.0))
        if semantic_score >= semantic_threshold:
            continue
        sample_id = _sample_id(episode)
        failed_samples.append(sample_id)
        counts["failure_count"] += 1
        if not bool(episode.get("compile_succeeded", True)):
            counts["compile_failed_count"] += 1
        kind = _classify_episode_failure(episode, semantic_threshold)
        if kind == "semantic_failure":
            counts["semantic_failed_count"] += 1
        elif kind == "tool_parse_failure":
            counts["tool_parse_failure_count"] += 1
        elif kind == "tool_compile_failure":
            counts["tool_compile_failure_count"] += 1
        elif kind == "tool_timeout":
            counts["tool_timeout_count"] += 1
    return counts, sorted(set(failed_samples))


def _apply_tigress_diagnosis(row: dict[str, Any], diagnosis_by_sample: dict[str, dict[str, Any]]) -> None:
    if row.get("method") != "tigress" or not diagnosis_by_sample:
        return
    tool_incompatible = 0
    semantic_risky = 0
    fixed_by_config = 0
    notes: list[str] = []
    for sample_id, item in sorted(diagnosis_by_sample.items()):
        decision = str(item.get("decision", ""))
        if decision.startswith("tool_incompatible"):
            tool_incompatible += 1
        elif decision == "preprocess_possible_but_semantic_risky":
            semantic_risky += 1
        elif decision == "adjust_tigress_verification_command":
            fixed_by_config += 1
        notes.append(f"{sample_id}:{decision}")
    row["tool_incompatible_count"] = tool_incompatible
    if semantic_risky and int(row.get("semantic_failed_count", 0)) == 0:
        row["semantic_failed_count"] = semantic_risky
    row["notes"] = "; ".join(
        filter(
            None,
            [
                str(row.get("notes", "")),
                f"tigress_diagnosis={len(diagnosis_by_sample)}",
                f"fixed_by_config={fixed_by_config}" if fixed_by_config else "",
                "diagnosis_details=" + ",".join(notes),
            ],
        )
    )


def summarize_baseline_result(
    result: dict[str, Any],
    *,
    source_path: Path,
    semantic_threshold: float,
    tigress_diagnosis: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    episodes = _episodes(result)
    completed = _completed_episodes(result)
    semantic_pass_count = sum(1 for item in episodes if float(item.get("semantic_score", 0.0)) >= semantic_threshold)
    llvm_episodes = [item for item in episodes if bool(item.get("llvm_metric_available", False))]
    failure_counts, failed_samples = _episode_failure_counts(episodes, semantic_threshold)
    sample_count = int(result.get("episode_count", len(episodes)))
    row = {
        "method": str(result.get("baseline_name", "unknown")),
        "base_method": str(result.get("baseline_name", "unknown")),
        "variant": "baseline",
        "dataset": "",
        "seed": "",
        "total_timesteps": "",
        "source_kind": "baseline_summary",
        "source_path": project_relative(source_path),
        "status": str(result.get("status", "unknown")),
        "sample_count": sample_count,
        "completed_count": int(result.get("completed_episode_count", len(completed))),
        "semantic_pass_count": semantic_pass_count,
        "semantic_pass_rate": _round(semantic_pass_count / sample_count) if sample_count else 0.0,
        "mean_reward": _round(float(result.get("mean_reward", _safe_mean([float(item.get("reward", 0.0)) for item in completed])))),
        "mean_semantic_score": _round(float(result.get("mean_semantic_score", _safe_mean([float(item.get("semantic_score", 0.0)) for item in episodes])))),
        "mean_verifier_coverage": _round(float(result.get("mean_verifier_coverage", _safe_mean([float(item.get("verifier_coverage", 0.0)) for item in episodes])))),
        "mean_potency_score": _round(float(result.get("mean_potency_score", _safe_mean([float(item.get("potency_score", 0.0)) for item in episodes])))),
        "mean_cost_penalty": _round(float(result.get("mean_cost_penalty", _safe_mean([float(item.get("cost_penalty", 0.0)) for item in episodes])))),
        "llvm_metric_episode_count": len(llvm_episodes),
        "mean_llvm_block_coverage": _round(float(result.get("mean_llvm_block_coverage", _safe_mean([float(item.get("llvm_block_coverage", 0.0)) for item in llvm_episodes])))),
        "mean_branch_rewrite_ratio": _round(float(result.get("mean_branch_rewrite_ratio", _safe_mean([float(item.get("branch_rewrite_ratio", 0.0)) for item in llvm_episodes])))),
        "mean_ir_structural_delta": _round(float(result.get("mean_ir_structural_delta", _safe_mean([float(item.get("ir_structural_delta", 0.0)) for item in llvm_episodes])))),
        "failed_samples": ",".join(failed_samples),
        "notes": "",
    }
    row.update(failure_counts)
    _apply_tigress_diagnosis(row, tigress_diagnosis)
    return row


def _read_episode_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    episodes: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        decoded = json.loads(line)
        if isinstance(decoded, dict):
            episodes.append(decoded)
    return episodes


def summarize_ppo_run(run_dir: Path, *, semantic_threshold: float) -> dict[str, Any]:
    runtime_path = run_dir / "training_runtime.json"
    frozen_summary_path = run_dir / "reports" / "frozen_policy_evaluation.json"
    if not runtime_path.exists() and frozen_summary_path.exists():
        frozen_summary = _load_dict(frozen_summary_path)
        result = dict(frozen_summary.get("result", {}))
        episode_metrics = [dict(item) for item in result.get("episodes", []) if isinstance(item, dict)]
        semantic_pass_count = sum(
            1
            for item in episode_metrics
            if str(item.get("status", "completed")) == "completed"
            and float(item.get("semantic_score", 0.0)) >= semantic_threshold
        )
        failure_counts, failed_samples = _episode_failure_counts(episode_metrics, semantic_threshold)
        completed_count = int(result.get("completed_episode_count", len(episode_metrics)))
        llvm_episodes = [item for item in episode_metrics if bool(item.get("llvm_metric_available", False))]
        method = str(result.get("baseline_name", "maskable_ppo_frozen"))
        return {
            "method": method,
            "base_method": "maskable_ppo",
            "variant": run_dir.name,
            "dataset": _infer_dataset({}, {}, run_dir),
            "seed": frozen_summary.get("seed", ""),
            "total_timesteps": "",
            "source_kind": "frozen_policy_evaluation",
            "source_path": project_relative(run_dir),
            "status": str(frozen_summary.get("status", result.get("status", "unknown"))),
            "sample_count": int(frozen_summary.get("sample_count", len(episode_metrics))),
            "completed_count": completed_count,
            "semantic_pass_count": semantic_pass_count,
            "semantic_pass_rate": _round(semantic_pass_count / completed_count) if completed_count else 0.0,
            "mean_reward": _round(_safe_mean([float(item.get("reward", 0.0)) for item in episode_metrics])),
            "mean_semantic_score": _round(_safe_mean([float(item.get("semantic_score", 0.0)) for item in episode_metrics])),
            "mean_verifier_coverage": _round(_safe_mean([float(item.get("verifier_coverage", 0.0)) for item in episode_metrics])),
            "mean_potency_score": _round(_safe_mean([float(item.get("potency_score", 0.0)) for item in episode_metrics])),
            "mean_cost_penalty": _round(_safe_mean([float(item.get("cost_penalty", 0.0)) for item in episode_metrics])),
            "llvm_metric_episode_count": len(llvm_episodes),
            "mean_llvm_block_coverage": _round(_safe_mean([float(item.get("llvm_block_coverage", 0.0)) for item in llvm_episodes])),
            "mean_branch_rewrite_ratio": _round(_safe_mean([float(item.get("branch_rewrite_ratio", 0.0)) for item in llvm_episodes])),
            "mean_ir_structural_delta": _round(_safe_mean([float(item.get("ir_structural_delta", 0.0)) for item in llvm_episodes])),
            **failure_counts,
            "failed_samples": ",".join(failed_samples),
            "notes": ["source=frozen_policy_evaluation"],
        }
    runtime = _load_dict(runtime_path)
    metadata = _load_dict_if_exists(run_dir / "metadata.json")
    callback_summary = dict(runtime.get("callback_summary", {}))
    episode_metrics_path = callback_summary.get("episode_metrics_path", "")
    corrected_episode_metrics_path = run_dir / "logs" / "episode_metrics.corrected.jsonl"
    episode_metrics_source = corrected_episode_metrics_path if corrected_episode_metrics_path.exists() else resolve_path(str(episode_metrics_path))
    episode_metrics = _read_episode_jsonl(episode_metrics_source) if episode_metrics_path or corrected_episode_metrics_path.exists() else []
    semantic_pass_count = sum(
        1 for item in episode_metrics if float(item.get("semantic_score", callback_summary.get("mean_semantic_score", 0.0))) >= semantic_threshold
    )
    if not episode_metrics:
        semantic_pass_count = int(callback_summary.get("episode_count", 0)) if float(callback_summary.get("mean_semantic_score", 0.0)) >= semantic_threshold else 0

    failure_counts, failed_samples = _episode_failure_counts(episode_metrics, semantic_threshold)
    completed_count = int(callback_summary.get("episode_count", len(episode_metrics)))
    mean_reward = _safe_mean([float(item.get("reward", item.get("episode_reward", 0.0))) for item in episode_metrics])
    mean_semantic = _safe_mean([float(item.get("semantic_score", 0.0)) for item in episode_metrics])
    mean_coverage = _safe_mean([float(item.get("verifier_coverage", 0.0)) for item in episode_metrics])
    mean_potency = _safe_mean([float(item.get("potency_score", 0.0)) for item in episode_metrics])
    mean_cost = _safe_mean([float(item.get("cost_penalty", 0.0)) for item in episode_metrics])
    llvm_episodes = [item for item in episode_metrics if bool(item.get("llvm_metric_available", False))]
    method = str(runtime.get("algorithm", "maskable_ppo"))
    seed = runtime.get("seed", dict(metadata.get("training", {})).get("seed", ""))
    total_timesteps = runtime.get("total_timesteps", dict(metadata.get("training", {})).get("total_timesteps", ""))
    return {
        "method": method,
        "base_method": method,
        "variant": _infer_variant(metadata, runtime, run_dir),
        "dataset": _infer_dataset(metadata, runtime, run_dir),
        "seed": seed,
        "total_timesteps": total_timesteps,
        "source_kind": "ppo_run",
        "source_path": project_relative(run_dir),
        "status": str(runtime.get("status", "unknown")),
        "sample_count": int(runtime.get("sample_count", 0)),
        "completed_count": completed_count,
        "semantic_pass_count": semantic_pass_count,
        "semantic_pass_rate": _round(semantic_pass_count / completed_count) if completed_count else 0.0,
        "mean_reward": _round(mean_reward if episode_metrics else float(callback_summary.get("mean_episode_reward", 0.0))),
        "mean_semantic_score": _round(mean_semantic if episode_metrics else float(callback_summary.get("mean_semantic_score", 0.0))),
        "mean_verifier_coverage": _round(mean_coverage if episode_metrics else float(callback_summary.get("mean_verifier_coverage", 0.0))),
        "mean_potency_score": _round(mean_potency if episode_metrics else float(callback_summary.get("mean_potency_score", 0.0))),
        "mean_cost_penalty": _round(mean_cost if episode_metrics else float(callback_summary.get("mean_cost_penalty", 0.0))),
        "llvm_metric_episode_count": len(llvm_episodes) if episode_metrics else int(callback_summary.get("llvm_metric_episode_count", 0)),
        "mean_llvm_block_coverage": _round(_safe_mean([float(item.get("llvm_block_coverage", 0.0)) for item in llvm_episodes]) if episode_metrics else float(callback_summary.get("mean_llvm_block_coverage", 0.0))),
        "mean_branch_rewrite_ratio": _round(_safe_mean([float(item.get("branch_rewrite_ratio", 0.0)) for item in llvm_episodes]) if episode_metrics else float(callback_summary.get("mean_branch_rewrite_ratio", 0.0))),
        "mean_ir_structural_delta": _round(_safe_mean([float(item.get("ir_structural_delta", 0.0)) for item in llvm_episodes]) if episode_metrics else float(callback_summary.get("mean_ir_structural_delta", 0.0))),
        "failed_samples": ",".join(failed_samples),
        "notes": "; ".join(
            filter(
                None,
                [
                    f"best_selection_score={callback_summary.get('best_selection_score', '')}",
                    f"episode_metrics_source={project_relative(episode_metrics_source)}" if episode_metrics else "",
                ],
            )
        ),
        **failure_counts,
    }


def load_tigress_diagnosis(paths: list[str]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for value in paths:
        path = resolve_path(value)
        data = _load_dict(path)
        diagnosis = data.get("diagnosis", {})
        if isinstance(diagnosis, dict):
            for sample_id, item in diagnosis.items():
                if isinstance(item, dict):
                    merged[str(sample_id)] = dict(item)
    return merged


def load_failure_audits(paths: list[str]) -> dict[str, Any]:
    by_reason: Counter[str] = Counter()
    by_method: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for value in paths:
        path = resolve_path(value)
        data = _load_dict(path)
        summary = dict(data.get("summary", {}))
        by_reason.update({str(k): int(v) for k, v in dict(summary.get("by_reason", {})).items()})
        by_method.update({str(k): int(v) for k, v in dict(summary.get("by_method", {})).items()})
        failures.extend([dict(item) for item in data.get("failures", []) if isinstance(item, dict)])
    return {
        "failure_count": len(failures),
        "by_reason": dict(sorted(by_reason.items())),
        "by_method": dict(sorted(by_method.items())),
        "failures": failures,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    headers = [
        "method",
        "status",
        "sample_count",
        "semantic_pass_rate",
        "mean_semantic_score",
        "mean_potency_score",
        "mean_verifier_coverage",
        "mean_llvm_block_coverage",
        "mean_branch_rewrite_ratio",
        "mean_ir_structural_delta",
        "failure_count",
        "tool_incompatible_count",
    ]
    lines = [
        "# Experiment Summary",
        "",
        f"- Job: `{summary['job_name']}`",
        f"- Created at: `{summary['created_at']}`",
        f"- Semantic threshold: `{summary['semantic_threshold']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in summary["methods"]:
        lines.append("| " + " | ".join(f"`{row.get(header, '')}`" for header in headers) + " |")

    lines.extend(["", "## Failure Notes", ""])
    for row in summary["methods"]:
        failed = str(row.get("failed_samples", ""))
        notes = str(row.get("notes", ""))
        if failed or notes:
            lines.append(f"- `{row['method']}` failed_samples=`{failed or '-'}` notes=`{notes or '-'}`")
    if not any(str(row.get("failed_samples", "")) or str(row.get("notes", "")) for row in summary["methods"]):
        lines.append("- No method-level failure notes were recorded.")

    audit = dict(summary.get("failure_audit", {}))
    lines.extend(
        [
            "",
            "## Audit Aggregates",
            "",
            f"- Failure audit count: `{audit.get('failure_count', 0)}`",
            f"- By method: `{audit.get('by_method', {})}`",
            f"- By reason: `{audit.get('by_reason', {})}`",
            "",
        ]
    )
    return "\n".join(lines)


GROUP_METRICS = [
    "semantic_pass_rate",
    "mean_semantic_score",
    "mean_potency_score",
    "mean_verifier_coverage",
    "mean_llvm_block_coverage",
    "mean_branch_rewrite_ratio",
    "mean_ir_structural_delta",
    "failure_count",
    "tool_incompatible_count",
]


def _safe_float(value: Any) -> float:
    try:
        if value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_grouped_summary(
    rows: list[dict[str, Any]],
    *,
    group_by: list[str],
    aggregate_by: list[str],
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(_row_group_value(row, field) for field in group_by)
        grouped[key].append(row)

    groups: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        group: dict[str, Any] = {
            "group": {field: value for field, value in zip(group_by, key)},
            "group_key": "|".join(key),
            "row_count": len(items),
            "aggregate_by": list(aggregate_by),
            "replicates": {
                field: _sort_replicate_values({str(item.get(field, "")) for item in items if str(item.get(field, ""))})
                for field in aggregate_by
            },
        }
        for metric in GROUP_METRICS:
            values = [_safe_float(item.get(metric, 0.0)) for item in items]
            group[metric] = {
                "mean": _round(mean(values)) if values else 0.0,
                "std": _round(stdev(values)) if len(values) > 1 else 0.0,
                "min": _round(min(values)) if values else 0.0,
                "max": _round(max(values)) if values else 0.0,
            }
        groups.append(group)
    return {
        "group_by": list(group_by),
        "aggregate_by": list(aggregate_by),
        "groups": groups,
    }


def render_grouped_markdown(summary: dict[str, Any]) -> str:
    group_by = list(summary.get("group_by", []))
    headers = [
        *group_by,
        "rows",
        "seeds",
        "semantic_pass_rate",
        "mean_semantic_score",
        "mean_potency_score",
        "mean_verifier_coverage",
        "mean_llvm_block_coverage",
        "mean_branch_rewrite_ratio",
        "mean_ir_structural_delta",
        "failure_count",
    ]
    lines = [
        "# Grouped Experiment Summary",
        "",
        f"- Group by: `{','.join(group_by)}`",
        f"- Aggregate by: `{','.join(summary.get('aggregate_by', []))}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for group in summary.get("groups", []):
        group_values = [f"`{dict(group.get('group', {})).get(field, '')}`" for field in group_by]
        replicates = dict(group.get("replicates", {}))
        seeds = ",".join(replicates.get("seed", [])) or "-"
        row = [
            *group_values,
            f"`{group.get('row_count', 0)}`",
            f"`{seeds}`",
        ]
        for metric in headers[len(group_by) + 2 :]:
            values = dict(group.get(metric, {}))
            row.append(f"`{values.get('mean', 0.0):.6f} +/- {values.get('std', 0.0):.6f}`")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_grouped_csv(path: Path, grouped_summary: dict[str, Any]) -> None:
    group_by = list(grouped_summary.get("group_by", []))
    fieldnames = [
        *group_by,
        "row_count",
        "aggregate_by",
        "replicates",
    ]
    for metric in GROUP_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"])
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in grouped_summary.get("groups", []):
            group_values = dict(group.get("group", {}))
            row = {
                **{field: group_values.get(field, "") for field in group_by},
                "row_count": group.get("row_count", 0),
                "aggregate_by": ",".join(group.get("aggregate_by", [])),
                "replicates": json.dumps(group.get("replicates", {}), sort_keys=True),
            }
            for metric in GROUP_METRICS:
                values = dict(group.get(metric, {}))
                row[f"{metric}_mean"] = values.get("mean", 0.0)
                row[f"{metric}_std"] = values.get("std", 0.0)
                row[f"{metric}_min"] = values.get("min", 0.0)
                row[f"{metric}_max"] = values.get("max", 0.0)
            writer.writerow(row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    tigress_diagnosis = load_tigress_diagnosis(args.tigress_diagnosis)

    methods: list[dict[str, Any]] = []
    seen_methods: defaultdict[str, int] = defaultdict(int)
    for value in args.baseline_summary:
        path = resolve_path(value)
        summary = _load_dict(path)
        for _, result_value in sorted(dict(summary.get("results", {})).items()):
            if not isinstance(result_value, dict):
                continue
            if result_value.get("status") in {"skipped_tool_missing", "skipped_not_configured", "skipped_unknown_baseline"}:
                continue
            row = summarize_baseline_result(
                dict(result_value),
                source_path=path,
                semantic_threshold=float(args.semantic_threshold),
                tigress_diagnosis=tigress_diagnosis,
            )
            seen_methods[row["method"]] += 1
            if seen_methods[row["method"]] > 1:
                row["method"] = f"{row['method']}#{seen_methods[row['method']]}"
            methods.append(row)

    for value in args.ppo_run_dir:
        run_dir = resolve_path(value)
        row = summarize_ppo_run(run_dir, semantic_threshold=float(args.semantic_threshold))
        seen_methods[row["method"]] += 1
        if seen_methods[row["method"]] > 1:
            row["method"] = f"{row['method']}#{seen_methods[row['method']]}"
        methods.append(row)

    summary = {
        "created_at": utc_timestamp(),
        "job_name": str(args.job_name),
        "semantic_threshold": float(args.semantic_threshold),
        "inputs": {
            "baseline_summaries": [project_relative(resolve_path(value)) for value in args.baseline_summary],
            "ppo_run_dirs": [project_relative(resolve_path(value)) for value in args.ppo_run_dir],
            "failure_audits": [project_relative(resolve_path(value)) for value in args.failure_audit],
            "tigress_diagnoses": [project_relative(resolve_path(value)) for value in args.tigress_diagnosis],
        },
        "methods": methods,
        "failure_audit": load_failure_audits(args.failure_audit),
    }
    group_by = _split_csv_fields(args.group_by)
    aggregate_by = _split_csv_fields(args.aggregate)
    grouped_summary = build_grouped_summary(methods, group_by=group_by, aggregate_by=aggregate_by) if group_by else None

    json_path = output_dir / "experiment_summary.json"
    csv_path = output_dir / "experiment_summary.csv"
    md_path = output_dir / "experiment_summary.md"
    dump_json(json_path, summary)
    write_csv(csv_path, methods)
    write_text(md_path, render_markdown(summary))
    print(f"Wrote experiment summary JSON: {json_path}")
    print(f"Wrote experiment summary CSV: {csv_path}")
    print(f"Wrote experiment summary Markdown: {md_path}")
    if grouped_summary is not None:
        grouped_json_path = output_dir / "grouped_summary.json"
        grouped_csv_path = output_dir / "grouped_summary.csv"
        grouped_md_path = output_dir / "grouped_summary.md"
        dump_json(grouped_json_path, grouped_summary)
        write_grouped_csv(grouped_csv_path, grouped_summary)
        write_text(grouped_md_path, render_grouped_markdown(grouped_summary))
        print(f"Wrote grouped summary JSON: {grouped_json_path}")
        print(f"Wrote grouped summary CSV: {grouped_csv_path}")
        print(f"Wrote grouped summary Markdown: {grouped_md_path}")


if __name__ == "__main__":
    main()
