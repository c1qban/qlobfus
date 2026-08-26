from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


DEFAULT_SPLIT_INDEX = "data/processed/project_codenet_c_100x4_strict/splits/formal_eval_30_v2.index.json"
DEFAULT_MANUAL_REVIEW = "artifacts/formal_eval_30_v2/harness_quality_audit/manual_review.json"
DEFAULT_OUTPUT_INDEX = "data/processed/project_codenet_c_100x4_strict/splits/formal_eval_30_v3_harness_clean.index.json"
DEFAULT_OUTPUT_DIR = "artifacts/formal_eval_30_v3_harness_clean"
DEFAULT_BASELINE_SUMMARIES = [
    "artifacts/formal_eval_30_v2/baselines_run/reports/baseline_summary.json",
    "artifacts/formal_eval_30_v2/tigress_run/reports/baseline_summary.json",
]
DEFAULT_CORRECTED_SUMMARY = "artifacts/formal_eval_30_v2/corrected_summaries/formal_eval_30_v2_corrected_summary.json"


ROW_FIELDS = [
    "method",
    "variant",
    "seed",
    "source_kind",
    "source_path",
    "episode_count",
    "clean_episode_count",
    "semantic_pass_count",
    "semantic_pass_rate",
    "mean_reward",
    "mean_semantic_score",
    "mean_verifier_coverage",
    "mean_potency_score",
    "mean_llvm_block_coverage",
    "mean_branch_rewrite_ratio",
    "mean_ir_structural_delta",
    "failure_count",
    "failed_samples",
]


AGG_FIELDS = [
    "method",
    "variant",
    "run_count",
    "seeds",
    "clean_episode_count",
    "semantic_pass_rate_mean",
    "semantic_pass_rate_std",
    "mean_semantic_score_mean",
    "mean_verifier_coverage_mean",
    "mean_potency_score_mean",
    "mean_llvm_block_coverage_mean",
    "mean_branch_rewrite_ratio_mean",
    "mean_ir_structural_delta_mean",
    "failure_count_mean",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create harness-clean formal eval index and post-hoc summary.")
    parser.add_argument("--split-index", default=DEFAULT_SPLIT_INDEX)
    parser.add_argument("--manual-review", default=DEFAULT_MANUAL_REVIEW)
    parser.add_argument("--output-index", default=DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-summary", action="append", default=list(DEFAULT_BASELINE_SUMMARIES))
    parser.add_argument("--corrected-summary", default=DEFAULT_CORRECTED_SUMMARY)
    parser.add_argument("--semantic-threshold", type=float, default=0.97)
    return parser.parse_args(argv)


def _as_dict(path_value: str | Path) -> dict[str, Any]:
    data = load_structured_file(path_value)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path_value}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _metric(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in row:
            return float(row.get(name) or 0.0)
    return 0.0


def _summarize_episodes(
    *,
    method: str,
    variant: str,
    seed: str,
    source_kind: str,
    source_path: Path,
    episodes: list[dict[str, Any]],
    clean_sample_ids: set[str],
    semantic_threshold: float,
) -> dict[str, Any]:
    clean = [dict(item) for item in episodes if str(item.get("sample_id")) in clean_sample_ids]
    failed = [str(item.get("sample_id", "unknown")) for item in clean if _metric(item, "semantic_score") < semantic_threshold]
    return {
        "method": method,
        "variant": variant,
        "seed": seed,
        "source_kind": source_kind,
        "source_path": project_relative(source_path),
        "episode_count": len(episodes),
        "clean_episode_count": len(clean),
        "semantic_pass_count": len(clean) - len(failed),
        "semantic_pass_rate": _round((len(clean) - len(failed)) / len(clean)) if clean else 0.0,
        "mean_reward": _round(_safe_mean([_metric(item, "reward", "episode_reward") for item in clean])),
        "mean_semantic_score": _round(_safe_mean([_metric(item, "semantic_score") for item in clean])),
        "mean_verifier_coverage": _round(_safe_mean([_metric(item, "verifier_coverage") for item in clean])),
        "mean_potency_score": _round(_safe_mean([_metric(item, "potency_score") for item in clean])),
        "mean_llvm_block_coverage": _round(_safe_mean([_metric(item, "llvm_block_coverage") for item in clean])),
        "mean_branch_rewrite_ratio": _round(_safe_mean([_metric(item, "branch_rewrite_ratio") for item in clean])),
        "mean_ir_structural_delta": _round(_safe_mean([_metric(item, "ir_structural_delta") for item in clean])),
        "failure_count": len(failed),
        "failed_samples": ", ".join(sorted(set(failed))) if failed else "-",
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["variant"]))].append(row)
    aggregates: list[dict[str, Any]] = []
    for (method, variant), items in sorted(grouped.items()):
        def values(name: str) -> list[float]:
            return [float(item.get(name, 0.0)) for item in items]

        aggregates.append(
            {
                "method": method,
                "variant": variant,
                "run_count": len(items),
                "seeds": ",".join(str(item["seed"]) for item in items if str(item["seed"])),
                "clean_episode_count": int(sum(int(item["clean_episode_count"]) for item in items)),
                "semantic_pass_rate_mean": _round(_safe_mean(values("semantic_pass_rate"))),
                "semantic_pass_rate_std": _round(_safe_std(values("semantic_pass_rate"))),
                "mean_semantic_score_mean": _round(_safe_mean(values("mean_semantic_score"))),
                "mean_verifier_coverage_mean": _round(_safe_mean(values("mean_verifier_coverage"))),
                "mean_potency_score_mean": _round(_safe_mean(values("mean_potency_score"))),
                "mean_llvm_block_coverage_mean": _round(_safe_mean(values("mean_llvm_block_coverage"))),
                "mean_branch_rewrite_ratio_mean": _round(_safe_mean(values("mean_branch_rewrite_ratio"))),
                "mean_ir_structural_delta_mean": _round(_safe_mean(values("mean_ir_structural_delta"))),
                "failure_count_mean": _round(_safe_mean(values("failure_count"))),
            }
        )
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Formal Eval 30 v3 Harness-Clean Summary",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- source_index: `{payload['source_index']}`",
        f"- clean_index: `{payload['clean_index']}`",
        f"- original_sample_count: `{payload['index_summary']['original_sample_count']}`",
        f"- clean_sample_count: `{payload['index_summary']['clean_sample_count']}`",
        f"- removed_sample_count: `{payload['index_summary']['removed_sample_count']}`",
        "",
        "## Removed Samples",
        "",
        "| sample | decision | reason |",
        "| --- | --- | --- |",
    ]
    for item in payload["removed_samples"]:
        lines.append(f"| `{item['sample_id']}` | {item['decision']} | {item['reason']} |")

    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            "| method | variant | seeds | clean episodes | semantic pass | semantic score | verifier coverage | potency | LLVM block | IR delta | failures |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["aggregates"]:
        lines.append(
            "| {method} | {variant} | {seeds} | {clean_episode_count} | {semantic_pass_rate_mean:.6f} ± {semantic_pass_rate_std:.6f} | "
            "{mean_semantic_score_mean:.6f} | {mean_verifier_coverage_mean:.6f} | {mean_potency_score_mean:.6f} | "
            "{mean_llvm_block_coverage_mean:.6f} | {mean_ir_structural_delta_mean:.6f} | {failure_count_mean:.6f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "This is a post-hoc harness-clean summary computed from existing v2 baseline and corrected PPO episode logs. It is suitable for fast paper sanity checks. A full v3 rerun is stronger if time allows.",
        ]
    )
    return "\n".join(lines) + "\n"


def _create_clean_index(
    split_index: dict[str, Any],
    manual_review: dict[str, Any],
    output_index: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    removed_samples = [
        dict(item)
        for item in manual_review.get("samples", [])
        if str(item.get("decision", "")).lower() in {"weak_harness", "invalid_harness"}
    ]
    removed_ids = {str(item["sample_id"]) for item in removed_samples}
    original_samples = [dict(item) for item in split_index.get("samples", []) if isinstance(item, dict)]
    clean_samples = [item for item in original_samples if str(item.get("sample_id")) not in removed_ids]
    clean_problem_ids = sorted({str(dict(item.get("metadata", {})).get("problem_id", str(item.get("sample_id", "")).split("/", 1)[0])) for item in clean_samples})

    payload = dict(split_index)
    payload["job_name"] = "project_codenet_c_100x4_formal_eval_30_v3_harness_clean"
    payload["created_at"] = utc_timestamp()
    payload["source_index"] = project_relative(resolve_path(DEFAULT_SPLIT_INDEX))
    payload["harness_cleaning"] = {
        "manual_review": project_relative(resolve_path(DEFAULT_MANUAL_REVIEW)),
        "removed_sample_ids": sorted(removed_ids),
        "reason": "Removed weak/invalid harness samples identified by automatic audit and manual review.",
    }
    payload["samples"] = clean_samples
    payload["sample_count"] = len(clean_samples)
    payload["problem_count"] = len(clean_problem_ids)
    payload["clean_problem_ids"] = clean_problem_ids
    ensure_dir(output_index.parent)
    dump_json(output_index, payload)
    return payload, removed_samples


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_index_path = resolve_path(args.split_index)
    manual_review_path = resolve_path(args.manual_review)
    output_index = resolve_path(args.output_index)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    split_index = _as_dict(split_index_path)
    manual_review = _as_dict(manual_review_path)

    clean_index, removed_samples = _create_clean_index(split_index, manual_review, output_index)
    clean_sample_ids = {str(item.get("sample_id")) for item in clean_index.get("samples", [])}

    rows: list[dict[str, Any]] = []
    for summary_text in args.baseline_summary:
        path = resolve_path(summary_text)
        if not path.exists():
            continue
        summary = _as_dict(path)
        for method, result in dict(summary.get("results", {})).items():
            episodes = [dict(item) for item in dict(result).get("episodes", []) if isinstance(item, dict)]
            rows.append(
                _summarize_episodes(
                    method=str(method),
                    variant="baseline",
                    seed="",
                    source_kind="baseline_summary",
                    source_path=path,
                    episodes=episodes,
                    clean_sample_ids=clean_sample_ids,
                    semantic_threshold=args.semantic_threshold,
                )
            )

    corrected_summary_path = resolve_path(args.corrected_summary)
    if corrected_summary_path.exists():
        corrected_summary = _as_dict(corrected_summary_path)
        for run in corrected_summary.get("runs", []):
            if not isinstance(run, dict):
                continue
            path = resolve_path(str(run.get("corrected_episode_path", "")))
            episodes = _read_jsonl(path)
            rows.append(
                _summarize_episodes(
                    method="maskable_ppo",
                    variant=str(run.get("variant", "")),
                    seed=str(run.get("seed", "")),
                    source_kind="corrected_episode_metrics",
                    source_path=path,
                    episodes=episodes,
                    clean_sample_ids=clean_sample_ids,
                    semantic_threshold=args.semantic_threshold,
                )
            )

    payload = {
        "summary_type": "formal_eval_30_v3_harness_clean_posthoc",
        "generated_at": utc_timestamp(),
        "semantic_threshold": args.semantic_threshold,
        "source_index": project_relative(split_index_path),
        "clean_index": project_relative(output_index),
        "index_summary": {
            "original_sample_count": len(split_index.get("samples", [])),
            "clean_sample_count": len(clean_index.get("samples", [])),
            "removed_sample_count": len(removed_samples),
            "clean_problem_count": clean_index.get("problem_count", 0),
        },
        "removed_samples": removed_samples,
        "rows": rows,
        "aggregates": _aggregate(rows),
    }
    dump_json(output_dir / "harness_clean_summary.json", payload)
    _write_csv(output_dir / "harness_clean_runs.csv", rows, ROW_FIELDS)
    _write_csv(output_dir / "harness_clean_aggregates.csv", payload["aggregates"], AGG_FIELDS)
    write_text(output_dir / "harness_clean_summary.md", _markdown(payload))
    print(f"Wrote clean index: {output_index}")
    print(f"Wrote harness-clean summary: {output_dir / 'harness_clean_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
