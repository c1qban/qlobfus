from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a larger harness-clean formal evaluation index.")
    parser.add_argument(
        "--split-index",
        action="append",
        required=True,
        help="Input split index. Pass validation and test indexes; can be repeated.",
    )
    parser.add_argument(
        "--harness-audit",
        default="",
        help="Optional harness_quality_audit.json. Only status=ok samples are selected when provided.",
    )
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--job-name", default="expanded_harness_clean_eval")
    parser.add_argument("--target-samples", type=int, default=150)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--max-samples-per-problem", type=int, default=4)
    parser.add_argument("--exclude-sample-id", action="append", default=[], help="Sample id to exclude from selection.")
    parser.add_argument("--require-harness", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _as_dict(path_value: str | Path) -> dict[str, Any]:
    data = load_structured_file(path_value)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path_value}")
    return data


def _problem_id(sample: dict[str, Any]) -> str:
    metadata = dict(sample.get("metadata", {}))
    return str(metadata.get("problem_id") or sample.get("split_group") or str(sample.get("sample_id", "")).split("/", 1)[0])


def _load_candidate_samples(paths: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    source_indexes: list[str] = []
    seen: set[str] = set()
    for path_text in paths:
        path = resolve_path(path_text)
        payload = _as_dict(path)
        source_indexes.append(project_relative(path))
        for raw in payload.get("samples", []):
            if not isinstance(raw, dict):
                continue
            sample = dict(raw)
            sample_id = str(sample.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                continue
            seen.add(sample_id)
            samples.append(sample)
    return samples, source_indexes


def _audit_status_lookup(audit_path: str) -> dict[str, str]:
    if not audit_path:
        return {}
    path = resolve_path(audit_path)
    if not path.exists():
        raise FileNotFoundError(f"Harness audit not found: {path}")
    payload = _as_dict(path)
    lookup: dict[str, str] = {}
    for item in payload.get("samples", []):
        if isinstance(item, dict) and item.get("sample_id"):
            lookup[str(item["sample_id"])] = str(item.get("status", "unknown"))
    return lookup


def _eligible(sample: dict[str, Any], audit_lookup: dict[str, str], require_harness: bool) -> tuple[bool, str]:
    harness = dict(sample.get("harness", {}))
    if require_harness and (not harness.get("exists") or int(harness.get("case_count", 0) or 0) <= 0):
        return False, "missing_or_empty_harness"
    if audit_lookup:
        status = audit_lookup.get(str(sample.get("sample_id", "")), "missing_audit")
        if status != "ok":
            return False, f"harness_audit_{status}"
    return True, ""


def select_samples(
    samples: list[dict[str, Any]],
    *,
    audit_lookup: dict[str, str],
    target_samples: int,
    max_samples_per_problem: int,
    require_harness: bool,
    excluded_sample_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    per_problem = Counter()
    excluded_sample_ids = excluded_sample_ids or set()
    for sample in sorted(samples, key=lambda item: (_problem_id(item), str(item.get("sample_id", "")))):
        sample_id = str(sample.get("sample_id", ""))
        if sample_id in excluded_sample_ids:
            rejected.append({"sample_id": sample_id, "problem_id": _problem_id(sample), "reason": "excluded_sample_id"})
            continue
        ok, reason = _eligible(sample, audit_lookup, require_harness)
        if not ok:
            rejected.append({"sample_id": sample.get("sample_id", ""), "problem_id": _problem_id(sample), "reason": reason})
            continue
        problem_id = _problem_id(sample)
        if per_problem[problem_id] >= max_samples_per_problem:
            rejected.append({"sample_id": sample.get("sample_id", ""), "problem_id": problem_id, "reason": "max_samples_per_problem"})
            continue
        selected.append(sample)
        per_problem[problem_id] += 1
        if len(selected) >= target_samples:
            break
    return selected, rejected


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Expanded Harness-Clean Eval",
        "",
        f"- job: `{report['job_name']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- output_index: `{report['output_index']}`",
        f"- target_samples: `{report['target_samples']}`",
        f"- min_samples: `{report['min_samples']}`",
        f"- selected_samples: `{report['selected_sample_count']}`",
        f"- selected_problems: `{report['selected_problem_count']}`",
        f"- status: `{report['status']}`",
        "",
        "## Source Indexes",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report["source_indexes"])
    lines.extend(
        [
            "",
            "## Rejection Breakdown",
            "",
            "| reason | count |",
            "| --- | ---: |",
        ]
    )
    for reason, count in report["rejection_breakdown"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Use validation/test indexes as input to preserve problem-level holdout from training.",
            "- If selected sample count is below `min_samples`, first expand the prepared Project CodeNet subset or generate more harnesses.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_index = resolve_path(args.output_index)
    output_report = resolve_path(args.output_report) if args.output_report else output_index.with_suffix(".report.md")
    samples, source_indexes = _load_candidate_samples(list(args.split_index))
    audit_lookup = _audit_status_lookup(str(args.harness_audit))
    selected, rejected = select_samples(
        samples,
        audit_lookup=audit_lookup,
        target_samples=int(args.target_samples),
        max_samples_per_problem=int(args.max_samples_per_problem),
        require_harness=bool(args.require_harness),
        excluded_sample_ids={str(item) for item in args.exclude_sample_id},
    )
    problem_ids = sorted({_problem_id(sample) for sample in selected})
    status = "ready" if len(selected) >= int(args.min_samples) else "below_min_samples"
    payload = {
        "job_name": str(args.job_name),
        "created_at": utc_timestamp(),
        "split": "formal_eval_harness_clean",
        "source_indexes": source_indexes,
        "harness_audit": project_relative(resolve_path(args.harness_audit)) if args.harness_audit else "",
        "target_samples": int(args.target_samples),
        "min_samples": int(args.min_samples),
        "max_samples_per_problem": int(args.max_samples_per_problem),
        "excluded_sample_ids": sorted(str(item) for item in args.exclude_sample_id),
        "status": status,
        "sample_count": len(selected),
        "problem_count": len(problem_ids),
        "problem_ids": problem_ids,
        "samples": selected,
    }
    ensure_dir(output_index.parent)
    dump_json(output_index, payload)
    report = {
        "job_name": str(args.job_name),
        "generated_at": utc_timestamp(),
        "output_index": project_relative(output_index),
        "source_indexes": source_indexes,
        "target_samples": int(args.target_samples),
        "min_samples": int(args.min_samples),
        "selected_sample_count": len(selected),
        "selected_problem_count": len(problem_ids),
        "status": status,
        "rejection_breakdown": dict(sorted(Counter(item["reason"] for item in rejected).items())),
        "rejected_preview": rejected[:50],
    }
    dump_json(output_report.with_suffix(".json"), report)
    write_text(output_report, _markdown(report))
    print(f"Wrote expanded harness-clean index: {output_index}")
    print(f"Wrote report: {output_report}")
    if status != "ready":
        print(f"WARNING: selected {len(selected)} samples, below min_samples={args.min_samples}")
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
