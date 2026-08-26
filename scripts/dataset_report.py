from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import dump_json, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dataset audit report from a prepared manifest.")
    parser.add_argument("--manifest", required=True, help="Path to a dataset manifest JSON file.")
    parser.add_argument("--output-json", help="Optional explicit report JSON path.")
    parser.add_argument("--output-md", help="Optional explicit report Markdown path.")
    return parser.parse_args(argv)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _problem_id(sample: dict[str, Any]) -> str:
    metadata = dict(sample.get("metadata", {}))
    if metadata.get("problem_id"):
        return str(metadata["problem_id"])
    split_group = str(sample.get("split_group", ""))
    if split_group:
        return split_group
    sample_id = str(sample.get("sample_id", ""))
    return sample_id.split("/", 1)[0] if "/" in sample_id else sample_id


def _source_hash(sample: dict[str, Any]) -> str:
    source_path = resolve_path(str(sample.get("source_path", "")))
    if not source_path.exists():
        return ""
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _line_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    line_counts = [float(dict(sample.get("source_stats", {})).get("line_count", 0)) for sample in samples]
    nonzero_counts = [value for value in line_counts if value > 0]
    return {
        "min": int(min(nonzero_counts)) if nonzero_counts else 0,
        "max": int(max(nonzero_counts)) if nonzero_counts else 0,
        "mean": round(_safe_mean(nonzero_counts), 2),
        "median": round(_safe_median(nonzero_counts), 2),
    }


def _compile_summary(samples: list[dict[str, Any]], filtered_out: list[dict[str, Any]]) -> dict[str, Any]:
    all_samples = samples + filtered_out
    checked = []
    succeeded = []
    failed = []
    for sample in all_samples:
        compile_check = dict(dict(sample.get("prepare_checks", {})).get("compile_check", {}))
        if bool(compile_check.get("checked", False)):
            checked.append(sample)
            if bool(compile_check.get("succeeded", False)):
                succeeded.append(sample)
            else:
                failed.append(sample)
    return {
        "checked_count": len(checked),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "pass_rate": round(len(succeeded) / float(len(checked)), 4) if checked else 0.0,
        "failed_samples": [
            {
                "sample_id": str(sample.get("sample_id", "")),
                "source_path": str(sample.get("source_path", "")),
                "filter_reasons": list(sample.get("filter_reasons", [])),
                "stderr_tail": str(
                    dict(dict(sample.get("prepare_checks", {})).get("compile_check", {})).get("stderr_tail", "")
                ),
            }
            for sample in failed
        ],
    }


def _split_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts: Counter[str] = Counter(str(sample.get("split", "unspecified")) for sample in samples)
    split_problem_map: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        split_problem_map[str(sample.get("split", "unspecified"))].add(_problem_id(sample))
    return {
        "sample_counts": dict(sorted(split_counts.items())),
        "problem_counts": {split: len(problem_ids) for split, problem_ids in sorted(split_problem_map.items())},
        "problems": {split: sorted(problem_ids) for split, problem_ids in sorted(split_problem_map.items())},
    }


def _duplicate_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    hash_to_samples: dict[str, list[str]] = defaultdict(list)
    missing_sources = []
    for sample in samples:
        digest = _source_hash(sample)
        sample_id = str(sample.get("sample_id", ""))
        if not digest:
            missing_sources.append(sample_id)
            continue
        hash_to_samples[digest].append(sample_id)

    duplicate_groups = [
        {"sha256": digest, "samples": sample_ids}
        for digest, sample_ids in sorted(hash_to_samples.items())
        if len(sample_ids) > 1
    ]
    return {
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_sample_count": sum(len(group["samples"]) for group in duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "missing_source_count": len(missing_sources),
        "missing_sources": missing_sources,
    }


def build_dataset_report(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    samples = [dict(sample) for sample in manifest.get("samples", [])]
    filtered_out = [dict(sample) for sample in manifest.get("filtered_out_samples", [])]
    all_entries = samples + filtered_out
    filter_reason_counts: Counter[str] = Counter()
    for sample in filtered_out:
        filter_reason_counts.update(str(reason) for reason in sample.get("filter_reasons", []))

    harnessed = [sample for sample in samples if bool(dict(sample.get("harness", {})).get("exists", False))]
    problem_ids = sorted({_problem_id(sample) for sample in samples if _problem_id(sample)})
    report = {
        "created_at": utc_timestamp(),
        "manifest_path": project_relative(manifest_path),
        "job_name": str(manifest.get("job_name", "unknown")),
        "dataset": dict(manifest.get("dataset", {})),
        "totals": {
            "discovered_sample_count": int(dict(manifest.get("discovery", {})).get("discovered_sample_count", len(all_entries))),
            "kept_sample_count": len(samples),
            "filtered_out_count": len(filtered_out),
            "problem_count": len(problem_ids),
            "harnessed_sample_count": len(harnessed),
            "harness_coverage": round(len(harnessed) / float(len(samples)), 4) if samples else 0.0,
        },
        "splits": _split_summary(samples),
        "line_stats": _line_stats(samples),
        "compile": _compile_summary(samples, filtered_out),
        "filter_reason_counts": dict(sorted(filter_reason_counts.items())),
        "duplicates": _duplicate_summary(samples),
        "notes": [
            "Harness coverage is zero when no harness_root is configured.",
            "Compile pass rate is meaningful only when filters.require_compilable=true was used.",
            "Duplicate detection uses exact SHA-256 source hashes, not near-duplicate detection.",
        ],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    totals = dict(report.get("totals", {}))
    splits = dict(report.get("splits", {}))
    compile_summary = dict(report.get("compile", {}))
    duplicates = dict(report.get("duplicates", {}))
    line_stats = dict(report.get("line_stats", {}))
    split_counts = dict(splits.get("sample_counts", {}))
    problem_counts = dict(splits.get("problem_counts", {}))

    lines = [
        "# Dataset Report",
        "",
        f"- Job: `{report.get('job_name', 'unknown')}`",
        f"- Manifest: `{report.get('manifest_path', '')}`",
        f"- Created at: `{report.get('created_at', '')}`",
        "",
        "## Summary",
        f"- Discovered samples: `{totals.get('discovered_sample_count', 0)}`",
        f"- Kept samples: `{totals.get('kept_sample_count', 0)}`",
        f"- Filtered out samples: `{totals.get('filtered_out_count', 0)}`",
        f"- Problem count: `{totals.get('problem_count', 0)}`",
        f"- Harness coverage: `{totals.get('harness_coverage', 0.0)}`",
        "",
        "## Splits",
    ]
    for split_name in ["train", "validation", "test", "unspecified"]:
        if split_name in split_counts or split_name in problem_counts:
            lines.append(
                f"- `{split_name}`: `{split_counts.get(split_name, 0)}` samples, `{problem_counts.get(split_name, 0)}` problems"
            )

    lines.extend(
        [
            "",
            "## Source Size",
            f"- Min lines: `{line_stats.get('min', 0)}`",
            f"- Max lines: `{line_stats.get('max', 0)}`",
            f"- Mean lines: `{line_stats.get('mean', 0.0)}`",
            f"- Median lines: `{line_stats.get('median', 0.0)}`",
            "",
            "## Compile Filter",
            f"- Checked samples: `{compile_summary.get('checked_count', 0)}`",
            f"- Compile passed: `{compile_summary.get('succeeded_count', 0)}`",
            f"- Compile failed: `{compile_summary.get('failed_count', 0)}`",
            f"- Compile pass rate: `{compile_summary.get('pass_rate', 0.0)}`",
            "",
            "## Duplicates",
            f"- Exact duplicate groups: `{duplicates.get('duplicate_group_count', 0)}`",
            f"- Exact duplicate samples: `{duplicates.get('duplicate_sample_count', 0)}`",
            "",
            "## Caveats",
        ]
    )
    lines.extend(f"- {note}" for note in report.get("notes", []))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = resolve_path(args.manifest)
    manifest = load_structured_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must decode to an object: {manifest_path}")

    report = build_dataset_report(manifest, manifest_path)
    output_json = resolve_path(args.output_json) if args.output_json else manifest_path.with_suffix(".dataset_report.json")
    output_md = resolve_path(args.output_md) if args.output_md else manifest_path.with_suffix(".dataset_report.md")
    dump_json(output_json, report)
    write_text(output_md, render_markdown(report))
    print(f"Wrote dataset report JSON: {output_json}")
    print(f"Wrote dataset report Markdown: {output_md}")


if __name__ == "__main__":
    main()
