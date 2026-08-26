from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text
from verifier.pipeline import VerificationPipeline


DEFAULT_SPLIT_INDEX = "data/processed/project_codenet_c_100x4_strict/splits/formal_eval_30_v2.index.json"
DEFAULT_MANIFEST = "data/manifests/project_codenet_c_100x4_strict.manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit stdio harness quality for a split index.")
    parser.add_argument("--split-index", default=DEFAULT_SPLIT_INDEX)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default="artifacts/formal_eval_30_v2/harness_quality_audit")
    parser.add_argument("--include-excluded", action="store_true", help="Also audit excluded_sample_ids when present.")
    parser.add_argument("--fuzz-max-cases", type=int, default=8)
    parser.add_argument("--no-fuzz", action="store_true")
    return parser.parse_args(argv)


def _as_dict(path_value: str | Path) -> dict[str, Any]:
    data = load_structured_file(path_value)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path_value}")
    return data


def _load_harness(path_value: str | Path | None) -> dict[str, Any]:
    if not path_value:
        return {}
    path = resolve_path(path_value)
    if not path.exists():
        return {}
    data = load_structured_file(path)
    return dict(data) if isinstance(data, dict) else {}


def _sample_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for section in ("samples", "filtered_out_samples"):
        for item in manifest.get(section, []):
            if isinstance(item, dict) and item.get("sample_id"):
                lookup[str(item["sample_id"])] = dict(item)
    return lookup


def _count_scanf_conversions(source_text: str) -> int:
    total = 0
    for match in re.finditer(r"\b(?:scanf|fscanf)\s*\(\s*\"((?:\\.|[^\"\\])*)\"", source_text):
        fmt = match.group(1)
        for spec in re.finditer(r"%(?!%)(\*)?(?:\d+)?(?:hh|h|ll|l|L|z|j|t)?[diuoxXfFeEgGaAcspn\[]", fmt):
            if spec.group(1) != "*":
                total += 1
    return total


def _first_scanf_conversion_count(source_text: str) -> int:
    match = re.search(r"\b(?:scanf|fscanf)\s*\(\s*\"((?:\\.|[^\"\\])*)\"", source_text)
    if not match:
        return 0
    fmt = match.group(1)
    return sum(
        1
        for spec in re.finditer(r"%(?!%)(\*)?(?:\d+)?(?:hh|h|ll|l|L|z|j|t)?[diuoxXfFeEgGaAcspn\[]", fmt)
        if spec.group(1) != "*"
    )


def _input_tokens(input_data: str) -> list[str]:
    return re.findall(r"[^\s,]+", input_data)


def _looks_like_binary_grid(input_data: str) -> tuple[bool, int, int]:
    lines = [line.strip() for line in input_data.splitlines() if line.strip()]
    binary_rows = [line for line in lines if re.fullmatch(r"[01]+", line)]
    if not binary_rows:
        return False, 0, 0
    widths = {len(line) for line in binary_rows}
    return len(binary_rows) == len(lines) and len(widths) == 1, len(binary_rows), next(iter(widths))


def _static_issues(source_text: str, cases: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    first_scanf_count = _first_scanf_conversion_count(source_text)
    total_scanf_count = _count_scanf_conversions(source_text)
    source_reads_string_rows = bool(re.search(r'scanf\s*\(\s*"%s"', source_text))

    min_tokens = min((_input_tokens(str(case.get("input_data", ""))) for case in cases), key=len, default=[])
    if first_scanf_count and len(min_tokens) < first_scanf_count:
        issues.append(
            f"case input has fewer tokens than first scanf needs: min_tokens={len(min_tokens)}, first_scanf={first_scanf_count}"
        )

    binary_grid_cases = []
    for case in cases:
        is_grid, rows, width = _looks_like_binary_grid(str(case.get("input_data", "")))
        if is_grid:
            binary_grid_cases.append({"name": case.get("name", ""), "rows": rows, "width": width})
            if source_reads_string_rows and rows != width:
                issues.append(f"binary grid is not square in case {case.get('name', '')}: rows={rows}, width={width}")
            if source_reads_string_rows and rows < 2:
                issues.append(f"binary grid has too few rows in case {case.get('name', '')}: rows={rows}")

    return issues, {
        "first_scanf_conversion_count": first_scanf_count,
        "total_static_scanf_conversion_count": total_scanf_count,
        "case_count": len(cases),
        "min_input_token_count": len(min_tokens),
        "binary_grid_cases": binary_grid_cases,
    }


def _case_list(harness: dict[str, Any]) -> list[dict[str, Any]]:
    raw_cases = harness.get("test_cases", [])
    return [dict(item) for item in raw_cases if isinstance(item, dict)]


def _audit_sample(
    sample: dict[str, Any],
    *,
    pipeline: VerificationPipeline,
    fuzz_max_cases: int,
    fuzz_enabled: bool,
    exclusion_reason: str | None = None,
) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id", "unknown"))
    source_path = resolve_path(str(sample.get("source_path", "")))
    harness_path = str(dict(sample.get("harness", {})).get("path", ""))
    harness = _load_harness(harness_path)
    cases = _case_list(harness)
    issues: list[str] = []

    if not source_path.exists():
        issues.append("source file missing")
        source_text = ""
    else:
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
    if not harness:
        issues.append("harness file missing or unreadable")
    if not cases:
        issues.append("harness has no executable test cases")

    static_issues, static_metrics = _static_issues(source_text, cases)
    issues.extend(static_issues)

    verification: dict[str, Any] = {
        "executed": False,
        "semantic_score": 0.0,
        "verifier_coverage": 0.0,
        "compile_succeeded": False,
        "tests_passed": False,
        "diff_passed": False,
        "fuzz_passed": False,
        "fuzz_executed": False,
        "notes": [],
    }
    if source_path.exists() and cases:
        summary = pipeline.verify_source(
            source_path=source_path,
            reference_source_path=source_path,
            test_cases=cases,
            sample_id=sample_id.replace("/", "_"),
            compiler=str(dict(sample.get("compile", {})).get("compiler", "")) or None,
            compiler_flags=list(dict(sample.get("compile", {})).get("compiler_flags", [])),
            compile_timeout_sec=float(dict(sample.get("compile", {})).get("timeout_sec", 15.0)),
            fuzz_max_cases=fuzz_max_cases,
            fuzz_enabled=fuzz_enabled,
        )
        verification = {
            "executed": True,
            "semantic_score": summary.semantic_score,
            "verifier_coverage": summary.verifier_coverage,
            "compile_succeeded": summary.compile.succeeded,
            "tests_passed": summary.tests.passed,
            "diff_passed": summary.diff.passed,
            "fuzz_passed": summary.fuzz.passed,
            "fuzz_executed": summary.fuzz.executed,
            "notes": summary.notes + summary.tests.notes + summary.diff.notes + summary.fuzz.notes,
        }
        if not summary.compile.succeeded:
            issues.append("reference source does not compile under harness audit settings")
        if not summary.tests.passed:
            issues.append("reference source does not pass its own harness")
        if summary.diff.executed and not summary.diff.passed:
            issues.append("self differential execution failed")
        if summary.fuzz.executed and not summary.fuzz.passed:
            issues.append("self fuzz execution failed")

    status = "ok" if not issues else "needs_review"
    if exclusion_reason:
        status = "excluded_known_issue"

    return {
        "sample_id": sample_id,
        "problem_id": str(dict(sample.get("metadata", {})).get("problem_id", sample_id.split("/", 1)[0])),
        "split": sample.get("split", ""),
        "source_path": project_relative(source_path),
        "harness_path": harness_path,
        "exclusion_reason": exclusion_reason or "",
        "status": status,
        "issues": sorted(set(issues)),
        "static_metrics": static_metrics,
        "verification": verification,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Harness Quality Audit",
        "",
        f"- job: `{report['job_name']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- split_index: `{report['split_index']}`",
        f"- audited_samples: `{report['summary']['audited_samples']}`",
        f"- ok: `{report['summary']['ok']}`",
        f"- needs_review: `{report['summary']['needs_review']}`",
        f"- excluded_known_issue: `{report['summary']['excluded_known_issue']}`",
        "",
        "## Findings",
        "",
        "| status | sample | split | issues |",
        "| --- | --- | --- | --- |",
    ]
    for item in report["samples"]:
        issues = "; ".join(item["issues"]) or "-"
        lines.append(f"| {item['status']} | `{item['sample_id']}` | {item.get('split', '')} | {issues} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `needs_review` does not always mean the sample is unusable; it means the harness deserves manual inspection before being used as strong semantic evidence.",
            "- `excluded_known_issue` samples are included for transparency when `--include-excluded` is used.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_index_path = resolve_path(args.split_index)
    manifest_path = resolve_path(args.manifest)
    output_dir = ensure_dir(resolve_path(args.output_dir))

    split_index = _as_dict(split_index_path)
    manifest = _as_dict(manifest_path)
    lookup = _sample_lookup(manifest)
    samples = [dict(item) for item in split_index.get("samples", []) if isinstance(item, dict)]

    if args.include_excluded:
        reasons = dict(split_index.get("exclusion_reasons", {}))
        existing = {str(item.get("sample_id")) for item in samples}
        for sample_id in split_index.get("excluded_sample_ids", []):
            if sample_id in lookup and sample_id not in existing:
                item = dict(lookup[sample_id])
                item["_exclusion_reason"] = str(reasons.get(sample_id, "excluded by split index"))
                samples.append(item)

    pipeline = VerificationPipeline(
        semantic_threshold=0.97,
        build_root=output_dir / "builds",
        cache_root=output_dir / "cache",
        cache_enabled=True,
    )
    audited = [
        _audit_sample(
            sample,
            pipeline=pipeline,
            fuzz_max_cases=args.fuzz_max_cases,
            fuzz_enabled=not args.no_fuzz,
            exclusion_reason=sample.get("_exclusion_reason"),
        )
        for sample in samples
    ]
    status_counts = Counter(item["status"] for item in audited)
    report = {
        "job_name": "harness_quality_audit",
        "generated_at": utc_timestamp(),
        "split_index": project_relative(split_index_path),
        "manifest": project_relative(manifest_path),
        "fuzz_enabled": not args.no_fuzz,
        "fuzz_max_cases": args.fuzz_max_cases,
        "summary": {
            "audited_samples": len(audited),
            "ok": status_counts.get("ok", 0),
            "needs_review": status_counts.get("needs_review", 0),
            "excluded_known_issue": status_counts.get("excluded_known_issue", 0),
        },
        "samples": audited,
    }

    dump_json(output_dir / "harness_quality_audit.json", report)
    write_text(output_dir / "harness_quality_audit.md", _markdown(report))
    print(f"Wrote harness quality audit: {output_dir / 'harness_quality_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
