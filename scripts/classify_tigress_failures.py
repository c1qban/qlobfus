from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


CSV_FIELDS = [
    "sample_id",
    "category",
    "primary_reason",
    "semantic_score",
    "compile_succeeded",
    "tests_passed",
    "diff_passed",
    "fuzz_passed",
    "transform_status",
    "evidence",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify Tigress baseline failures into paper-ready buckets.")
    parser.add_argument("--baseline-summary", required=True, help="Path to Tigress reports/baseline_summary.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for tigress_failure_breakdown outputs.")
    parser.add_argument("--semantic-threshold", type=float, default=0.97, help="Minimum semantic score counted as pass.")
    return parser.parse_args(argv)


def _load_dict(path: Path) -> dict[str, Any]:
    data = load_structured_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def _clean_text(value: Any, *, limit: int = 260) -> str:
    text = str(value or "")
    text = text.replace("\x00", "")
    text = text.replace("\ufffd", "?")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return "..." + text[-limit:]
    return text


def _verification(episode: dict[str, Any]) -> dict[str, Any]:
    value = episode.get("verification_summary", {})
    return dict(value) if isinstance(value, dict) else {}


def _failed_count(section: dict[str, Any], key: str = "failed_cases") -> int:
    try:
        return int(section.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _section_passed(section: dict[str, Any]) -> bool | None:
    if not section:
        return None
    if "passed" not in section:
        return None
    return bool(section.get("passed"))


def _first_failed_case_note(section: dict[str, Any]) -> str:
    cases = section.get("cases", [])
    if not isinstance(cases, list):
        return ""
    for item in cases:
        if not isinstance(item, dict) or bool(item.get("passed", False)):
            continue
        notes = item.get("notes", [])
        note_text = ",".join(str(note) for note in notes) if isinstance(notes, list) else str(notes)
        name = str(item.get("name", ""))
        mutation = str(item.get("mutation_name", ""))
        return _clean_text(" ".join(part for part in [name, mutation, note_text] if part), limit=180)
    return ""


def classify_episode(episode: dict[str, Any], *, semantic_threshold: float) -> dict[str, Any] | None:
    semantic_score = float(episode.get("semantic_score", 0.0))
    compile_succeeded = bool(episode.get("compile_succeeded", False))
    tests_passed = bool(episode.get("tests_passed", False))
    status = str(episode.get("status", ""))
    transform_stderr = _clean_text(episode.get("transform_stderr", ""), limit=360)
    transform_stdout = _clean_text(episode.get("transform_stdout", ""), limit=220)
    verification = _verification(episode)
    compile_info = dict(verification.get("compile", {})) if isinstance(verification.get("compile", {}), dict) else {}
    tests_info = dict(verification.get("tests", {})) if isinstance(verification.get("tests", {}), dict) else {}
    diff_info = dict(verification.get("diff", {})) if isinstance(verification.get("diff", {}), dict) else {}
    fuzz_info = dict(verification.get("fuzz", {})) if isinstance(verification.get("fuzz", {}), dict) else {}

    if semantic_score >= semantic_threshold and compile_succeeded and tests_passed:
        return None

    lowered = " ".join([status, transform_stderr, transform_stdout]).lower()
    compile_stderr = _clean_text(compile_info.get("stderr", ""), limit=260)
    compile_stdout = _clean_text(compile_info.get("stdout", ""), limit=180)
    evidence_parts: list[str] = []
    category = "semantic_failure"
    primary_reason = "semantic_below_threshold"

    if "timed out" in lowered or "timeout" in lowered:
        category = "tool_timeout"
        primary_reason = "tigress_or_verifier_timeout"
        evidence_parts.append(transform_stderr or transform_stdout)
    elif (
        "parsing error" in lowered
        or "syntax error" in lowered
        or status in {"transform_failed", "skipped_tool_missing"}
    ):
        category = "tool_parse_failure"
        primary_reason = "tigress_parse_or_transform_failed"
        evidence_parts.append(transform_stderr or transform_stdout)
    elif not compile_succeeded:
        category = "tool_compile_failure"
        primary_reason = "tigress_output_compile_failed"
        if compile_stderr or compile_stdout:
            evidence_parts.append(compile_stderr or compile_stdout)
        else:
            evidence_parts.append(
                f"candidate compile returncode={compile_info.get('returncode', '')}; captured compiler stderr/stdout is empty"
            )
    else:
        tests_failed = _failed_count(tests_info)
        diff_mismatch = _failed_count(diff_info, "mismatch_cases")
        fuzz_mismatch = _failed_count(fuzz_info, "mismatch_cases")
        fuzz_crash = _failed_count(fuzz_info, "crash_cases")
        if tests_failed:
            primary_reason = "harness_tests_failed"
            evidence_parts.append(_first_failed_case_note(tests_info))
        elif diff_mismatch:
            primary_reason = "diff_execution_mismatch"
            evidence_parts.append(_first_failed_case_note(diff_info))
        elif fuzz_mismatch:
            primary_reason = "fuzz_execution_mismatch"
            evidence_parts.append(_first_failed_case_note(fuzz_info))
        elif fuzz_crash:
            primary_reason = "fuzz_execution_crash"
            evidence_parts.append(_first_failed_case_note(fuzz_info))
        else:
            primary_reason = "semantic_score_below_threshold"
            evidence_parts.append("compile succeeded but aggregate semantic score is below threshold")

    return {
        "sample_id": str(episode.get("sample_id", "unknown")),
        "category": category,
        "primary_reason": primary_reason,
        "semantic_score": semantic_score,
        "compile_succeeded": compile_succeeded,
        "tests_passed": tests_passed,
        "diff_passed": _section_passed(diff_info),
        "fuzz_passed": _section_passed(fuzz_info),
        "transform_status": status,
        "evidence": _clean_text(" | ".join(part for part in evidence_parts if part), limit=420),
        "details": {
            "compile_returncode": compile_info.get("returncode", ""),
            "tests_failed_cases": _failed_count(tests_info),
            "diff_mismatch_cases": _failed_count(diff_info, "mismatch_cases"),
            "fuzz_mismatch_cases": _failed_count(fuzz_info, "mismatch_cases"),
            "fuzz_crash_cases": _failed_count(fuzz_info, "crash_cases"),
        },
    }


def load_tigress_episodes(summary_path: Path) -> list[dict[str, Any]]:
    payload = _load_dict(summary_path)
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return []
    tigress = results.get("tigress", {})
    if not isinstance(tigress, dict):
        for item in results.values():
            if isinstance(item, dict) and item.get("baseline_name") == "tigress":
                tigress = item
                break
    episodes = tigress.get("episodes", []) if isinstance(tigress, dict) else []
    return [dict(item) for item in episodes if isinstance(item, dict)]


def summarize(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = Counter(str(item["category"]) for item in classifications)
    by_reason = Counter(str(item["primary_reason"]) for item in classifications)
    return {
        "failure_count": len(classifications),
        "by_category": dict(sorted(by_category.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = payload["failures"]
    lines = [
        "# Tigress Failure Breakdown",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Baseline summary: `{payload['baseline_summary_path']}`",
        f"- Semantic threshold: `{payload['semantic_threshold']}`",
        f"- Failure count: `{summary['failure_count']}`",
        f"- By category: `{summary['by_category']}`",
        f"- By reason: `{summary['by_reason']}`",
        "",
        "| sample_id | category | primary_reason | semantic | compile | tests | diff | fuzz | evidence |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['sample_id']}`",
                    f"`{row['category']}`",
                    f"`{row['primary_reason']}`",
                    f"`{row['semantic_score']:.6f}`",
                    f"`{row['compile_succeeded']}`",
                    f"`{row['tests_passed']}`",
                    f"`{row['diff_passed']}`",
                    f"`{row['fuzz_passed']}`",
                    _clean_text(row.get("evidence", ""), limit=160).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paper Interpretation",
            "",
            "Tigress failures are split into tool-level failures and semantic failures. Tool-level failures mean the current Tigress Flatten configuration could not parse, transform, or compile a sample reliably. Semantic failures mean Tigress produced a compilable candidate, but the verifier observed behavior below the semantic threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary_path = resolve_path(args.baseline_summary)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    episodes = load_tigress_episodes(summary_path)
    classifications = [
        item
        for item in (classify_episode(episode, semantic_threshold=float(args.semantic_threshold)) for episode in episodes)
        if item is not None
    ]
    classifications.sort(key=lambda item: (str(item["category"]), str(item["sample_id"])))
    payload = {
        "created_at": utc_timestamp(),
        "baseline_summary_path": project_relative(summary_path),
        "semantic_threshold": float(args.semantic_threshold),
        "summary": summarize(classifications),
        "failures": classifications,
    }
    dump_json(output_dir / "tigress_failure_breakdown.json", payload)
    write_csv(output_dir / "tigress_failure_breakdown.csv", classifications)
    write_text(output_dir / "tigress_failure_breakdown.md", render_markdown(payload))
    print(f"Wrote Tigress failure breakdown JSON: {output_dir / 'tigress_failure_breakdown.json'}")
    print(f"Wrote Tigress failure breakdown CSV: {output_dir / 'tigress_failure_breakdown.csv'}")
    print(f"Wrote Tigress failure breakdown Markdown: {output_dir / 'tigress_failure_breakdown.md'}")


if __name__ == "__main__":
    main()
