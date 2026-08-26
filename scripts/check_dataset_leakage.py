from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.common import dump_json, load_structured_file, project_relative, resolve_path, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dataset split leakage by problem id and source fingerprints.")
    parser.add_argument("--manifest", required=True, help="Prepared manifest JSON.")
    parser.add_argument("--output-json", required=True, help="Output JSON report path.")
    parser.add_argument("--output-md", required=True, help="Output Markdown report path.")
    return parser.parse_args(argv)


def normalize_source_for_near_duplicate(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"\b[_A-Za-z]\w*\b", "ID", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "NUM", text)
    text = re.sub(r"\s+", "", text)
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sample_problem_id(sample: dict[str, Any]) -> str:
    metadata = dict(sample.get("metadata", {}))
    if metadata.get("problem_id"):
        return str(metadata["problem_id"])
    if sample.get("split_group"):
        return str(sample["split_group"])
    return str(sample.get("relative_source_path", "")).split("/")[0]


def build_report(manifest: dict[str, Any]) -> dict[str, Any]:
    samples = [dict(sample) for sample in manifest.get("samples", [])]
    problem_to_splits: dict[str, set[str]] = defaultdict(set)
    exact_hash_to_samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    normalized_hash_to_samples: dict[str, list[dict[str, str]]] = defaultdict(list)

    for sample in samples:
        split = str(sample.get("split", ""))
        problem_id = sample_problem_id(sample)
        source_path = resolve_path(str(sample["source_path"]))
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        exact_hash = sha256_text(source_text)
        normalized_hash = sha256_text(normalize_source_for_near_duplicate(source_text))
        record = {
            "sample_id": str(sample.get("sample_id", "")),
            "split": split,
            "problem_id": problem_id,
            "source_path": project_relative(source_path),
        }
        problem_to_splits[problem_id].add(split)
        exact_hash_to_samples[exact_hash].append(record)
        normalized_hash_to_samples[normalized_hash].append(record)

    problem_leaks = [
        {"problem_id": problem_id, "splits": sorted(splits)}
        for problem_id, splits in sorted(problem_to_splits.items())
        if len(splits) > 1
    ]

    def cross_split_duplicates(groups: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for digest, records in groups.items():
            splits = {record["split"] for record in records}
            if len(records) > 1 and len(splits) > 1:
                findings.append(
                    {
                        "hash": digest,
                        "splits": sorted(splits),
                        "sample_count": len(records),
                        "samples": records,
                    }
                )
        return findings

    exact_duplicates = cross_split_duplicates(exact_hash_to_samples)
    near_duplicates = cross_split_duplicates(normalized_hash_to_samples)
    status = "pass" if not problem_leaks and not exact_duplicates and not near_duplicates else "fail"
    return {
        "status": status,
        "sample_count": len(samples),
        "problem_count": len(problem_to_splits),
        "problem_holdout_leak_count": len(problem_leaks),
        "exact_cross_split_duplicate_count": len(exact_duplicates),
        "near_cross_split_duplicate_count": len(near_duplicates),
        "problem_holdout_leaks": problem_leaks,
        "exact_cross_split_duplicates": exact_duplicates,
        "near_cross_split_duplicates": near_duplicates,
    }


def render_markdown(report: dict[str, Any], manifest_path: Path) -> str:
    lines = [
        "# Dataset Leakage Report",
        "",
        f"- Manifest: `{project_relative(manifest_path)}`",
        f"- Status: `{report['status']}`",
        f"- Samples: `{report['sample_count']}`",
        f"- Problems: `{report['problem_count']}`",
        f"- Problem holdout leaks: `{report['problem_holdout_leak_count']}`",
        f"- Exact cross-split duplicates: `{report['exact_cross_split_duplicate_count']}`",
        f"- Near cross-split duplicates: `{report['near_cross_split_duplicate_count']}`",
        "",
    ]
    if report["status"] == "pass":
        lines.append("No split leakage was detected by the configured checks.")
    else:
        lines.append("Leakage findings require review before using this dataset for final claims.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = resolve_path(args.manifest)
    manifest = load_structured_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must decode to an object: {manifest_path}")
    report = build_report(manifest)
    output_json = resolve_path(args.output_json)
    output_md = resolve_path(args.output_md)
    dump_json(output_json, report)
    write_text(output_md, render_markdown(report, manifest_path))
    print(f"Leakage status: {report['status']}")
    print(f"Wrote JSON: {output_json}")
    print(f"Wrote Markdown: {output_md}")


if __name__ == "__main__":
    main()
