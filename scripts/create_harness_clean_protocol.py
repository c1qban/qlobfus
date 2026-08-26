from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


SPLITS = ("train", "validation", "test")
MIN_DEFAULTS = {"train": 400, "validation": 100, "test": 100}
TOKEN_PATTERN = re.compile(
    r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||<<|>>|->|[{}()[\];,:?+\-*/%<>=!&|^~]"
)
CONTROL_TOKENS = {
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "default",
    "break",
    "continue",
    "return",
    "goto",
    "{",
    "}",
    "(",
    ")",
    ";",
    "?",
    "&&",
    "||",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create independent harness-clean train/validation/test protocol indexes.")
    parser.add_argument("--processed-root", required=True, help="Prepared dataset root containing splits/*.index.json.")
    parser.add_argument("--output-root", required=True, help="Output directory for the frozen protocol.")
    parser.add_argument(
        "--harness-audit",
        action="append",
        default=[],
        help="Optional split=path harness audit. When supplied, only audit status=ok is retained.",
    )
    parser.add_argument("--max-samples-per-problem", type=int, default=4)
    parser.add_argument("--min-train-samples", type=int, default=MIN_DEFAULTS["train"])
    parser.add_argument("--min-validation-samples", type=int, default=MIN_DEFAULTS["validation"])
    parser.add_argument("--min-test-samples", type=int, default=MIN_DEFAULTS["test"])
    parser.add_argument("--job-name", default="harness_clean_protocol_v1")
    return parser.parse_args(argv)


def _as_dict(path: Path) -> dict[str, Any]:
    payload = load_structured_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _problem_id(sample: dict[str, Any]) -> str:
    metadata = dict(sample.get("metadata", {}))
    return str(metadata.get("problem_id") or sample.get("split_group") or str(sample.get("sample_id", "")).split("/", 1)[0])


def _parse_audits(values: list[str]) -> dict[str, dict[str, str]]:
    audits: dict[str, dict[str, str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --harness-audit split=path, got: {value}")
        split, path_text = value.split("=", 1)
        if split not in SPLITS:
            raise ValueError(f"Unknown audit split: {split}")
        payload = _as_dict(resolve_path(path_text))
        audits[split] = {
            str(item["sample_id"]): str(item.get("status", "unknown"))
            for item in payload.get("samples", [])
            if isinstance(item, dict) and item.get("sample_id")
        }
    return audits


def _select_split(
    samples: list[dict[str, Any]],
    *,
    split: str,
    audit: dict[str, str] | None,
    max_samples_per_problem: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for raw_sample in sorted(samples, key=lambda item: (_problem_id(dict(item)), str(dict(item).get("sample_id", "")))):
        sample = dict(raw_sample)
        sample_id = str(sample.get("sample_id", ""))
        problem_id = _problem_id(sample)
        reason = ""
        if str(sample.get("split", "")) != split:
            reason = "split_label_mismatch"
        else:
            harness = dict(sample.get("harness", {}))
            if not harness.get("exists") or int(harness.get("case_count", 0) or 0) <= 0:
                reason = "missing_or_empty_harness"
            elif audit is not None and audit.get(sample_id, "missing_audit") != "ok":
                reason = f"harness_audit_{audit.get(sample_id, 'missing_audit')}"
            elif counts[problem_id] >= max_samples_per_problem:
                reason = "max_samples_per_problem"
        if reason:
            rejected.append({"sample_id": sample_id, "problem_id": problem_id, "reason": reason})
            continue
        selected.append(sample)
        counts[problem_id] += 1
    return selected, rejected


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _fingerprints(source_text: str) -> dict[str, str]:
    without_comments = re.sub(r"/\*.*?\*/|//[^\n]*", "", source_text, flags=re.S)
    tokens = TOKEN_PATTERN.findall(without_comments)
    canonical_tokens = [
        "ID" if re.fullmatch(r"[A-Za-z_]\w*", token) else "NUM" if re.fullmatch(r"\d+(?:\.\d+)?", token) else token
        for token in tokens
    ]
    structural_tokens = [token for token in tokens if token in CONTROL_TOKENS]
    return {
        "source_hash": _sha256(source_text),
        "token_normalized_hash": _sha256(" ".join(canonical_tokens)),
        "ast_shape_proxy_hash": _sha256(" ".join(structural_tokens)),
    }


def build_leakage_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    problem_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    fingerprint_groups: dict[str, dict[str, list[dict[str, str]]]] = {
        "source_hash": defaultdict(list),
        "token_normalized_hash": defaultdict(list),
        "ast_shape_proxy_hash": defaultdict(list),
    }
    for sample in samples:
        source_path = resolve_path(str(sample["source_path"]))
        record = {
            "sample_id": str(sample.get("sample_id", "")),
            "problem_id": _problem_id(sample),
            "split": str(sample.get("split", "")),
            "source_path": project_relative(source_path),
        }
        problem_groups[record["problem_id"]].append(record)
        for name, digest in _fingerprints(source_path.read_text(encoding="utf-8", errors="ignore")).items():
            fingerprint_groups[name][digest].append(record)

    def cross_split(groups: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
        return [
            {"hash": digest, "splits": sorted({item["split"] for item in records}), "samples": records}
            for digest, records in groups.items()
            if len({item["split"] for item in records}) > 1
        ]

    problem_leaks = [
        {"problem_id": problem_id, "splits": sorted({item["split"] for item in records}), "samples": records}
        for problem_id, records in problem_groups.items()
        if len({item["split"] for item in records}) > 1
    ]
    findings = {name: cross_split(groups) for name, groups in fingerprint_groups.items()}
    blocking_findings = problem_leaks or findings["source_hash"] or findings["token_normalized_hash"]
    status = "pass" if not blocking_findings else "fail"
    ast_shape_proxy_status = "warning" if findings["ast_shape_proxy_hash"] else "pass"
    return {
        "status": status,
        "blocking_policy": [
            "problem_holdout",
            "source_hash",
            "token_normalized_hash",
        ],
        "problem_holdout_leak_count": len(problem_leaks),
        "source_hash_leak_count": len(findings["source_hash"]),
        "token_near_duplicate_leak_count": len(findings["token_normalized_hash"]),
        "ast_shape_proxy_leak_count": len(findings["ast_shape_proxy_hash"]),
        "ast_shape_proxy_status": ast_shape_proxy_status,
        "problem_holdout_leaks": problem_leaks,
        "source_hash_leaks": findings["source_hash"],
        "token_near_duplicate_leaks": findings["token_normalized_hash"],
        "ast_shape_proxy_leaks": findings["ast_shape_proxy_hash"],
        "notes": [
            "AST-shape checking is currently a conservative control-flow token proxy, not a full AST graph similarity metric.",
            "AST-shape proxy matches are diagnostic warnings and do not independently fail the protocol.",
        ],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Harness-Clean Protocol Report",
        "",
        f"- job: `{report['job_name']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- status: `{report['status']}`",
        f"- leakage_status: `{report['leakage']['status']}`",
        "",
        "## Split Readiness",
        "",
        "| split | selected | problems | minimum | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for split in SPLITS:
        item = report["splits"][split]
        lines.append(f"| {split} | {item['sample_count']} | {item['problem_count']} | {item['minimum']} | {item['status']} |")
    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            f"- problem holdout leaks: `{report['leakage']['problem_holdout_leak_count']}`",
            f"- source hash leaks: `{report['leakage']['source_hash_leak_count']}`",
            f"- token near-duplicate leaks: `{report['leakage']['token_near_duplicate_leak_count']}`",
            f"- AST-shape proxy warnings: `{report['leakage']['ast_shape_proxy_leak_count']}` "
            f"(`{report['leakage']['ast_shape_proxy_status']}`, non-blocking)",
            "",
            "## Rejection Breakdown",
            "",
        ]
    )
    for split in SPLITS:
        lines.append(f"### {split}")
        breakdown = report["splits"][split]["rejection_breakdown"]
        lines.extend([f"- `{reason}`: {count}" for reason, count in breakdown.items()] or ["- none"])
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    processed_root = resolve_path(args.processed_root)
    output_root = ensure_dir(resolve_path(args.output_root))
    audits = _parse_audits(list(args.harness_audit))
    minimums = {
        "train": int(args.min_train_samples),
        "validation": int(args.min_validation_samples),
        "test": int(args.min_test_samples),
    }
    split_reports: dict[str, Any] = {}
    all_samples: list[dict[str, Any]] = []

    for split in SPLITS:
        source_index = processed_root / "splits" / f"{split}.index.json"
        payload = _as_dict(source_index)
        selected, rejected = _select_split(
            [dict(item) for item in payload.get("samples", [])],
            split=split,
            audit=audits.get(split),
            max_samples_per_problem=int(args.max_samples_per_problem),
        )
        problem_ids = sorted({_problem_id(sample) for sample in selected})
        status = "ready" if len(selected) >= minimums[split] else "below_min_samples"
        index_path = output_root / f"{split}_harness_clean.index.json"
        dump_json(
            index_path,
            {
                "job_name": str(args.job_name),
                "created_at": utc_timestamp(),
                "split": split,
                "status": status,
                "source_index": project_relative(source_index),
                "sample_count": len(selected),
                "problem_count": len(problem_ids),
                "problem_ids": problem_ids,
                "samples": selected,
            },
        )
        all_samples.extend(selected)
        split_reports[split] = {
            "status": status,
            "minimum": minimums[split],
            "sample_count": len(selected),
            "problem_count": len(problem_ids),
            "source_index": project_relative(source_index),
            "output_index": project_relative(index_path),
            "rejection_breakdown": dict(sorted(Counter(item["reason"] for item in rejected).items())),
            "rejected_preview": rejected[:50],
        }

    leakage = build_leakage_report(all_samples)
    status = "ready" if leakage["status"] == "pass" and all(item["status"] == "ready" for item in split_reports.values()) else "not_ready"
    report = {
        "job_name": str(args.job_name),
        "generated_at": utc_timestamp(),
        "status": status,
        "processed_root": project_relative(processed_root),
        "splits": split_reports,
        "leakage": leakage,
    }
    dump_json(output_root / "protocol.report.json", report)
    write_text(output_root / "protocol.report.md", _render_markdown(report))
    print(f"Protocol status: {status}")
    for split in SPLITS:
        print(f"{split}: {split_reports[split]['sample_count']} samples, {split_reports[split]['status']}")
    print(f"Leakage status: {leakage['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
