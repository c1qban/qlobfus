from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from verifier.compile import compile_c_source
from verifier.pipeline import VerificationPipeline

from .common import (
    PROJECT_ROOT,
    dump_json,
    ensure_dir,
    load_config,
    load_structured_file,
    project_relative,
    resolve_path,
    slugify,
    utc_timestamp,
    write_text,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset manifests and processed-data scaffolding.")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML config file.")
    parser.add_argument("--dry-run", action="store_true", help="Preview outputs without writing files.")
    parser.add_argument("--progress", action="store_true", help="Print a progress bar with elapsed time and ETA.")
    return parser.parse_args(argv)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m{second:02d}s"
    if minutes:
        return f"{minutes}m{second:02d}s"
    return f"{second}s"


def print_progress(
    *,
    current: int,
    total: int,
    included_count: int,
    filtered_count: int,
    started_at: float,
    label: str = "prepare",
) -> None:
    if total <= 0:
        return
    elapsed = time.monotonic() - started_at
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else 0.0
    bar_width = 28
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    message = (
        f"\r[{label}] [{bar}] {current}/{total} "
        f"kept={included_count} filtered={filtered_count} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)}"
    )
    sys.stderr.write(message)
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def discover_samples(source_root: Path, patterns: list[str]) -> list[Path]:
    if not source_root.exists():
        return []

    discovered: set[Path] = set()
    for pattern in patterns:
        for path in source_root.rglob(pattern):
            if path.is_file():
                discovered.add(path.resolve())
    return sorted(discovered)


def _infer_codenet_problem_id(relative_source: Path, dataset: dict[str, Any]) -> str:
    parts = list(relative_source.parts)
    language_dirs = {str(item) for item in dataset.get("language_dirs", ["C", "c"])}
    language_index = next((index for index, item in enumerate(parts) if item in language_dirs), -1)
    if language_index > 0:
        return parts[language_index - 1]
    if parts:
        return parts[0]
    return relative_source.stem


def _build_requested_metadata_keys(
    dataset: dict[str, Any],
    source_root: Path | None,
    discovered_sources: list[Path] | None,
) -> tuple[set[str], set[str]]:
    if not source_root or not discovered_sources:
        return set(), set()

    metadata_lookup = str(dataset.get("metadata_lookup", "submission_id"))
    requested_keys: set[str] = set()
    relevant_problem_ids: set[str] = set()
    for source_path in discovered_sources:
        relative_source = source_path.relative_to(source_root)
        relevant_problem_ids.add(_infer_codenet_problem_id(relative_source, dataset))
        if metadata_lookup == "submission_id":
            requested_keys.add(relative_source.stem)
        else:
            requested_keys.add(relative_source.as_posix())

    return requested_keys, relevant_problem_ids


def load_dataset_metadata(
    dataset: dict[str, Any],
    *,
    source_root: Path | None = None,
    discovered_sources: list[Path] | None = None,
) -> dict[str, dict[str, str]]:
    metadata_path_value = dataset.get("metadata_csv") or dataset.get("metadata_root")
    if not metadata_path_value:
        return {}
    metadata_path = resolve_path(metadata_path_value, PROJECT_ROOT)
    if not metadata_path.exists():
        return {}

    key_field = str(dataset.get("metadata_key_field", "submission_id"))
    metadata_rows: dict[str, dict[str, str]] = {}
    requested_keys, relevant_problem_ids = _build_requested_metadata_keys(dataset, source_root, discovered_sources)

    metadata_files: list[Path]
    if metadata_path.is_dir():
        metadata_glob = str(dataset.get("metadata_glob", "p*.csv"))
        if relevant_problem_ids:
            metadata_files = [
                metadata_path / f"{problem_id}.csv"
                for problem_id in sorted(relevant_problem_ids)
                if (metadata_path / f"{problem_id}.csv").exists()
            ]
        else:
            metadata_files = [
                path
                for path in sorted(metadata_path.glob(metadata_glob))
                if path.name.lower() != "problem_list.csv"
            ]
    else:
        metadata_files = [metadata_path]

    for csv_path in metadata_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = {str(key): str(value) for key, value in row.items() if key is not None}
                key_value = normalized.get(key_field, "").strip()
                if not key_value:
                    continue
                if requested_keys and key_value not in requested_keys:
                    continue
                metadata_rows[key_value] = normalized
    return metadata_rows


def count_source_lines(source_path: Path) -> int:
    return len(source_path.read_text(encoding="utf-8").splitlines())


def load_harness_cases(harness_path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = load_structured_file(harness_path)
    if isinstance(payload, dict):
        harness_format = str(payload.get("format", "executable_stdio_v1"))
        raw_cases = payload.get("test_cases", [])
    elif isinstance(payload, list):
        harness_format = "executable_stdio_v1"
        raw_cases = payload
    else:
        raise ValueError(f"Harness file must decode to an object or list: {harness_path}")

    if not isinstance(raw_cases, list):
        raise ValueError(f"Harness test_cases must be a list: {harness_path}")

    return harness_format, [dict(case) for case in raw_cases]


def build_harness_entry(
    source_path: Path,
    source_root: Path,
    harness_config: dict[str, Any],
) -> dict[str, Any]:
    harness_root_value = harness_config.get("harness_root")
    if not harness_root_value:
        return {
            "format": str(harness_config.get("format", "executable_stdio_v1")),
            "path": "",
            "exists": False,
            "case_count": 0,
            "test_cases": [],
            "notes": ["No harness_root configured for this dataset."],
        }

    harness_root = resolve_path(harness_root_value, PROJECT_ROOT)
    harness_suffix = str(harness_config.get("filename_suffix", ".tests.json"))
    inline_cases = bool(harness_config.get("inline_test_cases_in_manifest", True))
    relative_source = source_path.relative_to(source_root)
    harness_path = harness_root / relative_source.with_suffix(harness_suffix)

    if not harness_path.exists():
        return {
            "format": str(harness_config.get("format", "executable_stdio_v1")),
            "path": project_relative(harness_path),
            "exists": False,
            "case_count": 0,
            "test_cases": [],
            "notes": ["Harness file was not found for this sample."],
        }

    harness_format, test_cases = load_harness_cases(harness_path)
    return {
        "format": harness_format,
        "path": project_relative(harness_path),
        "exists": True,
        "case_count": len(test_cases),
        "test_cases": test_cases if inline_cases else [],
        "notes": [] if inline_cases else ["Harness cases are stored in the external harness file path."],
    }


def build_sample_entry(
    source_path: Path,
    source_root: Path,
    dataset: dict[str, Any],
    compile_config: dict[str, Any],
    harness_config: dict[str, Any],
    metadata_rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    relative_source = source_path.relative_to(source_root)
    source_stem = relative_source.with_suffix("").as_posix()
    layout = str(dataset.get("layout", "generic_recursive"))
    sample_id = source_stem
    split_group_candidate = ""
    metadata: dict[str, Any] = {}
    if layout == "project_codenet":
        inferred = infer_project_codenet_fields(relative_source, dataset, metadata_rows)
        sample_id = str(inferred["sample_id"])
        split_group_candidate = str(inferred["split_group_candidate"])
        metadata = dict(inferred["metadata"])
    harness_entry = build_harness_entry(source_path, source_root, harness_config)

    return {
        "sample_id": sample_id,
        "source_path": project_relative(source_path),
        "relative_source_path": relative_source.as_posix(),
        "dataset_layout": layout,
        "metadata": metadata,
        "split_group_candidate": split_group_candidate,
        "source_stats": {
            "line_count": count_source_lines(source_path),
        },
        "compile": {
            "compiler": compile_config.get("compiler"),
            "compiler_flags": list(compile_config.get("compiler_flags", [])),
            "timeout_sec": float(compile_config.get("timeout_sec", 10.0)),
        },
        "harness": harness_entry,
    }


def infer_project_codenet_fields(
    relative_source: Path,
    dataset: dict[str, Any],
    metadata_rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    parts = list(relative_source.parts)
    language_dirs = {str(item) for item in dataset.get("language_dirs", ["C", "c"])}
    language_index = next((index for index, item in enumerate(parts) if item in language_dirs), -1)
    problem_id = parts[0] if parts else relative_source.stem
    language_dir = ""
    if language_index > 0:
        problem_id = parts[language_index - 1]
        language_dir = parts[language_index]
    elif len(parts) >= 2:
        problem_id = parts[0]
        language_dir = parts[1]

    submission_id = relative_source.stem
    metadata_lookup = str(dataset.get("metadata_lookup", "submission_id"))
    metadata_key = submission_id if metadata_lookup == "submission_id" else relative_source.as_posix()
    metadata_row = dict(metadata_rows.get(metadata_key, {}))
    sample_id_template = str(dataset.get("sample_id_template", "{problem_id}/{submission_id}"))
    sample_id = sample_id_template.format(
        problem_id=problem_id,
        submission_id=submission_id,
        language_dir=language_dir,
        relative_source_path=relative_source.as_posix(),
    )
    return {
        "sample_id": sample_id,
        "split_group_candidate": problem_id,
        "metadata": {
            "problem_id": problem_id,
            "submission_id": submission_id,
            "language_dir": language_dir,
            "metadata_key": metadata_key,
            "metadata_available": bool(metadata_row),
            "metadata_row": metadata_row,
        },
    }


def summarize_compile_result(result: Any) -> dict[str, Any]:
    return {
        "succeeded": bool(result.succeeded),
        "command": result.command,
        "returncode": int(result.returncode),
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
        "notes": list(result.notes),
    }


def run_compile_check(
    sample: dict[str, Any],
    compile_check_root: Path,
) -> dict[str, Any]:
    source_path = resolve_path(str(sample["source_path"]))
    compile_config = sample["compile"]
    build_dir = ensure_dir(compile_check_root / slugify(str(sample["sample_id"])))
    executable_path = build_dir / (source_path.stem + (".exe" if os.name == "nt" else ""))

    try:
        result = compile_c_source(
            source_path=source_path,
            output_path=executable_path,
            compiler=compile_config.get("compiler"),
            compiler_flags=list(compile_config.get("compiler_flags", [])),
            timeout_sec=float(compile_config.get("timeout_sec", 10.0)),
            workdir=source_path.parent,
        )
        return summarize_compile_result(result)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def run_harness_check(
    sample: dict[str, Any],
    harness_check_root: Path,
    semantic_threshold: float,
) -> dict[str, Any]:
    source_path = resolve_path(str(sample["source_path"]))
    compile_config = sample["compile"]
    build_dir = ensure_dir(harness_check_root / slugify(str(sample["sample_id"])))
    try:
        summary = VerificationPipeline(
            semantic_threshold=semantic_threshold,
            compiler=compile_config.get("compiler"),
            compiler_flags=list(compile_config.get("compiler_flags", [])),
            compile_timeout_sec=float(compile_config.get("timeout_sec", 10.0)),
            build_root=build_dir,
            cache_root=harness_check_root / "_cache",
        ).verify_source(
            source_path=source_path,
            test_cases=list(sample.get("harness", {}).get("test_cases", [])),
            sample_id=f"{sample['sample_id']}_prepare_harness",
            compiler=compile_config.get("compiler"),
            compiler_flags=list(compile_config.get("compiler_flags", [])),
            compile_timeout_sec=float(compile_config.get("timeout_sec", 10.0)),
            workdir=source_path.parent,
            fuzz_enabled=False,
        )
        return {
            "checked": True,
            "passed": float(summary.semantic_score) >= semantic_threshold,
            "semantic_score": float(summary.semantic_score),
            "verifier_coverage": float(summary.verifier_coverage),
            "compile_succeeded": bool(summary.compile.succeeded),
            "tests_passed": bool(summary.tests.passed),
            "test_failed_cases": int(summary.tests.failed_cases),
            "notes": list(summary.notes),
        }
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def evaluate_filter_reasons(
    sample: dict[str, Any],
    filters: dict[str, Any],
    compile_check_root: Path,
) -> list[str]:
    reasons: list[str] = []
    excluded_sample_ids = {str(value) for value in filters.get("excluded_sample_ids", [])}
    if str(sample.get("sample_id", "")) in excluded_sample_ids:
        reasons.append("excluded_sample_id")

    line_count = int(sample["source_stats"]["line_count"])
    min_lines = filters.get("min_lines")
    max_lines = filters.get("max_lines")

    if min_lines is not None and line_count < int(min_lines):
        reasons.append("below_min_lines")
    if max_lines is not None and line_count > int(max_lines):
        reasons.append("above_max_lines")

    if bool(filters.get("accepted_only", False)):
        metadata = dict(sample.get("metadata", {}))
        metadata_row = dict(metadata.get("metadata_row", {}))
        if not metadata_row:
            reasons.append("missing_metadata")
        else:
            status_field = str(filters.get("metadata_status_field", "status"))
            accepted_values = {
                str(value).strip().lower()
                for value in filters.get("accepted_values", ["accepted", "ac"])
            }
            verdict = str(metadata_row.get(status_field, "")).strip().lower()
            if verdict not in accepted_values:
                reasons.append("not_accepted")

    compile_check = {
        "checked": False,
        "succeeded": None,
        "command": "",
        "returncode": None,
        "stderr_tail": "",
        "notes": [],
    }
    harness_check = {
        "checked": False,
        "passed": None,
        "semantic_score": None,
        "verifier_coverage": None,
        "compile_succeeded": None,
        "tests_passed": None,
        "test_failed_cases": None,
        "notes": [],
    }
    if not reasons and bool(filters.get("require_compilable", False)):
        compile_check = {"checked": True, **run_compile_check(sample, compile_check_root)}
        if not bool(compile_check["succeeded"]):
            reasons.append("not_compilable")
    if (
        not reasons
        and bool(filters.get("require_harness_pass", False))
        and bool(sample.get("harness", {}).get("exists", False))
    ):
        semantic_threshold = float(filters.get("harness_semantic_threshold", 0.97))
        harness_check = run_harness_check(sample, compile_check_root / "_harness_checks", semantic_threshold)
        if not bool(harness_check["passed"]):
            reasons.append("harness_verification_failed")

    sample["prepare_checks"] = {
        "compilable": compile_check["succeeded"],
        "compile_check": compile_check,
        "harness_check": harness_check,
    }
    sample["filter_reasons"] = reasons
    return reasons


def hash_to_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def normalize_split_ratios(split: dict[str, Any]) -> tuple[float, float, float]:
    train_ratio = float(split.get("train_ratio", 0.8))
    validation_ratio = float(split.get("validation_ratio", 0.1))
    test_ratio = float(split.get("test_ratio", 0.1))
    total = train_ratio + validation_ratio + test_ratio
    if total <= 0:
        return 1.0, 0.0, 0.0
    return train_ratio / total, validation_ratio / total, test_ratio / total


def get_split_group_key(sample: dict[str, Any], strategy: str) -> str:
    explicit_group = str(sample.get("split_group_candidate", "")).strip()
    if explicit_group:
        return explicit_group
    if strategy == "project_holdout":
        relative_path = str(sample["relative_source_path"])
        parts = relative_path.split("/")
        return parts[0] if len(parts) > 1 else str(sample["sample_id"])
    return str(sample["sample_id"])


def assign_split_labels(samples: list[dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    strategy = str(split.get("strategy", "hash_random"))
    seed = str(split.get("seed", "default"))
    split_counts = {"train": 0, "validation": 0, "test": 0}

    if not samples:
        return {
            "strategy": strategy,
            "seed": seed,
            "counts": split_counts,
            "group_count": 0,
        }

    if strategy == "single_bucket":
        bucket = str(split.get("bucket", "train"))
        for sample in samples:
            sample["split"] = bucket
            sample["split_group"] = str(sample["sample_id"])
            split_counts[bucket] = split_counts.get(bucket, 0) + 1
        return {
            "strategy": strategy,
            "seed": seed,
            "counts": split_counts,
            "group_count": len(samples),
        }

    train_ratio, validation_ratio, _ = normalize_split_ratios(split)
    threshold_train = train_ratio
    threshold_validation = train_ratio + validation_ratio

    group_assignments: dict[str, str] = {}
    for sample in samples:
        group_key = get_split_group_key(sample, strategy)
        if group_key not in group_assignments:
            position = hash_to_unit_interval(f"{seed}:{group_key}")
            if position < threshold_train:
                group_assignments[group_key] = "train"
            elif position < threshold_validation:
                group_assignments[group_key] = "validation"
            else:
                group_assignments[group_key] = "test"

        assigned_split = group_assignments[group_key]
        sample["split"] = assigned_split
        sample["split_group"] = group_key
        split_counts[assigned_split] = split_counts.get(assigned_split, 0) + 1

    return {
        "strategy": strategy,
        "seed": seed,
        "counts": split_counts,
        "group_count": len(group_assignments),
    }


def build_filtered_sample_entry(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "source_path": sample["source_path"],
        "relative_source_path": sample["relative_source_path"],
        "dataset_layout": sample.get("dataset_layout", "generic_recursive"),
        "metadata": sample.get("metadata", {}),
        "source_stats": sample["source_stats"],
        "filter_reasons": list(sample.get("filter_reasons", [])),
        "prepare_checks": sample.get("prepare_checks", {}),
    }


def build_split_index_entry(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "source_path": sample["source_path"],
        "relative_source_path": sample["relative_source_path"],
        "dataset_layout": sample.get("dataset_layout", "generic_recursive"),
        "metadata": sample.get("metadata", {}),
        "split": sample["split"],
        "split_group": sample["split_group"],
        "source_stats": sample["source_stats"],
        "compile": sample["compile"],
        "harness": {
            "format": sample["harness"]["format"],
            "path": sample["harness"]["path"],
            "exists": sample["harness"]["exists"],
            "case_count": sample["harness"]["case_count"],
        },
    }


def write_split_index_files(
    *,
    job_name: str,
    processed_root: Path,
    samples: list[dict[str, Any]],
    split_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    splits_root = ensure_dir(processed_root / "splits")
    split_names = ["train", "validation", "test", "all"]
    index_info: dict[str, dict[str, Any]] = {}

    for split_name in split_names:
        if split_name == "all":
            split_samples = [build_split_index_entry(sample) for sample in samples]
        else:
            split_samples = [build_split_index_entry(sample) for sample in samples if sample.get("split") == split_name]
        index_path = splits_root / f"{split_name}.index.json"
        payload = {
            "job_name": job_name,
            "split": split_name,
            "sample_count": len(split_samples),
            "split_summary": split_summary,
            "samples": split_samples,
        }
        dump_json(index_path, payload)
        index_info[split_name] = {
            "path": project_relative(index_path),
            "sample_count": len(split_samples),
        }

    return index_info


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path, config = load_config(args.config)

    job_name = str(config.get("job_name", "dataset_prepare"))
    dataset = config.get("dataset", {})
    split = config.get("split", {})
    filters = config.get("filters", {})
    verification = config.get("verification", {})
    compile_config = dict(verification.get("compile", {}))
    harness_config = dict(verification.get("harness", {}))
    preview_limit = int(config.get("preview_limit", 20))
    source_root = resolve_path(dataset.get("source_root", "data/raw"), PROJECT_ROOT)
    processed_root = resolve_path(
        dataset.get("processed_root", f"data/processed/{job_name}"),
        PROJECT_ROOT,
    )
    manifest_path = resolve_path(
        dataset.get("manifest_path", f"data/manifests/{job_name}.manifest.json"),
        PROJECT_ROOT,
    )
    file_patterns = list(dataset.get("file_patterns", ["*.c"]))
    compile_check_root = ensure_dir(processed_root / "_prepare_compile_checks")

    discovered_sources = discover_samples(source_root, file_patterns)
    metadata_rows = load_dataset_metadata(
        dataset,
        source_root=source_root,
        discovered_sources=discovered_sources,
    )
    raw_sample_entries = [
        build_sample_entry(
            source_path=source_path,
            source_root=source_root,
            dataset=dataset,
            compile_config=compile_config,
            harness_config=harness_config,
            metadata_rows=metadata_rows,
        )
        for source_path in discovered_sources
    ]

    included_samples: list[dict[str, Any]] = []
    filtered_out_samples: list[dict[str, Any]] = []
    progress_started_at = time.monotonic()
    for index, sample in enumerate(raw_sample_entries, start=1):
        reasons = evaluate_filter_reasons(sample, filters, compile_check_root)
        if reasons:
            filtered_out_samples.append(build_filtered_sample_entry(sample))
        else:
            included_samples.append(sample)
        if args.progress:
            print_progress(
                current=index,
                total=len(raw_sample_entries),
                included_count=len(included_samples),
                filtered_count=len(filtered_out_samples),
                started_at=progress_started_at,
            )

    split_summary = assign_split_labels(included_samples, split)
    harnessed_count = sum(1 for sample in included_samples if sample["harness"]["exists"])

    manifest = {
        "job_name": job_name,
        "created_at": utc_timestamp(),
        "status": "dry_run" if args.dry_run else "prepared",
        "dataset": {
            "name": dataset.get("name", "unspecified"),
            "language": dataset.get("language", "c"),
            "layout": dataset.get("layout", "generic_recursive"),
            "source_root": project_relative(source_root),
            "processed_root": project_relative(processed_root),
            "manifest_path": project_relative(manifest_path),
            "file_patterns": file_patterns,
            "metadata_csv": project_relative(resolve_path(dataset["metadata_csv"], PROJECT_ROOT))
            if dataset.get("metadata_csv")
            else "",
        },
        "split": split,
        "split_summary": split_summary,
        "filters": filters,
        "verification_defaults": {
            "compile": {
                "compiler": compile_config.get("compiler"),
                "compiler_flags": list(compile_config.get("compiler_flags", [])),
                "timeout_sec": float(compile_config.get("timeout_sec", 10.0)),
            },
            "harness": {
                "format": str(harness_config.get("format", "executable_stdio_v1")),
                "harness_root": project_relative(resolve_path(harness_config["harness_root"], PROJECT_ROOT))
                if harness_config.get("harness_root")
                else "",
                "filename_suffix": str(harness_config.get("filename_suffix", ".tests.json")),
                "inline_test_cases_in_manifest": bool(harness_config.get("inline_test_cases_in_manifest", True)),
            },
        },
        "discovery": {
            "source_root_exists": source_root.exists(),
            "metadata_row_count": len(metadata_rows),
            "discovered_sample_count": len(raw_sample_entries),
            "sample_count": len(included_samples),
            "filtered_out_count": len(filtered_out_samples),
            "samples_with_harness": harnessed_count,
            "preview": included_samples[:preview_limit],
            "filtered_preview": filtered_out_samples[:preview_limit],
        },
        "samples": included_samples,
        "filtered_out_samples": filtered_out_samples,
        "next_steps": [
            "Persist project-level split manifests once dataset curation logic expands.",
            "Attach richer semantic checks beyond compilation and executable test cases.",
            "Use split_summary and filtered_out_samples in benchmark reporting.",
        ],
    }

    if args.dry_run:
        shutil.rmtree(compile_check_root, ignore_errors=True)
        print(f"[dry-run] would write manifest: {manifest_path}")
        print(f"[dry-run] discovered samples: {len(raw_sample_entries)}")
        print(f"[dry-run] kept samples: {len(included_samples)}")
        print(f"[dry-run] filtered out samples: {len(filtered_out_samples)}")
        return

    ensure_dir(processed_root)
    split_index_files = write_split_index_files(
        job_name=job_name,
        processed_root=processed_root,
        samples=included_samples,
        split_summary=split_summary,
    )
    manifest["split_index_files"] = split_index_files
    dump_json(manifest_path, manifest)
    write_text(
        processed_root / "README.txt",
        "\n".join(
            [
                f"Job: {job_name}",
                f"Created: {manifest['created_at']}",
                f"Source root: {project_relative(source_root)}",
                f"Discovered samples: {len(raw_sample_entries)}",
                f"Kept samples: {len(included_samples)}",
                f"Filtered out samples: {len(filtered_out_samples)}",
                f"Samples with harness: {harnessed_count}",
                f"Split counts: {split_summary['counts']}",
                f"Split index files: {split_index_files}",
                "Status: dataset manifest now includes filtering, compile checks, and split assignment.",
                "",
            ]
        ),
    )

    shutil.rmtree(compile_check_root, ignore_errors=True)
    print(f"Prepared dataset scaffold: {manifest_path}")


if __name__ == "__main__":
    main()
