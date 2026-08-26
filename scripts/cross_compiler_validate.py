from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text
from scripts.measure_static_obfuscation import (
    _external_candidate_lookup,
    _final_source_from_trace,
    _infer_seed_from_run,
    _infer_variant_from_run,
)
from scripts.run_baselines import format_duration, run_external_compile_verification
from verifier.compile import choose_compiler
from verifier.pipeline import VerificationPipeline


DEFAULT_BASELINE_SUMMARIES = [
    "artifacts/formal_eval_150_harness_clean/fe150_baselines_20260519_201421/reports/baseline_summary.json",
    "artifacts/formal_eval_150_harness_clean/fe150_tigress_20260519_220111/reports/baseline_summary.json",
]

DEFAULT_PPO_RUN_DIRS = [
    "artifacts/formal_eval_150_harness_clean/multiseed/mppo_seed1",
    "artifacts/formal_eval_150_harness_clean/multiseed/mppo_seed7",
    "artifacts/formal_eval_150_harness_clean/multiseed/mppo_seed13",
]

DEFAULT_PROFILES = [
    "clang_O0=local:clang:-O0",
    "clang_O2=local:clang:-O2",
    "wsl_gcc_O0=wsl:gcc:-O0",
    "wsl_gcc_O2=wsl:gcc:-O2",
]

ROW_FIELDS = [
    "method",
    "variant",
    "seed",
    "sample_id",
    "profile",
    "backend",
    "compiler",
    "flags",
    "status",
    "semantic_passed",
    "compile_succeeded",
    "tests_passed",
    "diff_passed",
    "fuzz_passed",
    "semantic_score",
    "verifier_coverage",
    "failure_category",
    "original_source",
    "candidate_source",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate obfuscated candidates across compiler backends and optimization levels."
    )
    parser.add_argument("--baseline-summary", action="append", default=[], help="Path to run_baselines summary JSON.")
    parser.add_argument("--ppo-run-dir", action="append", default=[], help="Path to a Maskable PPO run directory.")
    parser.add_argument("--output-dir", default="artifacts/formal_eval_150_harness_clean/cross_compiler_validation")
    parser.add_argument("--job-name", default="formal_eval_150_cross_compiler_validation")
    parser.add_argument(
        "--compiler-profile",
        action="append",
        default=[],
        help="Profile spec: name=local:compiler:flags or name=wsl:compiler:flags, e.g. gcc_O2=wsl:gcc:-O2.",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.97)
    parser.add_argument("--compile-timeout-sec", type=float, default=15.0)
    parser.add_argument("--fuzz-max-cases", type=int, default=8)
    parser.add_argument("--no-fuzz", action="store_true")
    parser.add_argument("--max-rows-per-group", type=int, default=0, help="Optional smoke limit per method/variant/seed.")
    parser.add_argument("--include-unavailable", action="store_true", help="Record unavailable compilers instead of skipping.")
    return parser.parse_args(argv)


def parse_profile(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError(f"Compiler profile must contain '=': {spec}")
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    if len(parts) < 2:
        raise ValueError(f"Compiler profile must be name=backend:compiler[:flags]: {spec}")
    backend = parts[0].strip()
    compiler = parts[1].strip()
    flags_text = ":".join(parts[2:]).strip() if len(parts) > 2 else ""
    flags = [part for part in re.split(r"\s+", flags_text) if part]
    if backend not in {"local", "wsl"}:
        raise ValueError(f"Unsupported compiler profile backend {backend!r}; expected local or wsl.")
    if not name.strip() or not compiler:
        raise ValueError(f"Invalid compiler profile: {spec}")
    return {"name": name.strip(), "backend": backend, "compiler": compiler, "flags": flags}


def compiler_available(profile: dict[str, Any]) -> bool:
    compiler = str(profile["compiler"])
    if profile["backend"] == "local":
        return choose_compiler(compiler) is not None
    if shutil.which("wsl") is None:
        return False
    for _ in range(3):
        try:
            completed = subprocess.run(
                ["wsl", "bash", "-lc", f"command -v {compiler}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            return True
        time.sleep(1.0)
    return False


def _load_dict(path: Path) -> dict[str, Any]:
    payload = load_structured_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _sample_lookup_from_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _load_dict(path)
    lookup: dict[str, dict[str, Any]] = {}
    for raw in payload.get("samples", []):
        if isinstance(raw, dict) and raw.get("sample_id"):
            lookup[str(raw["sample_id"])] = dict(raw)
    return lookup


def _episode_rows(
    *,
    method: str,
    variant: str,
    seed: str,
    episodes: list[dict[str, Any]],
    sample_lookup: dict[str, dict[str, Any]],
    candidate_lookup: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        sample_id = str(episode.get("sample_id", ""))
        if not sample_id or sample_id not in sample_lookup:
            continue
        sample = sample_lookup[sample_id]
        original_path = resolve_path(str(sample.get("source_path", "")))
        trace_path = str(episode.get("trace_path", ""))
        candidate_path = _final_source_from_trace(resolve_path(trace_path)) if trace_path else None
        if candidate_path is None and candidate_lookup:
            candidate_path = candidate_lookup.get(sample_id)
        if candidate_path is None or not original_path.exists() or not candidate_path.exists():
            continue
        verification_input = dict(sample.get("verification_input", {}))
        rows.append(
            {
                "method": method,
                "variant": variant,
                "seed": seed,
                "sample_id": sample_id,
                "original_source": project_relative(original_path),
                "candidate_source": project_relative(candidate_path),
                "test_cases": list(verification_input.get("test_cases", [])),
            }
        )
    return rows


def collect_candidate_rows(
    *,
    baseline_summaries: list[str],
    ppo_run_dirs: list[str],
    max_rows_per_group: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_text in baseline_summaries:
        summary_path = resolve_path(summary_text)
        summary = _load_dict(summary_path)
        run_dir = resolve_path(str(summary.get("run_dir", summary_path.parent.parent)))
        sample_lookup = _sample_lookup_from_cache(run_dir / "data" / "baseline_env_input_cache.json")
        for method, result in dict(summary.get("results", {})).items():
            episodes = [dict(item) for item in dict(result).get("episodes", []) if isinstance(item, dict)]
            group_rows = _episode_rows(
                method=str(method),
                variant="baseline",
                seed="",
                episodes=episodes,
                sample_lookup=sample_lookup,
                candidate_lookup=_external_candidate_lookup(run_dir, str(method), episodes),
            )
            rows.extend(group_rows[:max_rows_per_group] if max_rows_per_group else group_rows)

    for run_text in ppo_run_dirs:
        run_dir = resolve_path(run_text)
        sample_lookup = _sample_lookup_from_cache(run_dir / "data" / "env_input_cache.json")
        log_path = run_dir / "logs" / "episode_metrics.corrected.jsonl"
        if not log_path.exists():
            log_path = run_dir / "logs" / "episode_metrics.jsonl"
        if not log_path.exists():
            log_path = run_dir / "reports" / "episodes.jsonl"
        episodes: list[dict[str, Any]] = []
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    episodes.append(payload)
        group_rows = _episode_rows(
            method="maskable_ppo",
            variant=_infer_variant_from_run(run_dir),
            seed=_infer_seed_from_run(run_dir),
            episodes=episodes,
            sample_lookup=sample_lookup,
        )
        rows.extend(group_rows[:max_rows_per_group] if max_rows_per_group else group_rows)
    return rows


def _summary_to_payload(summary: Any) -> dict[str, Any]:
    return summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)


def _failure_category(payload: dict[str, Any], semantic_threshold: float) -> str:
    compile_succeeded = bool(dict(payload.get("compile", {})).get("succeeded", False))
    if not compile_succeeded:
        return "compile_failure"
    tests = dict(payload.get("tests", {}))
    if tests and not bool(tests.get("passed", False)):
        return "test_failure"
    diff = dict(payload.get("diff", {}))
    if bool(diff.get("executed", False)) and not bool(diff.get("passed", False)):
        return "diff_failure"
    fuzz = dict(payload.get("fuzz", {}))
    if bool(fuzz.get("executed", False)) and not bool(fuzz.get("passed", False)):
        return "fuzz_failure"
    if float(payload.get("semantic_score", 0.0)) < semantic_threshold:
        return "low_semantic_score"
    return ""


def _wsl_command_template(profile: dict[str, Any]) -> list[str]:
    flags = " ".join(str(flag) for flag in profile["flags"])
    command = f"{profile['compiler']} {{verify_source_posix}} -o {{verify_executable_posix}}"
    if flags:
        command += f" {flags}"
    return ["wsl", "bash", "-lc", command]


def validate_one(
    row: dict[str, Any],
    profile: dict[str, Any],
    *,
    output_dir: Path,
    semantic_threshold: float,
    compile_timeout_sec: float,
    fuzz_max_cases: int,
    fuzz_enabled: bool,
) -> dict[str, Any]:
    sample_slug = str(row["sample_id"]).replace("/", "_")
    profile_name = str(profile["name"])
    build_dir = ensure_dir(output_dir / "_build" / profile_name / sample_slug)
    original_path = resolve_path(str(row["original_source"]))
    candidate_path = resolve_path(str(row["candidate_source"]))
    test_cases = list(row.get("test_cases", []))

    if profile["backend"] == "local":
        pipeline = VerificationPipeline(
            semantic_threshold=semantic_threshold,
            compiler=str(profile["compiler"]),
            compiler_flags=list(profile["flags"]),
            compile_timeout_sec=compile_timeout_sec,
            build_root=output_dir / "_verifier",
            cache_root=output_dir / "_cache" / profile_name,
        )
        payload = _summary_to_payload(
            pipeline.verify_source(
                source_path=candidate_path,
                test_cases=test_cases,
                sample_id=f"{profile_name}_{sample_slug}",
                compiler=str(profile["compiler"]),
                compiler_flags=list(profile["flags"]),
                compile_timeout_sec=compile_timeout_sec,
                reference_source_path=original_path,
                build_dir=build_dir,
                fuzz_max_cases=fuzz_max_cases,
                fuzz_enabled=fuzz_enabled,
            )
        )
    else:
        payload = run_external_compile_verification(
            command_template=_wsl_command_template(profile),
            tool_path="wsl",
            tigress_home=resolve_path("."),
            source_path=original_path,
            output_path=candidate_path,
            sample_root=build_dir,
            sample_id=str(row["sample_id"]),
            semantic_threshold=semantic_threshold,
            test_cases=test_cases,
            fuzz_enabled=fuzz_enabled,
            fuzz_max_cases=fuzz_max_cases,
            timeout_sec=compile_timeout_sec,
        )

    compile_succeeded = bool(dict(payload.get("compile", {})).get("succeeded", False))
    tests_payload = dict(payload.get("tests", {}))
    diff_payload = dict(payload.get("diff", {}))
    fuzz_payload = dict(payload.get("fuzz", {}))
    semantic_score = float(payload.get("semantic_score", 0.0))
    semantic_passed = compile_succeeded and semantic_score >= semantic_threshold
    failure_category = _failure_category(payload, semantic_threshold)
    return {
        "method": row["method"],
        "variant": row["variant"],
        "seed": row["seed"],
        "sample_id": row["sample_id"],
        "profile": profile_name,
        "backend": profile["backend"],
        "compiler": profile["compiler"],
        "flags": " ".join(profile["flags"]),
        "status": "passed" if semantic_passed and not failure_category else "failed",
        "semantic_passed": semantic_passed,
        "compile_succeeded": compile_succeeded,
        "tests_passed": bool(tests_payload.get("passed", False)),
        "diff_passed": bool(diff_payload.get("passed", False)),
        "fuzz_passed": bool(fuzz_payload.get("passed", False)),
        "semantic_score": round(semantic_score, 6),
        "verifier_coverage": round(float(payload.get("verifier_coverage", 0.0)), 6),
        "failure_category": failure_category,
        "original_source": row["original_source"],
        "candidate_source": row["candidate_source"],
        "verification_summary": payload,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["variant"]), str(row["seed"]), str(row["profile"]))].append(row)
    aggregates: list[dict[str, Any]] = []
    for (method, variant, seed, profile), items in sorted(grouped.items()):
        total = len(items)
        compile_pass = sum(1 for item in items if bool(item.get("compile_succeeded", False)))
        semantic_pass = sum(1 for item in items if bool(item.get("semantic_passed", False)))
        tests_pass = sum(1 for item in items if bool(item.get("tests_passed", False)))
        diff_pass = sum(1 for item in items if bool(item.get("diff_passed", False)))
        fuzz_pass = sum(1 for item in items if bool(item.get("fuzz_passed", False)))
        scores = [float(item.get("semantic_score", 0.0)) for item in items]
        failures: dict[str, int] = defaultdict(int)
        for item in items:
            category = str(item.get("failure_category", ""))
            if category:
                failures[category] += 1
        aggregates.append(
            {
                "method": method,
                "variant": variant,
                "seed": seed,
                "profile": profile,
                "rows": total,
                "compile_pass_rate": round(compile_pass / total, 6) if total else 0.0,
                "semantic_pass_rate": round(semantic_pass / total, 6) if total else 0.0,
                "test_pass_rate": round(tests_pass / total, 6) if total else 0.0,
                "diff_pass_rate": round(diff_pass / total, 6) if total else 0.0,
                "fuzz_pass_rate": round(fuzz_pass / total, 6) if total else 0.0,
                "mean_semantic_score": round(mean(scores), 6) if scores else 0.0,
                "failure_count": total - semantic_pass,
                "failure_breakdown": dict(sorted(failures.items())),
            }
        )
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ROW_FIELDS})


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-Compiler Validation",
        "",
        f"- job: `{payload['job_name']}`",
        f"- generated_at: `{payload['generated_at']}`",
        f"- candidate_rows: `{payload['candidate_row_count']}`",
        f"- validation_rows: `{payload['validation_row_count']}`",
        f"- profiles: `{', '.join(profile['name'] for profile in payload['profiles'])}`",
        "",
        "## Aggregates",
        "",
        "| method | variant | seed | profile | rows | compile pass | semantic pass | tests pass | diff pass | fuzz pass | mean semantic | failures | breakdown |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["aggregates"]:
        lines.append(
            "| {method} | {variant} | {seed} | {profile} | {rows} | {compile_pass_rate:.6f} | "
            "{semantic_pass_rate:.6f} | {test_pass_rate:.6f} | {diff_pass_rate:.6f} | "
            "{fuzz_pass_rate:.6f} | {mean_semantic_score:.6f} | {failure_count} | `{failure_breakdown}` |".format(
                **row
            )
        )
    if payload.get("unavailable_profiles"):
        lines.extend(["", "## Unavailable Profiles", ""])
        for profile in payload["unavailable_profiles"]:
            lines.append(f"- `{profile['name']}` ({profile['backend']}:{profile['compiler']})")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `local` profiles use the native verifier pipeline on the current machine.",
            "- `wsl` profiles compile and execute inside WSL, reusing the same harness/diff/fuzz semantics as the Tigress baseline path.",
            "- A semantic pass requires compilation success and semantic score greater than or equal to the configured threshold.",
        ]
    )
    return "\n".join(lines) + "\n"


def print_progress(index: int, total: int, row: dict[str, Any], profile: dict[str, Any], started_at: float) -> None:
    elapsed = time.monotonic() - started_at
    eta = (elapsed / max(index, 1)) * max(total - index, 0) if total else 0.0
    print(
        f"[cross-compiler] {index}/{total} profile={profile['name']} "
        f"method={row['method']} sample={row['sample_id']} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    baseline_summaries = list(args.baseline_summary) or DEFAULT_BASELINE_SUMMARIES
    ppo_run_dirs = list(args.ppo_run_dir) or DEFAULT_PPO_RUN_DIRS
    profiles = [parse_profile(spec) for spec in (list(args.compiler_profile) or DEFAULT_PROFILES)]
    available_profiles: list[dict[str, Any]] = []
    unavailable_profiles: list[dict[str, Any]] = []
    for profile in profiles:
        if compiler_available(profile):
            available_profiles.append(profile)
        else:
            unavailable_profiles.append(profile)

    rows = collect_candidate_rows(
        baseline_summaries=baseline_summaries,
        ppo_run_dirs=ppo_run_dirs,
        max_rows_per_group=int(args.max_rows_per_group),
    )
    validation_rows: list[dict[str, Any]] = []
    started_at = time.monotonic()
    total = len(rows) * len(available_profiles)
    index = 0
    for row in rows:
        for profile in available_profiles:
            index += 1
            print_progress(index, total, row, profile, started_at)
            validation_rows.append(
                validate_one(
                    row,
                    profile,
                    output_dir=output_dir,
                    semantic_threshold=float(args.semantic_threshold),
                    compile_timeout_sec=float(args.compile_timeout_sec),
                    fuzz_max_cases=int(args.fuzz_max_cases),
                    fuzz_enabled=not bool(args.no_fuzz),
                )
            )

    if args.include_unavailable:
        for row in rows:
            for profile in unavailable_profiles:
                validation_rows.append(
                    {
                        "method": row["method"],
                        "variant": row["variant"],
                        "seed": row["seed"],
                        "sample_id": row["sample_id"],
                        "profile": profile["name"],
                        "backend": profile["backend"],
                        "compiler": profile["compiler"],
                        "flags": " ".join(profile["flags"]),
                        "status": "unavailable",
                        "semantic_passed": False,
                        "compile_succeeded": False,
                        "tests_passed": False,
                        "diff_passed": False,
                        "fuzz_passed": False,
                        "semantic_score": 0.0,
                        "verifier_coverage": 0.0,
                        "failure_category": "compiler_unavailable",
                        "original_source": row["original_source"],
                        "candidate_source": row["candidate_source"],
                    }
                )

    payload = {
        "job_name": str(args.job_name),
        "generated_at": utc_timestamp(),
        "candidate_row_count": len(rows),
        "validation_row_count": len(validation_rows),
        "semantic_threshold": float(args.semantic_threshold),
        "profiles": available_profiles,
        "unavailable_profiles": unavailable_profiles,
        "baseline_summaries": [project_relative(resolve_path(path)) for path in baseline_summaries],
        "ppo_run_dirs": [project_relative(resolve_path(path)) for path in ppo_run_dirs],
        "rows": validation_rows,
        "aggregates": aggregate(validation_rows),
    }
    dump_json(output_dir / "cross_compiler_validation.json", payload)
    _write_csv(output_dir / "cross_compiler_validation_rows.csv", validation_rows)
    write_text(output_dir / "cross_compiler_validation.md", _markdown(payload))
    print(f"Wrote cross-compiler validation report: {output_dir / 'cross_compiler_validation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
