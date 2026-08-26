from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import shlex
import subprocess
import time
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from eval.llvm_metrics import build_ir_metrics_snapshot, normalize_block_id
from obfuscator.operators.c_source import analyze_source
from rl.envs import ObfuscationEnv
from scripts.common import (
    PROJECT_ROOT,
    copy_file,
    create_timestamped_dir,
    dump_json,
    ensure_dir,
    load_config,
    load_structured_file,
    project_relative,
    resolve_path,
    utc_timestamp,
)
from scripts.train import build_env_cache_sample, load_harness_test_cases, load_split_index
from verifier.models import DiffCaseResult, DiffResult, FuzzCaseResult, FuzzResult, TestCase, TestCaseResult, TestResult
from verifier.pipeline import VerificationPipeline
from verifier.tests import coerce_test_cases
from verifier.tests.runner import normalize_text


INTEGER_PATTERN = re.compile(r"-?\d+")
BINARY_GRID_LINE_PATTERN = re.compile(r"^[01]{8,}$")


def contains_binary_grid_lines(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(1 for line in lines if BINARY_GRID_LINE_PATTERN.fullmatch(line)) >= 2


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def print_progress(label: str, index: int, total: int, sample_id: str, started_at: float) -> None:
    elapsed = time.monotonic() - started_at
    done = max(index, 1)
    eta = (elapsed / done) * max(total - done, 0) if total else 0.0
    print(
        f"[{label}] {index}/{total} sample={sample_id} "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline obfuscation methods on a prepared split index.")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML baseline config.")
    parser.add_argument("--run-dir", help="Optional explicit baseline run directory.")
    return parser.parse_args(argv)


def build_env_cache(split_index: dict[str, Any]) -> dict[str, Any]:
    samples = [dict(sample) for sample in split_index.get("samples", [])]
    return {
        "job_name": str(split_index.get("job_name", "unknown")),
        "created_at": utc_timestamp(),
        "split": str(split_index.get("split", "unspecified")),
        "sample_count": int(split_index.get("sample_count", len(samples))),
        "samples": [build_env_cache_sample(sample) for sample in samples],
    }


def run_random_baseline(
    *,
    env_cache: dict[str, Any],
    run_dir: Path,
    environment_config: dict[str, Any],
    reward_config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    env = ObfuscationEnv(
        env_cache=env_cache,
        max_steps=int(environment_config.get("max_steps", 4)),
        semantic_threshold=float(environment_config.get("semantic_threshold", 0.97)),
        reward_config=reward_config,
        action_safety_config=dict(environment_config.get("action_safety", {})),
        verification_config=dict(environment_config.get("verification", {})),
        artifact_root=run_dir / "artifacts" / "baselines" / "random",
    )
    rng = random.Random(int(baseline_config.get("seed", 7)))
    episodes: list[dict[str, Any]] = []
    sample_ids = env.available_sample_ids()
    started_at = time.monotonic()

    for index, sample_id in enumerate(sample_ids, start=1):
        print_progress("random", index, len(sample_ids), sample_id, started_at)
        state = env.reset(sample_id=sample_id)
        done = False
        info: dict[str, Any] = {
            "sample_id": sample_id,
            "reward_breakdown": {"total_reward": 0.0},
            "semantic_score": state.semantic_score,
            "llvm_block_coverage": state.llvm_block_coverage,
            "branch_rewrite_ratio": state.branch_rewrite_ratio,
            "ir_structural_delta": state.ir_structural_delta,
            "compile_succeeded": state.compile_succeeded,
            "verification_summary": {"verifier_coverage": state.verifier_coverage},
            "termination_reason": "reset",
            "trace_path": "",
        }
        reward = 0.0
        while not done:
            candidates = [action for action in state.legal_actions if action != "stop"]
            action = rng.choice(candidates) if candidates else "stop"
            state, reward, done, info = env.step(action)
        episodes.append(build_episode_result(state, reward, info, baseline_name="random"))

    return summarize_baseline("random", episodes)


def run_rule_based_baseline(
    *,
    env_cache: dict[str, Any],
    run_dir: Path,
    environment_config: dict[str, Any],
    reward_config: dict[str, Any],
    baseline_config: dict[str, Any],
) -> dict[str, Any]:
    env = ObfuscationEnv(
        env_cache=env_cache,
        max_steps=int(environment_config.get("max_steps", 4)),
        semantic_threshold=float(environment_config.get("semantic_threshold", 0.97)),
        reward_config=reward_config,
        action_safety_config=dict(environment_config.get("action_safety", {})),
        verification_config=dict(environment_config.get("verification", {})),
        artifact_root=run_dir / "artifacts" / "baselines" / "rule_based",
    )
    priority = list(
        baseline_config.get(
            "operator_priority",
            [
                "flatten_cfg",
                "substitute_instructions",
                "split_blocks",
                "encode_literals",
                "rename_locals",
            ],
        )
    )
    episodes: list[dict[str, Any]] = []
    sample_ids = env.available_sample_ids()
    started_at = time.monotonic()

    for index, sample_id in enumerate(sample_ids, start=1):
        print_progress("rule_based", index, len(sample_ids), sample_id, started_at)
        state = env.reset(sample_id=sample_id)
        done = False
        info: dict[str, Any] = {
            "sample_id": sample_id,
            "reward_breakdown": {"total_reward": 0.0},
            "semantic_score": state.semantic_score,
            "llvm_block_coverage": state.llvm_block_coverage,
            "branch_rewrite_ratio": state.branch_rewrite_ratio,
            "ir_structural_delta": state.ir_structural_delta,
            "compile_succeeded": state.compile_succeeded,
            "verification_summary": {"verifier_coverage": state.verifier_coverage},
            "termination_reason": "reset",
            "trace_path": "",
        }
        reward = 0.0
        while not done:
            action = select_rule_action(state.legal_actions, priority)
            state, reward, done, info = env.step(action)
        episodes.append(build_episode_result(state, reward, info, baseline_name="rule_based"))

    return summarize_baseline("rule_based", episodes)


def select_rule_action(legal_actions: list[str], priority: list[str]) -> str:
    non_stop = [action for action in legal_actions if action != "stop"]
    for operator_name in priority:
        for action in non_stop:
            if action.startswith(f"{operator_name}:"):
                return action
    return "stop"


def run_external_tool_baseline(
    *,
    baseline_name: str,
    split_index: dict[str, Any],
    run_dir: Path,
    baseline_config: dict[str, Any],
    environment_config: dict[str, Any],
) -> dict[str, Any]:
    tool_path = resolve_tool_path(str(baseline_config.get("tool_path", "")))
    command_template = list(baseline_config.get("command_template", []))
    sample_limit = int(baseline_config.get("sample_limit", 0) or 0)
    if tool_path is None:
        return {
            "baseline_name": baseline_name,
            "status": "skipped_tool_missing",
            "reason": f"{baseline_name} tool was not found. Configure tool_path or install the tool.",
            "episodes": [],
        }
    if not command_template:
        return {
            "baseline_name": baseline_name,
            "status": "skipped_not_configured",
            "reason": f"{baseline_name} command_template was not configured.",
            "episodes": [],
        }

    episodes: list[dict[str, Any]] = []
    reports_root = ensure_dir(run_dir / "artifacts" / "baselines" / baseline_name)
    semantic_threshold = float(environment_config.get("semantic_threshold", 0.97))
    verification_config = dict(environment_config.get("verification", {}))
    compiler = baseline_config.get("compiler")
    compiler_flags = list(baseline_config.get("compiler_flags", []))
    timeout_sec = float(baseline_config.get("timeout_sec", 30.0))
    verification_command_template = list(baseline_config.get("verification_command_template", []))

    samples = list(split_index.get("samples", []))
    if sample_limit > 0:
        samples = samples[:sample_limit]
    started_at = time.monotonic()

    for index, sample in enumerate(samples, start=1):
        sample_dict = dict(sample)
        print_progress(baseline_name, index, len(samples), str(sample_dict["sample_id"]), started_at)
        sample_root = ensure_dir(reports_root / sample_dict["sample_id"].replace("/", "_"))
        source_path = resolve_path(str(sample_dict["source_path"]), PROJECT_ROOT)
        output_path = sample_root / f"{source_path.stem}.{baseline_name}.c"
        tigress_home = resolve_path(str(baseline_config.get("tigress_home", "")), PROJECT_ROOT) if baseline_config.get("tigress_home") else Path(tool_path).parent
        command = render_external_template(
            command_template,
            tool_path=tool_path,
            tigress_home=tigress_home,
            source_path=source_path,
            output_path=output_path,
            sample_root=sample_root,
            sample_id=str(sample_dict["sample_id"]),
        )

        try:
            completed = subprocess.run(
                command,
                cwd=str(sample_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            episodes.append(
                {
                    "baseline_name": baseline_name,
                    "sample_id": str(sample_dict["sample_id"]),
                    "status": "timeout",
                    "transform_command": command,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                }
            )
            continue

        if completed.returncode != 0 or not output_path.exists():
            episodes.append(
                {
                    "baseline_name": baseline_name,
                    "sample_id": str(sample_dict["sample_id"]),
                    "status": "transform_failed",
                    "transform_command": command,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            continue

        if verification_command_template:
            harness = dict(sample_dict.get("harness", {}))
            harness_path = str(harness.get("path", ""))
            test_cases = load_harness_test_cases(harness_path) if harness.get("exists") and harness_path else []
            summary_dict = run_external_compile_verification(
                command_template=verification_command_template,
                tool_path=tool_path,
                tigress_home=tigress_home,
                source_path=source_path,
                output_path=output_path,
                sample_root=sample_root,
                sample_id=str(sample_dict["sample_id"]),
                semantic_threshold=semantic_threshold,
                test_cases=test_cases,
                fuzz_enabled=bool(verification_config.get("fuzz_enabled", True)),
                fuzz_max_cases=int(verification_config.get("fuzz_max_cases", 16)),
                timeout_sec=timeout_sec,
            )
            semantic_score = float(summary_dict["semantic_score"])
            verifier_coverage = float(summary_dict["verifier_coverage"])
            compile_succeeded = bool(summary_dict["compile"]["succeeded"])
            tests_passed = bool(summary_dict.get("tests", {}).get("passed", False))
            diff_executed = bool(summary_dict.get("diff", {}).get("executed", False))
            fuzz_executed = bool(summary_dict.get("fuzz", {}).get("executed", False))
        else:
            harness = dict(sample_dict.get("harness", {}))
            harness_path = str(harness.get("path", ""))
            test_cases = load_harness_test_cases(harness_path) if harness.get("exists") and harness_path else []
            pipeline = VerificationPipeline(
                semantic_threshold=semantic_threshold,
                compiler=compiler,
                compiler_flags=compiler_flags,
                compile_timeout_sec=timeout_sec,
                build_root=sample_root / "verifier",
            )
            summary = pipeline.verify_source(
                source_path=output_path,
                reference_source_path=source_path,
                test_cases=test_cases,
                sample_id=f"{sample_dict['sample_id']}_{baseline_name}",
                compiler=compiler,
                compiler_flags=compiler_flags,
                compile_timeout_sec=timeout_sec,
                workdir=sample_root,
            )
            summary_dict = summary.to_dict()
            semantic_score = summary.semantic_score
            verifier_coverage = summary.verifier_coverage
            compile_succeeded = summary.compile.succeeded
            tests_passed = summary.tests.passed
            diff_executed = summary.diff.executed
            fuzz_executed = summary.fuzz.executed
        external_metrics = compute_external_obfuscation_metrics(
            original_source_path=source_path,
            transformed_source_path=output_path,
        )
        episodes.append(
            {
                "baseline_name": baseline_name,
                "sample_id": str(sample_dict["sample_id"]),
                "status": "completed",
                "transform_command": command,
                "transform_stdout": completed.stdout,
                "transform_stderr": completed.stderr,
                "semantic_score": semantic_score,
                "verifier_coverage": verifier_coverage,
                "compile_succeeded": compile_succeeded,
                "tests_passed": tests_passed,
                "diff_executed": diff_executed,
                "fuzz_executed": fuzz_executed,
                "verification_summary": summary_dict,
                **external_metrics,
            }
        )

    return summarize_external_baseline(baseline_name, episodes)


def resolve_tool_path(explicit_path: str) -> str | None:
    if explicit_path:
        resolved = shutil.which(explicit_path)
        if resolved:
            return resolved
        path = Path(explicit_path)
        if path.is_file():
            return str(path)
    return None


def render_external_template(
    command_template: list[Any],
    *,
    tool_path: str,
    tigress_home: Path,
    source_path: Path,
    output_path: Path,
    verify_source_path: Path | None = None,
    verify_executable_path: Path | None = None,
    sample_root: Path,
    sample_id: str,
) -> list[str]:
    source_stem = source_path.stem
    actual_verify_source_path = verify_source_path or output_path
    actual_verify_executable_path = verify_executable_path or (sample_root / f"{source_stem}.external_verify.exe")
    return [
        str(token).format(
            tool=tool_path,
            tool_posix=path_for_posix_shell(Path(tool_path)),
            tigress_home=str(tigress_home),
            tigress_home_posix=path_for_posix_shell(tigress_home),
            input=str(source_path),
            input_posix=path_for_posix_shell(source_path),
            output=str(output_path),
            output_posix=path_for_posix_shell(output_path),
            verify_source=str(actual_verify_source_path),
            verify_source_posix=path_for_posix_shell(actual_verify_source_path),
            verify_executable=str(actual_verify_executable_path),
            verify_executable_posix=path_for_posix_shell(actual_verify_executable_path),
            source_stem=source_stem,
            sample_id=sample_id,
            workdir=str(sample_root),
            workdir_posix=path_for_posix_shell(sample_root),
        )
        for token in command_template
    ]


def run_external_compile_verification(
    *,
    command_template: list[Any],
    tool_path: str,
    tigress_home: Path,
    source_path: Path,
    output_path: Path,
    sample_root: Path,
    sample_id: str,
    semantic_threshold: float,
    test_cases: list[dict[str, object]] | list[object] | None,
    fuzz_enabled: bool,
    fuzz_max_cases: int,
    timeout_sec: float,
) -> dict[str, Any]:
    candidate_executable = sample_root / f"{output_path.stem}.external_candidate.exe"
    reference_executable = sample_root / f"{source_path.stem}.external_reference.exe"

    candidate_compile = run_external_compile_command(
        command_template=command_template,
        tool_path=tool_path,
        tigress_home=tigress_home,
        source_path=source_path,
        output_path=output_path,
        verify_source_path=output_path,
        verify_executable_path=candidate_executable,
        sample_root=sample_root,
        sample_id=sample_id,
        timeout_sec=timeout_sec,
    )
    reference_compile = run_external_compile_command(
        command_template=command_template,
        tool_path=tool_path,
        tigress_home=tigress_home,
        source_path=source_path,
        output_path=output_path,
        verify_source_path=source_path,
        verify_executable_path=reference_executable,
        sample_root=sample_root,
        sample_id=sample_id,
        timeout_sec=timeout_sec,
    )

    compile_succeeded = bool(candidate_compile["succeeded"]) and bool(reference_compile["succeeded"])
    coerced_cases = coerce_test_cases(test_cases)
    test_result = run_external_test_cases(
        tool_path=tool_path,
        executable_path=candidate_executable,
        test_cases=coerced_cases,
        workdir=sample_root,
    ) if bool(candidate_compile["succeeded"]) else TestResult(
        passed=False,
        passed_cases=0,
        failed_cases=len(coerced_cases),
        cases=[],
        notes=["Candidate compilation failed, so no external test cases were executed."],
    )

    if compile_succeeded:
        diff_result = run_external_differential_cases(
            tool_path=tool_path,
            reference_executable=reference_executable,
            candidate_executable=candidate_executable,
            test_cases=coerced_cases,
            workdir=sample_root,
        )
        fuzz_result = run_external_fuzz_campaign(
            tool_path=tool_path,
            reference_executable=reference_executable,
            candidate_executable=candidate_executable,
            seed_cases=coerced_cases,
            workdir=sample_root,
            max_cases=fuzz_max_cases,
        ) if fuzz_enabled else FuzzResult(
            executed=False,
            passed=True,
            notes=["Fuzzing was disabled by external verifier configuration."],
        )
    else:
        diff_result = DiffResult(
            executed=False,
            passed=False,
            notes=["Reference or candidate source failed to compile, so external diff execution was skipped."],
        )
        fuzz_result = FuzzResult(
            executed=False,
            passed=False,
            notes=["Reference or candidate source failed to compile, so external fuzzing was skipped."],
        )

    semantic_score, verifier_coverage, available_signals = score_external_semantics(
        compile_succeeded=compile_succeeded,
        test_result=test_result,
        has_tests=bool(coerced_cases),
        diff_result=diff_result,
        fuzz_result=fuzz_result,
    )
    compile_result = {
        **candidate_compile,
        "reference": reference_compile,
        "notes": [
            "External WSL compile verification was used.",
            *list(candidate_compile.get("notes", [])),
        ],
    }
    return {
        "sample_id": sample_id,
        "semantic_threshold": semantic_threshold,
        "semantic_score": semantic_score,
        "verifier_coverage": verifier_coverage,
        "available_signals": available_signals,
        "compile": compile_result,
        "tests": test_result.__dict__ | {"cases": [case.__dict__ for case in test_result.cases]},
        "diff": diff_result.__dict__ | {"cases": [case.__dict__ for case in diff_result.cases]},
        "fuzz": fuzz_result.__dict__ | {"cases": [case.__dict__ for case in fuzz_result.cases]},
        "fuzz_diff_error_rate": fuzz_result.diff_error_rate,
        "source_path": str(output_path),
        "executable_path": str(candidate_executable),
        "notes": ["External WSL verifier was used for Tigress-compatible execution."],
    }


def run_external_compile_command(
    *,
    command_template: list[Any],
    tool_path: str,
    tigress_home: Path,
    source_path: Path,
    output_path: Path,
    verify_source_path: Path,
    verify_executable_path: Path,
    sample_root: Path,
    sample_id: str,
    timeout_sec: float,
) -> dict[str, Any]:
    command = render_external_template(
        command_template,
        tool_path=tool_path,
        tigress_home=tigress_home,
        source_path=source_path,
        output_path=output_path,
        verify_source_path=verify_source_path,
        verify_executable_path=verify_executable_path,
        sample_root=sample_root,
        sample_id=sample_id,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(sample_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        completed = subprocess.CompletedProcess(
            completed.args,
            completed.returncode,
            completed.stdout,
            clean_wsl_wrapper_stderr(completed.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "succeeded": False,
            "command": " ".join(command),
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nExternal compile verification timed out.",
            "output_path": str(verify_executable_path),
            "notes": [f"External compile verification exceeded timeout of {timeout_sec} seconds."],
        }

    return {
        "succeeded": completed.returncode == 0 and verify_executable_path.exists(),
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": clean_wsl_wrapper_stderr(completed.stderr),
        "output_path": str(verify_executable_path),
        "notes": ["External WSL compile verification was used."],
    }


def score_external_semantics(
    *,
    compile_succeeded: bool,
    test_result: TestResult,
    has_tests: bool,
    diff_result: DiffResult,
    fuzz_result: FuzzResult,
) -> tuple[float, float, list[str]]:
    weights = {
        "compile": 0.35,
        "tests": 0.30,
        "fuzz": 0.20,
        "symbolic": 0.10,
        "functional_similarity": 0.05,
    }
    signals: dict[str, float] = {"compile": 1.0 if compile_succeeded else 0.0}
    available_signals = ["compile"]

    if has_tests:
        total_cases = test_result.passed_cases + test_result.failed_cases
        signals["tests"] = 1.0 if total_cases == 0 else test_result.passed_cases / total_cases
        available_signals.append("tests")

    if diff_result.executed:
        total_cases = diff_result.passed_cases + diff_result.failed_cases
        signals["functional_similarity"] = 1.0 if total_cases == 0 else diff_result.passed_cases / total_cases
        available_signals.append("functional_similarity")

    if fuzz_result.executed:
        signals["fuzz"] = 1.0 - float(min(max(fuzz_result.diff_error_rate, 0.0), 1.0))
        available_signals.append("fuzz")

    total_weight = sum(weights[name] for name in available_signals)
    weighted_score = sum(weights[name] * signals[name] for name in available_signals)
    coverage = total_weight / sum(weights.values())
    return (weighted_score / total_weight if total_weight else 0.0), coverage, available_signals


def clean_wsl_wrapper_stderr(stderr: str) -> str:
    """Remove WSL launcher warnings so harness checks compare program stderr only."""
    raw = str(stderr or "")
    if "\x00" in raw and ("W\x00S\x00L" in raw or "w\x00s\x00l" in raw or "l\x00o\x00c\x00a\x00l\x00h\x00o\x00s\x00t" in raw):
        return ""
    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        compact = line.replace("\x00", "").strip()
        if compact.lower().startswith("wsl:"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def run_external_process(
    *,
    tool_path: str,
    executable_path: Path,
    case: TestCase,
    input_data: str,
    workdir: Path,
) -> subprocess.CompletedProcess[str]:
    args = " ".join(shlex.quote(str(arg)) for arg in case.args)
    executable = shlex.quote(path_for_posix_shell(executable_path))
    cd_workdir = shlex.quote(path_for_posix_shell(workdir))
    command_text = f"cd {cd_workdir}; {executable}" + (f" {args}" if args else "")
    command = [tool_path, "bash", "-lc", command_text]
    try:
        completed = subprocess.run(
            command,
            cwd=str(workdir),
            input=input_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=case.timeout_sec,
            check=False,
        )
        return subprocess.CompletedProcess(
            completed.args,
            completed.returncode,
            completed.stdout,
            clean_wsl_wrapper_stderr(completed.stderr),
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            exc.stdout or "",
            (exc.stderr or "") + "\nExternal WSL execution timed out.",
        )


def run_external_test_cases(
    *,
    tool_path: str,
    executable_path: Path,
    test_cases: list[TestCase],
    workdir: Path,
) -> TestResult:
    if not test_cases:
        return TestResult(
            passed=True,
            passed_cases=0,
            failed_cases=0,
            cases=[],
            notes=["No test cases were supplied, so external testing was skipped."],
        )

    case_results: list[TestCaseResult] = []
    for case in test_cases:
        completed = run_external_process(
            tool_path=tool_path,
            executable_path=executable_path,
            case=case,
            input_data=case.input_data,
            workdir=workdir,
        )
        stdout_matches = normalize_text(
            completed.stdout,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        ) == normalize_text(
            case.expected_stdout,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        stderr_matches = normalize_text(
            completed.stderr,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        ) == normalize_text(
            case.expected_stderr,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        returncode_matches = completed.returncode == case.expected_returncode
        notes: list[str] = []
        if not stdout_matches:
            notes.append("stdout mismatch")
        if not stderr_matches:
            notes.append("stderr mismatch")
        if not returncode_matches:
            notes.append("return code mismatch")
        if completed.returncode == 124:
            notes.append(f"execution timed out after {case.timeout_sec} seconds")

        case_results.append(
            TestCaseResult(
                name=case.name,
                passed=stdout_matches and stderr_matches and returncode_matches,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                expected_stdout=case.expected_stdout,
                expected_stderr=case.expected_stderr,
                expected_returncode=case.expected_returncode,
                notes=notes,
            )
        )

    passed_cases = sum(1 for case in case_results if case.passed)
    failed_cases = len(case_results) - passed_cases
    return TestResult(
        passed=failed_cases == 0,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        cases=case_results,
    )


def run_external_differential_cases(
    *,
    tool_path: str,
    reference_executable: Path,
    candidate_executable: Path,
    test_cases: list[TestCase],
    workdir: Path,
) -> DiffResult:
    if not test_cases:
        return DiffResult(
            executed=False,
            passed=True,
            notes=["No differential test cases were supplied, so external diff execution was skipped."],
        )

    case_results: list[DiffCaseResult] = []
    for case in test_cases:
        reference_run = run_external_process(
            tool_path=tool_path,
            executable_path=reference_executable,
            case=case,
            input_data=case.input_data,
            workdir=workdir,
        )
        candidate_run = run_external_process(
            tool_path=tool_path,
            executable_path=candidate_executable,
            case=case,
            input_data=case.input_data,
            workdir=workdir,
        )
        reference_stdout = normalize_text(
            reference_run.stdout,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        candidate_stdout = normalize_text(
            candidate_run.stdout,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        reference_stderr = normalize_text(
            reference_run.stderr,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        candidate_stderr = normalize_text(
            candidate_run.stderr,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        passed = (
            reference_stdout == candidate_stdout
            and reference_stderr == candidate_stderr
            and reference_run.returncode == candidate_run.returncode
        )
        notes: list[str] = []
        if reference_stdout != candidate_stdout:
            notes.append("stdout diverged from reference execution")
        if reference_stderr != candidate_stderr:
            notes.append("stderr diverged from reference execution")
        if reference_run.returncode != candidate_run.returncode:
            notes.append("return code diverged from reference execution")
        if reference_run.returncode == 124 or candidate_run.returncode == 124:
            notes.append(f"execution timed out after {case.timeout_sec} seconds")
        case_results.append(
            DiffCaseResult(
                name=case.name,
                passed=passed,
                input_data=case.input_data,
                args=list(case.args),
                reference_returncode=reference_run.returncode,
                candidate_returncode=candidate_run.returncode,
                reference_stdout=reference_run.stdout,
                candidate_stdout=candidate_run.stdout,
                reference_stderr=reference_run.stderr,
                candidate_stderr=candidate_run.stderr,
                notes=notes,
            )
        )

    passed_cases = sum(1 for case in case_results if case.passed)
    failed_cases = len(case_results) - passed_cases
    return DiffResult(
        executed=True,
        passed=failed_cases == 0,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        mismatch_cases=failed_cases,
        cases=case_results,
    )


def build_external_fuzz_mutations(seed_text: str) -> list[tuple[str, str]]:
    if contains_binary_grid_lines(seed_text):
        return []

    def replace_first_integer(transform: Any, *, require_positive: bool = True) -> str | None:
        match = INTEGER_PATTERN.search(seed_text)
        if match is None:
            return None
        value = int(match.group(0))
        new_value = int(transform(value))
        if require_positive and value > 0 and new_value <= 0:
            return None
        return seed_text[: match.start()] + str(new_value) + seed_text[match.end() :]

    def swap_first_two_integers() -> str | None:
        matches = list(INTEGER_PATTERN.finditer(seed_text))
        if len(matches) < 2:
            return None
        first, second = matches[0], matches[1]
        if int(first.group(0)) > 0 and int(second.group(0)) <= 0:
            return None
        return (
            seed_text[: first.start()]
            + second.group(0)
            + seed_text[first.end() : second.start()]
            + first.group(0)
            + seed_text[second.end() :]
        )

    candidates = [
        ("increment_first_int", replace_first_integer(lambda value: value + 1)),
        ("decrement_first_int", replace_first_integer(lambda value: value - 1)),
        ("swap_first_two_ints", swap_first_two_integers()),
    ]
    mutations: list[tuple[str, str]] = []
    seen = {seed_text}
    for name, value in candidates:
        if value is None or value in seen:
            continue
        seen.add(value)
        mutations.append((name, value))
    return mutations


def run_external_fuzz_campaign(
    *,
    tool_path: str,
    reference_executable: Path,
    candidate_executable: Path,
    seed_cases: list[TestCase],
    workdir: Path,
    max_cases: int,
) -> FuzzResult:
    if not seed_cases:
        return FuzzResult(
            executed=False,
            passed=True,
            notes=["No seed cases were supplied, so external fuzzing was skipped."],
        )

    case_results: list[FuzzCaseResult] = []
    for seed_case in seed_cases:
        for mutation_name, mutated_input in build_external_fuzz_mutations(seed_case.input_data):
            reference_run = run_external_process(
                tool_path=tool_path,
                executable_path=reference_executable,
                case=seed_case,
                input_data=mutated_input,
                workdir=workdir,
            )
            candidate_run = run_external_process(
                tool_path=tool_path,
                executable_path=candidate_executable,
                case=seed_case,
                input_data=mutated_input,
                workdir=workdir,
            )
            reference_stdout = normalize_text(
                reference_run.stdout,
                strip_output=seed_case.strip_output,
                normalize_newlines=seed_case.normalize_newlines,
                collapse_whitespace=seed_case.collapse_whitespace,
            )
            candidate_stdout = normalize_text(
                candidate_run.stdout,
                strip_output=seed_case.strip_output,
                normalize_newlines=seed_case.normalize_newlines,
                collapse_whitespace=seed_case.collapse_whitespace,
            )
            reference_stderr = normalize_text(
                reference_run.stderr,
                strip_output=seed_case.strip_output,
                normalize_newlines=seed_case.normalize_newlines,
                collapse_whitespace=seed_case.collapse_whitespace,
            )
            candidate_stderr = normalize_text(
                candidate_run.stderr,
                strip_output=seed_case.strip_output,
                normalize_newlines=seed_case.normalize_newlines,
                collapse_whitespace=seed_case.collapse_whitespace,
            )
            passed = (
                reference_stdout == candidate_stdout
                and reference_stderr == candidate_stderr
                and reference_run.returncode == candidate_run.returncode
            )
            notes: list[str] = []
            if reference_stdout != candidate_stdout:
                notes.append("stdout diverged under fuzz input")
            if reference_stderr != candidate_stderr:
                notes.append("stderr diverged under fuzz input")
            if reference_run.returncode != candidate_run.returncode:
                notes.append("return code diverged under fuzz input")
            case_results.append(
                FuzzCaseResult(
                    name=f"{seed_case.name}:{mutation_name}",
                    seed_name=seed_case.name,
                    mutation_name=mutation_name,
                    input_data=mutated_input,
                    args=list(seed_case.args),
                    passed=passed,
                    reference_returncode=reference_run.returncode,
                    candidate_returncode=candidate_run.returncode,
                    reference_stdout=reference_run.stdout,
                    candidate_stdout=candidate_run.stdout,
                    reference_stderr=reference_run.stderr,
                    candidate_stderr=candidate_run.stderr,
                    notes=notes,
                )
            )
            if len(case_results) >= max_cases:
                break
        if len(case_results) >= max_cases:
            break

    if not case_results:
        return FuzzResult(
            executed=False,
            passed=True,
            notes=["Seed cases did not produce any usable external fuzz mutations."],
        )

    passed_cases = sum(1 for case in case_results if case.passed)
    failed_cases = len(case_results) - passed_cases
    mismatch_cases = failed_cases
    crash_cases = sum(
        1
        for case in case_results
        if (case.reference_returncode != 0 or case.candidate_returncode != 0) and not case.passed
    )
    return FuzzResult(
        executed=True,
        passed=failed_cases == 0,
        generated_cases=len(case_results),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        mismatch_cases=mismatch_cases,
        crash_cases=crash_cases,
        diff_error_rate=(mismatch_cases / len(case_results)) if case_results else 0.0,
        cases=case_results,
    )


def path_for_posix_shell(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    parts = resolved.parts
    if drive and len(parts) > 1:
        tail = "/".join(part.replace("\\", "/") for part in parts[1:])
        return f"/mnt/{drive}/{tail}"
    return resolved.as_posix()


def collect_source_llvm_block_ids(source_text: str) -> set[str]:
    try:
        analysis = analyze_source(source_text)
    except Exception:
        return set()
    return {
        normalize_block_id(block.block_id)
        for block in analysis.clang_summary.llvm_ir_blocks
        if normalize_block_id(block.block_id)
    }


def collect_source_branch_block_ids(source_text: str) -> set[str]:
    try:
        analysis = analyze_source(source_text)
    except Exception:
        return set()
    branch_blocks: set[str] = set()
    for branch in analysis.clang_summary.branches:
        llvm_block_id = normalize_block_id(branch.metadata.get("llvm_block_id", ""))
        if llvm_block_id:
            branch_blocks.add(llvm_block_id)
    return branch_blocks


def compute_external_obfuscation_metrics(
    *,
    original_source_path: Path,
    transformed_source_path: Path,
) -> dict[str, Any]:
    original_text = original_source_path.read_text(encoding="utf-8", errors="replace")
    transformed_text = transformed_source_path.read_text(encoding="utf-8", errors="replace")
    edit_delta = 1.0 - SequenceMatcher(None, original_text, transformed_text).ratio()
    original_lines = len(original_text.splitlines())
    transformed_lines = len(transformed_text.splitlines())
    line_growth = max(transformed_lines - original_lines, 0) / max(float(original_lines), 1.0)
    potency_score = float(min(max((0.65 * edit_delta) + (0.35 * min(line_growth, 1.0)), 0.0), 1.0))
    cost_penalty = float(min(max(line_growth, 0.0), 1.0))

    baseline_blocks = collect_source_llvm_block_ids(original_text)
    current_blocks = collect_source_llvm_block_ids(transformed_text)
    baseline_branch_blocks = collect_source_branch_block_ids(original_text)
    current_branch_blocks = collect_source_branch_block_ids(transformed_text)
    ir_metrics = build_ir_metrics_snapshot(
        baseline_llvm_blocks=baseline_blocks,
        current_llvm_blocks=current_blocks,
        touched_llvm_blocks=baseline_blocks.intersection(current_blocks.symmetric_difference(baseline_blocks)),
        baseline_branch_blocks=baseline_branch_blocks,
        rewritten_branch_blocks=baseline_branch_blocks.symmetric_difference(current_branch_blocks),
    )

    return {
        "potency_score": potency_score,
        "cost_penalty": cost_penalty,
        "llvm_metric_available": bool(ir_metrics.get("llvm_metric_available", False)),
        "llvm_block_coverage": float(ir_metrics.get("llvm_block_coverage", 0.0)),
        "branch_rewrite_ratio": float(ir_metrics.get("branch_rewrite_ratio", 0.0)),
        "ir_structural_delta": float(ir_metrics.get("ir_structural_delta", 0.0)),
        "ir_metrics": ir_metrics,
    }


def build_episode_result(state: Any, reward: float, info: dict[str, Any], *, baseline_name: str) -> dict[str, Any]:
    return {
        "baseline_name": baseline_name,
        "sample_id": str(info.get("sample_id", state.sample_id)),
        "status": "completed",
        "step_index": int(state.step_index),
        "reward": float(reward),
        "semantic_score": float(info.get("semantic_score", state.semantic_score)),
        "verifier_coverage": float(info.get("verification_summary", {}).get("verifier_coverage", state.verifier_coverage)),
        "potency_score": float(state.potency_score),
        "cost_penalty": float(state.cost_penalty),
        "llvm_block_coverage": float(info.get("llvm_block_coverage", state.llvm_block_coverage)),
        "branch_rewrite_ratio": float(info.get("branch_rewrite_ratio", state.branch_rewrite_ratio)),
        "ir_structural_delta": float(info.get("ir_structural_delta", state.ir_structural_delta)),
        "llvm_metric_available": bool(info.get("llvm_metric_available", state.llvm_metric_available)),
        "compile_succeeded": bool(info.get("compile_succeeded", state.compile_succeeded)),
        "termination_reason": str(info.get("termination_reason", "")),
        "trace_path": str(info.get("trace_path", "")),
        "reward_breakdown": dict(info.get("reward_breakdown", {})),
    }


def summarize_baseline(baseline_name: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [episode for episode in episodes if episode.get("status") == "completed"]
    llvm_ready = [episode for episode in completed if bool(episode.get("llvm_metric_available", False))]
    return {
        "baseline_name": baseline_name,
        "status": "completed",
        "episode_count": len(episodes),
        "completed_episode_count": len(completed),
        "mean_reward": mean([float(item["reward"]) for item in completed]) if completed else 0.0,
        "mean_semantic_score": mean([float(item["semantic_score"]) for item in completed]) if completed else 0.0,
        "mean_verifier_coverage": mean([float(item["verifier_coverage"]) for item in completed]) if completed else 0.0,
        "mean_llvm_block_coverage": mean([float(item["llvm_block_coverage"]) for item in llvm_ready]) if llvm_ready else 0.0,
        "mean_branch_rewrite_ratio": mean([float(item["branch_rewrite_ratio"]) for item in llvm_ready]) if llvm_ready else 0.0,
        "mean_ir_structural_delta": mean([float(item["ir_structural_delta"]) for item in llvm_ready]) if llvm_ready else 0.0,
        "episodes": episodes,
    }


def summarize_external_baseline(baseline_name: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [episode for episode in episodes if episode.get("status") == "completed"]
    llvm_ready = [episode for episode in completed if bool(episode.get("llvm_metric_available", False))]
    if not completed:
        return {
            "baseline_name": baseline_name,
            "status": "completed_with_no_successful_samples",
            "episode_count": len(episodes),
            "episodes": episodes,
        }
    return {
        "baseline_name": baseline_name,
        "status": "completed",
        "episode_count": len(episodes),
        "completed_episode_count": len(completed),
        "mean_semantic_score": mean([float(item["semantic_score"]) for item in completed]),
        "mean_verifier_coverage": mean([float(item["verifier_coverage"]) for item in completed]),
        "mean_potency_score": mean([float(item.get("potency_score", 0.0)) for item in completed]),
        "mean_cost_penalty": mean([float(item.get("cost_penalty", 0.0)) for item in completed]),
        "mean_llvm_block_coverage": mean([float(item.get("llvm_block_coverage", 0.0)) for item in llvm_ready]) if llvm_ready else 0.0,
        "mean_branch_rewrite_ratio": mean([float(item.get("branch_rewrite_ratio", 0.0)) for item in llvm_ready]) if llvm_ready else 0.0,
        "mean_ir_structural_delta": mean([float(item.get("ir_structural_delta", 0.0)) for item in llvm_ready]) if llvm_ready else 0.0,
        "episodes": episodes,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path, config = load_config(args.config)

    job_name = str(config.get("job_name", "baseline_run"))
    split_index_path, split_index = load_split_index(config["split_index_path"])
    baselines = dict(config.get("baselines", {}))
    environment_config = dict(config.get("environment", {}))
    reward_config = dict(config.get("reward", {}))
    run_root = resolve_path(config.get("run_root", "artifacts/baselines"), PROJECT_ROOT)
    run_dir = resolve_path(args.run_dir, PROJECT_ROOT) if args.run_dir else create_timestamped_dir(run_root, job_name)
    ensure_dir(run_dir)
    ensure_dir(run_dir / "reports")
    copy_file(config_path, run_dir / f"baseline.config{config_path.suffix}")

    env_cache = build_env_cache(split_index)
    dump_json(run_dir / "data" / "baseline_env_input_cache.json", env_cache)

    selected = list(baselines.get("selected", ["random", "rule_based"]))
    results: dict[str, Any] = {}
    for baseline_name in selected:
        if baseline_name == "random":
            results[baseline_name] = run_random_baseline(
                env_cache=env_cache,
                run_dir=run_dir,
                environment_config=environment_config,
                reward_config=reward_config,
                baseline_config=dict(baselines.get("random", {})),
            )
        elif baseline_name == "rule_based":
            results[baseline_name] = run_rule_based_baseline(
                env_cache=env_cache,
                run_dir=run_dir,
                environment_config=environment_config,
                reward_config=reward_config,
                baseline_config=dict(baselines.get("rule_based", {})),
            )
        elif baseline_name in {"tigress", "ollvm"}:
            results[baseline_name] = run_external_tool_baseline(
                baseline_name=baseline_name,
                split_index=split_index,
                run_dir=run_dir,
                baseline_config=dict(baselines.get(baseline_name, {})),
                environment_config=environment_config,
            )
        else:
            results[baseline_name] = {
                "baseline_name": baseline_name,
                "status": "skipped_unknown_baseline",
                "reason": f"Unsupported baseline: {baseline_name}",
                "episodes": [],
            }

    summary = {
        "job_name": job_name,
        "created_at": utc_timestamp(),
        "run_dir": project_relative(run_dir),
        "split_index_path": project_relative(split_index_path),
        "sample_count": int(split_index.get("sample_count", 0)),
        "selected": selected,
        "results": results,
    }
    dump_json(run_dir / "reports" / "baseline_summary.json", summary)
    print(f"Wrote baseline summary: {run_dir / 'reports' / 'baseline_summary.json'}")


if __name__ == "__main__":
    main()
