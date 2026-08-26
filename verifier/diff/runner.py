from __future__ import annotations

import subprocess
from pathlib import Path

from verifier.models import DiffCaseResult, DiffResult, TestCase
from verifier.tests.runner import normalize_text


def _execute_case(executable_path: Path, case: TestCase, workdir: Path | None) -> subprocess.CompletedProcess[str]:
    command = [str(executable_path), *case.args]
    try:
        return subprocess.run(
            command,
            cwd=str(workdir) if workdir else None,
            input=case.input_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=case.timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(command, 124, exc.stdout or "", (exc.stderr or "") + "\nExecution timed out.")


def run_differential_cases(
    reference_executable: Path,
    candidate_executable: Path,
    test_cases: list[TestCase],
    *,
    workdir: Path | None = None,
) -> DiffResult:
    if not test_cases:
        return DiffResult(
            executed=False,
            passed=True,
            notes=["No differential test cases were supplied, so diff execution was skipped."],
        )

    case_results: list[DiffCaseResult] = []
    for case in test_cases:
        reference_run = _execute_case(reference_executable, case, workdir)
        candidate_run = _execute_case(candidate_executable, case, workdir)

        normalized_reference_stdout = normalize_text(
            reference_run.stdout,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        normalized_candidate_stdout = normalize_text(
            candidate_run.stdout,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        normalized_reference_stderr = normalize_text(
            reference_run.stderr,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )
        normalized_candidate_stderr = normalize_text(
            candidate_run.stderr,
            strip_output=case.strip_output,
            normalize_newlines=case.normalize_newlines,
            collapse_whitespace=case.collapse_whitespace,
        )

        stdout_matches = normalized_reference_stdout == normalized_candidate_stdout
        stderr_matches = normalized_reference_stderr == normalized_candidate_stderr
        returncode_matches = reference_run.returncode == candidate_run.returncode
        passed = stdout_matches and stderr_matches and returncode_matches

        notes: list[str] = []
        if not stdout_matches:
            notes.append("stdout diverged from reference execution")
        if not stderr_matches:
            notes.append("stderr diverged from reference execution")
        if not returncode_matches:
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
        notes=[],
    )
