from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from verifier.models import TestCase, TestCaseResult, TestResult


def normalize_text(
    value: str,
    *,
    strip_output: bool,
    normalize_newlines: bool,
    collapse_whitespace: bool,
) -> str:
    text = value
    if normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    if strip_output:
        text = text.strip()
    return text


def coerce_test_cases(raw_cases: list[TestCase | dict[str, Any]] | None) -> list[TestCase]:
    if not raw_cases:
        return []

    cases: list[TestCase] = []
    for index, raw_case in enumerate(raw_cases):
        if isinstance(raw_case, TestCase):
            cases.append(raw_case)
            continue

        case = TestCase(
            name=str(raw_case.get("name", f"case_{index}")),
            input_data=str(raw_case.get("input_data", raw_case.get("stdin", ""))),
            args=[str(item) for item in raw_case.get("args", [])],
            expected_stdout=str(raw_case.get("expected_stdout", raw_case.get("stdout", ""))),
            expected_stderr=str(raw_case.get("expected_stderr", raw_case.get("stderr", ""))),
            expected_returncode=int(raw_case.get("expected_returncode", raw_case.get("returncode", 0))),
            timeout_sec=float(raw_case.get("timeout_sec", 2.0)),
            strip_output=bool(raw_case.get("strip_output", True)),
            normalize_newlines=bool(raw_case.get("normalize_newlines", True)),
            collapse_whitespace=bool(raw_case.get("collapse_whitespace", False)),
        )
        cases.append(case)

    return cases


def run_test_cases(
    executable_path: Path,
    test_cases: list[TestCase],
    workdir: Path | None = None,
) -> TestResult:
    if not test_cases:
        return TestResult(
            passed=True,
            passed_cases=0,
            failed_cases=0,
            cases=[],
            notes=["No test cases were supplied, so only compilation was checked."],
        )

    case_results: list[TestCaseResult] = []
    for case in test_cases:
        command = [str(executable_path), *case.args]
        try:
            completed = subprocess.run(
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
            passed = stdout_matches and stderr_matches and returncode_matches
            notes: list[str] = []
            if not stdout_matches:
                notes.append("stdout mismatch")
            if not stderr_matches:
                notes.append("stderr mismatch")
            if not returncode_matches:
                notes.append("return code mismatch")
        except subprocess.TimeoutExpired as exc:
            completed_stdout = exc.stdout or ""
            completed_stderr = exc.stderr or ""
            passed = False
            notes = [f"execution timed out after {case.timeout_sec} seconds"]
            completed = subprocess.CompletedProcess(command, 124, completed_stdout, completed_stderr)

        case_results.append(
            TestCaseResult(
                name=case.name,
                passed=passed,
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
        notes=[],
    )
