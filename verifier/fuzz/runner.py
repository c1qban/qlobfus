from __future__ import annotations

import re
import subprocess
from pathlib import Path

from verifier.models import FuzzCaseResult, FuzzResult, TestCase
from verifier.tests.runner import normalize_text


INTEGER_PATTERN = re.compile(r"-?\d+")
BINARY_GRID_LINE_PATTERN = re.compile(r"^[01]{8,}$")


def _contains_binary_grid_lines(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(1 for line in lines if BINARY_GRID_LINE_PATTERN.fullmatch(line)) >= 2


def _replace_first_integer(text: str, transform: callable, *, require_positive: bool = True) -> str | None:
    match = INTEGER_PATTERN.search(text)
    if match is None:
        return None
    value = int(match.group(0))
    new_value = int(transform(value))
    if require_positive and value > 0 and new_value <= 0:
        return None
    return text[: match.start()] + str(new_value) + text[match.end() :]


def _swap_first_two_integers(text: str) -> str | None:
    matches = list(INTEGER_PATTERN.finditer(text))
    if len(matches) < 2:
        return None
    first, second = matches[0], matches[1]
    first_text = first.group(0)
    second_text = second.group(0)
    if int(first_text) > 0 and int(second_text) <= 0:
        return None
    pieces = [
        text[: first.start()],
        second_text,
        text[first.end() : second.start()],
        first_text,
        text[second.end() :],
    ]
    return "".join(pieces)


def _build_mutations(seed_text: str) -> list[tuple[str, str]]:
    if _contains_binary_grid_lines(seed_text):
        return []

    candidates: list[tuple[str, str | None]] = [
        ("increment_first_int", _replace_first_integer(seed_text, lambda value: value + 1)),
        ("decrement_first_int", _replace_first_integer(seed_text, lambda value: value - 1)),
        ("swap_first_two_ints", _swap_first_two_integers(seed_text)),
    ]

    mutations: list[tuple[str, str]] = []
    seen_inputs = {seed_text}
    for mutation_name, candidate in candidates:
        if candidate is None or candidate in seen_inputs:
            continue
        seen_inputs.add(candidate)
        mutations.append((mutation_name, candidate))
    return mutations


def _execute_case(executable_path: Path, case: TestCase, input_data: str, workdir: Path | None) -> subprocess.CompletedProcess[str]:
    command = [str(executable_path), *case.args]
    try:
        return subprocess.run(
            command,
            cwd=str(workdir) if workdir else None,
            input=input_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=case.timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(command, 124, exc.stdout or "", (exc.stderr or "") + "\nExecution timed out.")


def run_fuzz_campaign(
    reference_executable: Path,
    candidate_executable: Path,
    seed_cases: list[TestCase],
    *,
    workdir: Path | None = None,
    max_cases: int = 16,
) -> FuzzResult:
    if not seed_cases:
        return FuzzResult(
            executed=False,
            passed=True,
            notes=["No seed cases were supplied, so fuzzing was skipped."],
        )

    case_results: list[FuzzCaseResult] = []
    for seed_case in seed_cases:
        for mutation_name, mutated_input in _build_mutations(seed_case.input_data):
            reference_run = _execute_case(reference_executable, seed_case, mutated_input, workdir)
            candidate_run = _execute_case(candidate_executable, seed_case, mutated_input, workdir)

            normalized_reference_stdout = normalize_text(
                reference_run.stdout,
                strip_output=seed_case.strip_output,
                normalize_newlines=seed_case.normalize_newlines,
                collapse_whitespace=seed_case.collapse_whitespace,
            )
            normalized_candidate_stdout = normalize_text(
                candidate_run.stdout,
                strip_output=seed_case.strip_output,
                normalize_newlines=seed_case.normalize_newlines,
                collapse_whitespace=seed_case.collapse_whitespace,
            )
            normalized_reference_stderr = normalize_text(
                reference_run.stderr,
                strip_output=seed_case.strip_output,
                normalize_newlines=seed_case.normalize_newlines,
                collapse_whitespace=seed_case.collapse_whitespace,
            )
            normalized_candidate_stderr = normalize_text(
                candidate_run.stderr,
                strip_output=seed_case.strip_output,
                normalize_newlines=seed_case.normalize_newlines,
                collapse_whitespace=seed_case.collapse_whitespace,
            )

            stdout_matches = normalized_reference_stdout == normalized_candidate_stdout
            stderr_matches = normalized_reference_stderr == normalized_candidate_stderr
            returncode_matches = reference_run.returncode == candidate_run.returncode
            reference_failed = reference_run.returncode != 0
            candidate_failed = candidate_run.returncode != 0
            crash_detected = reference_failed or candidate_failed
            equivalent_failure = (
                reference_failed
                and candidate_failed
                and returncode_matches
                and stdout_matches
                and stderr_matches
            )
            reference_invalid = reference_failed
            passed = reference_invalid or (stdout_matches and stderr_matches and returncode_matches)

            notes: list[str] = []
            if reference_invalid:
                notes.append("reference failed under fuzz input; semantic comparison skipped")
            if not reference_invalid and not stdout_matches:
                notes.append("stdout diverged under fuzz input")
            if not reference_invalid and not stderr_matches:
                notes.append("stderr diverged under fuzz input")
            if not reference_invalid and not returncode_matches:
                notes.append("return code diverged under fuzz input")
            if not reference_invalid and crash_detected and not equivalent_failure:
                notes.append("non-zero return code observed under fuzz input")
            if equivalent_failure:
                notes.append("reference and candidate failed equivalently under fuzz input")

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
            notes=["Seed cases did not produce any usable fuzz mutations."],
        )

    passed_cases = sum(1 for case in case_results if case.passed)
    failed_cases = len(case_results) - passed_cases
    crash_cases = sum(
        1
        for case in case_results
        if (case.reference_returncode != 0 or case.candidate_returncode != 0) and not case.passed
    )
    mismatch_cases = sum(1 for case in case_results if not case.passed)
    generated_cases = len(case_results)
    return FuzzResult(
        executed=True,
        passed=failed_cases == 0,
        generated_cases=generated_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        mismatch_cases=mismatch_cases,
        crash_cases=crash_cases,
        diff_error_rate=(mismatch_cases / generated_cases) if generated_cases else 0.0,
        cases=case_results,
        notes=[],
    )
