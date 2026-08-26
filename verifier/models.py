from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CompileResult:
    succeeded: bool
    command: str = ""
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    output_path: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompileResult":
        return cls(
            succeeded=bool(payload.get("succeeded", False)),
            command=str(payload.get("command", "")),
            returncode=int(payload.get("returncode", 0)),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            output_path=str(payload.get("output_path", "")),
            notes=[str(note) for note in payload.get("notes", [])],
        )


@dataclass
class TestCase:
    name: str
    input_data: str = ""
    args: list[str] = field(default_factory=list)
    expected_stdout: str = ""
    expected_stderr: str = ""
    expected_returncode: int = 0
    timeout_sec: float = 2.0
    strip_output: bool = True
    normalize_newlines: bool = True
    collapse_whitespace: bool = False


@dataclass
class TestCaseResult:
    name: str
    passed: bool
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    expected_stdout: str = ""
    expected_stderr: str = ""
    expected_returncode: int = 0
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TestCaseResult":
        return cls(
            name=str(payload.get("name", "")),
            passed=bool(payload.get("passed", False)),
            returncode=int(payload.get("returncode", 0)),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            expected_stdout=str(payload.get("expected_stdout", "")),
            expected_stderr=str(payload.get("expected_stderr", "")),
            expected_returncode=int(payload.get("expected_returncode", 0)),
            notes=[str(note) for note in payload.get("notes", [])],
        )


@dataclass
class TestResult:
    passed: bool
    passed_cases: int = 0
    failed_cases: int = 0
    cases: list[TestCaseResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TestResult":
        return cls(
            passed=bool(payload.get("passed", False)),
            passed_cases=int(payload.get("passed_cases", 0)),
            failed_cases=int(payload.get("failed_cases", 0)),
            cases=[TestCaseResult.from_dict(dict(case)) for case in payload.get("cases", [])],
            notes=[str(note) for note in payload.get("notes", [])],
        )


@dataclass
class DiffCaseResult:
    name: str
    passed: bool
    input_data: str = ""
    args: list[str] = field(default_factory=list)
    reference_returncode: int = 0
    candidate_returncode: int = 0
    reference_stdout: str = ""
    candidate_stdout: str = ""
    reference_stderr: str = ""
    candidate_stderr: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiffCaseResult":
        return cls(
            name=str(payload.get("name", "")),
            passed=bool(payload.get("passed", False)),
            input_data=str(payload.get("input_data", "")),
            args=[str(arg) for arg in payload.get("args", [])],
            reference_returncode=int(payload.get("reference_returncode", 0)),
            candidate_returncode=int(payload.get("candidate_returncode", 0)),
            reference_stdout=str(payload.get("reference_stdout", "")),
            candidate_stdout=str(payload.get("candidate_stdout", "")),
            reference_stderr=str(payload.get("reference_stderr", "")),
            candidate_stderr=str(payload.get("candidate_stderr", "")),
            notes=[str(note) for note in payload.get("notes", [])],
        )


@dataclass
class DiffResult:
    executed: bool
    passed: bool
    passed_cases: int = 0
    failed_cases: int = 0
    mismatch_cases: int = 0
    cases: list[DiffCaseResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiffResult":
        return cls(
            executed=bool(payload.get("executed", False)),
            passed=bool(payload.get("passed", False)),
            passed_cases=int(payload.get("passed_cases", 0)),
            failed_cases=int(payload.get("failed_cases", 0)),
            mismatch_cases=int(payload.get("mismatch_cases", 0)),
            cases=[DiffCaseResult.from_dict(dict(case)) for case in payload.get("cases", [])],
            notes=[str(note) for note in payload.get("notes", [])],
        )


@dataclass
class FuzzCaseResult:
    name: str
    seed_name: str
    mutation_name: str
    input_data: str
    args: list[str] = field(default_factory=list)
    passed: bool = False
    reference_returncode: int = 0
    candidate_returncode: int = 0
    reference_stdout: str = ""
    candidate_stdout: str = ""
    reference_stderr: str = ""
    candidate_stderr: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FuzzCaseResult":
        return cls(
            name=str(payload.get("name", "")),
            seed_name=str(payload.get("seed_name", "")),
            mutation_name=str(payload.get("mutation_name", "")),
            input_data=str(payload.get("input_data", "")),
            args=[str(arg) for arg in payload.get("args", [])],
            passed=bool(payload.get("passed", False)),
            reference_returncode=int(payload.get("reference_returncode", 0)),
            candidate_returncode=int(payload.get("candidate_returncode", 0)),
            reference_stdout=str(payload.get("reference_stdout", "")),
            candidate_stdout=str(payload.get("candidate_stdout", "")),
            reference_stderr=str(payload.get("reference_stderr", "")),
            candidate_stderr=str(payload.get("candidate_stderr", "")),
            notes=[str(note) for note in payload.get("notes", [])],
        )


@dataclass
class FuzzResult:
    executed: bool
    passed: bool
    generated_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    mismatch_cases: int = 0
    crash_cases: int = 0
    diff_error_rate: float = 0.0
    cases: list[FuzzCaseResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FuzzResult":
        return cls(
            executed=bool(payload.get("executed", False)),
            passed=bool(payload.get("passed", False)),
            generated_cases=int(payload.get("generated_cases", 0)),
            passed_cases=int(payload.get("passed_cases", 0)),
            failed_cases=int(payload.get("failed_cases", 0)),
            mismatch_cases=int(payload.get("mismatch_cases", 0)),
            crash_cases=int(payload.get("crash_cases", 0)),
            diff_error_rate=float(payload.get("diff_error_rate", 0.0)),
            cases=[FuzzCaseResult.from_dict(dict(case)) for case in payload.get("cases", [])],
            notes=[str(note) for note in payload.get("notes", [])],
        )


@dataclass
class VerificationSummary:
    sample_id: str
    semantic_threshold: float
    semantic_score: float
    compile: CompileResult
    tests: TestResult
    diff: DiffResult = field(default_factory=lambda: DiffResult(executed=False, passed=True))
    fuzz: FuzzResult = field(default_factory=lambda: FuzzResult(executed=False, passed=True))
    fuzz_diff_error_rate: float = 0.0
    verifier_coverage: float = 0.0
    source_path: str = ""
    executable_path: str = ""
    available_signals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VerificationSummary":
        return cls(
            sample_id=str(payload.get("sample_id", "")),
            semantic_threshold=float(payload.get("semantic_threshold", 0.0)),
            semantic_score=float(payload.get("semantic_score", 0.0)),
            compile=CompileResult.from_dict(dict(payload.get("compile", {}))),
            tests=TestResult.from_dict(dict(payload.get("tests", {}))),
            diff=DiffResult.from_dict(dict(payload.get("diff", {}))),
            fuzz=FuzzResult.from_dict(dict(payload.get("fuzz", {}))),
            fuzz_diff_error_rate=float(payload.get("fuzz_diff_error_rate", 0.0)),
            verifier_coverage=float(payload.get("verifier_coverage", 0.0)),
            source_path=str(payload.get("source_path", "")),
            executable_path=str(payload.get("executable_path", "")),
            available_signals=[str(signal) for signal in payload.get("available_signals", [])],
            notes=[str(note) for note in payload.get("notes", [])],
        )
