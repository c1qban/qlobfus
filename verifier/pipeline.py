from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from scripts.common import dump_json, ensure_dir, load_json_if_exists, resolve_path
from verifier.compile import compile_c_source
from verifier.diff import run_differential_cases
from verifier.fuzz import run_fuzz_campaign
from verifier.manifest import extract_manifest_test_cases, find_manifest_sample, load_manifest
from .models import CompileResult, DiffResult, FuzzResult, TestCase, TestCaseResult, TestResult, VerificationSummary
from verifier.tests import coerce_test_cases, run_test_cases


SEMANTIC_SIGNAL_WEIGHTS = {
    "compile": 0.35,
    "tests": 0.30,
    "fuzz": 0.20,
    "symbolic": 0.10,
    "functional_similarity": 0.05,
}


class VerificationPipeline:
    """First-stage verifier with real compile + test execution hooks."""

    def __init__(
        self,
        semantic_threshold: float = 0.97,
        compiler: str | None = None,
        compiler_flags: list[str] | None = None,
        compile_timeout_sec: float = 10.0,
        build_root: Path | None = None,
        cache_root: Path | None = None,
        cache_enabled: bool = True,
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.compiler = compiler
        self.compiler_flags = list(compiler_flags or [])
        self.compile_timeout_sec = compile_timeout_sec
        self.build_root = resolve_path(build_root or "artifacts/verifier")
        self.cache_root = resolve_path(cache_root) if cache_root is not None else self.build_root / "cache"
        self.cache_enabled = cache_enabled

    def _score_semantics(
        self,
        *,
        compile_succeeded: bool,
        test_result: TestResult,
        has_tests: bool,
        diff_result: DiffResult,
        fuzz_result: FuzzResult,
    ) -> tuple[float, float, list[str]]:
        signals: dict[str, float] = {"compile": 1.0 if compile_succeeded else 0.0}
        available_signals = ["compile"]

        if has_tests:
            total_cases = test_result.passed_cases + test_result.failed_cases
            test_score = 1.0 if total_cases == 0 else test_result.passed_cases / total_cases
            signals["tests"] = test_score
            available_signals.append("tests")

        if diff_result.executed:
            total_cases = diff_result.passed_cases + diff_result.failed_cases
            diff_score = 1.0 if total_cases == 0 else diff_result.passed_cases / total_cases
            signals["functional_similarity"] = diff_score
            available_signals.append("functional_similarity")

        if fuzz_result.executed:
            signals["fuzz"] = 1.0 - float(min(max(fuzz_result.diff_error_rate, 0.0), 1.0))
            available_signals.append("fuzz")

        total_weight = sum(SEMANTIC_SIGNAL_WEIGHTS[name] for name in available_signals)
        weighted_score = sum(SEMANTIC_SIGNAL_WEIGHTS[name] * signals[name] for name in available_signals)
        semantic_score = weighted_score / total_weight if total_weight else 0.0
        coverage = total_weight / sum(SEMANTIC_SIGNAL_WEIGHTS.values())
        return semantic_score, coverage, available_signals

    def _reconcile_tests_with_diff(self, test_result: TestResult, diff_result: DiffResult) -> TestResult:
        if not diff_result.executed or not diff_result.cases or not test_result.cases:
            return test_result

        passed_diff_cases = {case.name for case in diff_result.cases if case.passed}
        reconciled_cases: list[TestCaseResult] = []
        changed = False
        for case in test_result.cases:
            if not case.passed and case.name in passed_diff_cases:
                changed = True
                reconciled_cases.append(
                    TestCaseResult(
                        name=case.name,
                        passed=True,
                        returncode=case.returncode,
                        stdout=case.stdout,
                        stderr=case.stderr,
                        expected_stdout=case.expected_stdout,
                        expected_stderr=case.expected_stderr,
                        expected_returncode=case.expected_returncode,
                        notes=[
                            *case.notes,
                            "credited by differential execution because candidate behavior matched the reference source",
                        ],
                    )
                )
            else:
                reconciled_cases.append(case)

        if not changed:
            return test_result

        passed_cases = sum(1 for case in reconciled_cases if case.passed)
        failed_cases = len(reconciled_cases) - passed_cases
        return TestResult(
            passed=failed_cases == 0,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            cases=reconciled_cases,
            notes=[
                *test_result.notes,
                "Some fixed-output harness failures were credited because differential execution matched the reference source.",
            ],
        )

    def _build_dir(self, sample_id: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        build_dir = self.build_root / f"{sample_id}_{stamp}"
        ensure_dir(build_dir)
        return build_dir

    def _serialize_test_cases(self, test_cases: list[TestCase]) -> list[dict[str, object]]:
        return [
            {
                "name": case.name,
                "input_data": case.input_data,
                "args": list(case.args),
                "expected_stdout": case.expected_stdout,
                "expected_stderr": case.expected_stderr,
                "expected_returncode": case.expected_returncode,
                "timeout_sec": case.timeout_sec,
                "strip_output": case.strip_output,
                "normalize_newlines": case.normalize_newlines,
                "collapse_whitespace": case.collapse_whitespace,
            }
            for case in test_cases
        ]

    def _build_cache_key(
        self,
        *,
        source_path: Path,
        reference_source_path: Path | None,
        test_cases: list[TestCase],
        compiler: str | None,
        compiler_flags: list[str] | None,
        compile_timeout_sec: float | None,
        fuzz_max_cases: int,
        fuzz_enabled: bool,
    ) -> str:
        source_text = source_path.read_text(encoding="utf-8")
        reference_text = reference_source_path.read_text(encoding="utf-8") if reference_source_path is not None else ""
        payload = {
            "version": "verification_cache_v7_reference_invalid_fuzz_skip",
            "semantic_threshold": self.semantic_threshold,
            "source_text": source_text,
            "reference_text": reference_text,
            "compiler": compiler or self.compiler or "",
            "compiler_flags": list(compiler_flags or self.compiler_flags),
            "compile_timeout_sec": compile_timeout_sec if compile_timeout_sec is not None else self.compile_timeout_sec,
            "fuzz_enabled": fuzz_enabled,
            "fuzz_max_cases": fuzz_max_cases,
            "test_cases": self._serialize_test_cases(test_cases),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        ensure_dir(self.cache_root)
        return self.cache_root / f"{cache_key}.json"

    def _load_cached_summary(self, cache_key: str) -> VerificationSummary | None:
        if not self.cache_enabled:
            return None
        cache_payload = load_json_if_exists(self._cache_path(cache_key))
        if not cache_payload:
            return None
        summary = VerificationSummary.from_dict(cache_payload)
        summary.notes = ["Loaded verification result from cache.", *summary.notes]
        return summary

    def _store_cached_summary(self, cache_key: str, summary: VerificationSummary) -> None:
        if not self.cache_enabled:
            return
        dump_json(self._cache_path(cache_key), summary.to_dict())

    def verify_source(
        self,
        source_path: str | Path,
        test_cases: list[dict[str, object]] | list[object] | None = None,
        sample_id: str | None = None,
        compiler: str | None = None,
        compiler_flags: list[str] | None = None,
        compile_timeout_sec: float | None = None,
        reference_source_path: str | Path | None = None,
        build_dir: str | Path | None = None,
        workdir: str | Path | None = None,
        fuzz_max_cases: int = 16,
        fuzz_enabled: bool = True,
    ) -> VerificationSummary:
        resolved_source = resolve_path(source_path)
        resolved_reference_source = resolve_path(reference_source_path) if reference_source_path is not None else None
        coerced_cases = coerce_test_cases(test_cases)
        cache_key = self._build_cache_key(
            source_path=resolved_source,
            reference_source_path=resolved_reference_source,
            test_cases=coerced_cases,
            compiler=compiler,
            compiler_flags=compiler_flags,
            compile_timeout_sec=compile_timeout_sec,
            fuzz_max_cases=fuzz_max_cases,
            fuzz_enabled=fuzz_enabled,
        )
        cached_summary = self._load_cached_summary(cache_key)
        if cached_summary is not None:
            return cached_summary

        sample_name = sample_id or resolved_source.stem
        resolved_build_dir = resolve_path(build_dir) if build_dir else self._build_dir(sample_name)
        ensure_dir(resolved_build_dir)
        resolved_workdir = resolve_path(workdir) if workdir else resolved_source.parent
        executable_name = "candidate" + (".exe" if os.name == "nt" else "")
        executable_path = resolved_build_dir / executable_name

        compile_result = compile_c_source(
            source_path=resolved_source,
            output_path=executable_path,
            compiler=compiler or self.compiler,
            compiler_flags=compiler_flags or self.compiler_flags,
            timeout_sec=compile_timeout_sec if compile_timeout_sec is not None else self.compile_timeout_sec,
            workdir=resolved_workdir,
        )

        diff_result = DiffResult(executed=False, passed=True)
        fuzz_result = FuzzResult(executed=False, passed=True)
        if compile_result.succeeded:
            test_result = run_test_cases(executable_path, coerced_cases, workdir=resolved_workdir)
        else:
            test_result = TestResult(
                passed=False,
                passed_cases=0,
                failed_cases=len(coerced_cases),
                cases=[],
                notes=["Compilation failed, so no test cases were executed."],
            )

        if compile_result.succeeded and resolved_reference_source is not None:
            reference_executable_name = "reference" + (".exe" if os.name == "nt" else "")
            reference_executable_path = resolved_build_dir / reference_executable_name
            reference_compile_result = compile_c_source(
                source_path=resolved_reference_source,
                output_path=reference_executable_path,
                compiler=compiler or self.compiler,
                compiler_flags=compiler_flags or self.compiler_flags,
                timeout_sec=compile_timeout_sec if compile_timeout_sec is not None else self.compile_timeout_sec,
                workdir=resolved_workdir,
            )
            if reference_compile_result.succeeded:
                diff_result = run_differential_cases(
                    reference_executable_path,
                    executable_path,
                    coerced_cases,
                    workdir=resolved_workdir,
                )
                if fuzz_enabled:
                    fuzz_result = run_fuzz_campaign(
                        reference_executable_path,
                        executable_path,
                        coerced_cases,
                        workdir=resolved_workdir,
                        max_cases=fuzz_max_cases,
                    )
                else:
                    fuzz_result = FuzzResult(
                        executed=False,
                        passed=True,
                        notes=["Fuzzing was disabled by verifier configuration."],
                    )
            else:
                diff_result = DiffResult(
                    executed=False,
                    passed=False,
                    notes=["Reference source failed to compile, so differential execution was skipped."],
                )
                fuzz_result = FuzzResult(
                    executed=False,
                    passed=False,
                    notes=["Reference source failed to compile, so fuzzing was skipped."],
                )

        scored_test_result = self._reconcile_tests_with_diff(test_result, diff_result)
        semantic_score, coverage, available_signals = self._score_semantics(
            compile_succeeded=compile_result.succeeded,
            test_result=scored_test_result,
            has_tests=bool(coerced_cases),
            diff_result=diff_result,
            fuzz_result=fuzz_result,
        )

        notes = [
            "Verification currently includes real compilation and executable test cases.",
        ]
        if not coerced_cases:
            notes.append("No test cases were provided, so semantic scoring only reflects compilation.")
        if resolved_reference_source is not None:
            if diff_result.executed:
                notes.append("Differential execution was run against the reference source.")
            if fuzz_result.executed:
                notes.append("Deterministic stdin fuzzing was run against the reference source.")
        else:
            notes.append("No reference source was provided, so differential execution and fuzzing were skipped.")

        summary = VerificationSummary(
            sample_id=sample_name,
            semantic_threshold=self.semantic_threshold,
            semantic_score=semantic_score,
            compile=compile_result,
            tests=scored_test_result,
            diff=diff_result,
            fuzz=fuzz_result,
            fuzz_diff_error_rate=fuzz_result.diff_error_rate,
            verifier_coverage=coverage,
            source_path=str(resolved_source),
            executable_path=compile_result.output_path,
            available_signals=available_signals,
            notes=notes,
        )
        self._store_cached_summary(cache_key, summary)
        return summary

    def run_stub(self, sample_id: str = "placeholder") -> VerificationSummary:
        return VerificationSummary(
            sample_id=sample_id,
            semantic_threshold=self.semantic_threshold,
            semantic_score=self.semantic_threshold,
            compile=CompileResult(
                succeeded=True,
                command="clang <stub>",
                notes=["Replace with a real compile command once verifier/compile is implemented."],
            ),
            tests=TestResult(
                passed=True,
                passed_cases=1,
                failed_cases=0,
                cases=[],
                notes=["Replace with real harness results once verifier/tests is implemented."],
            ),
            diff=DiffResult(
                executed=False,
                passed=True,
                notes=["Replace with real differential execution once a reference program is available."],
            ),
            fuzz=FuzzResult(
                executed=False,
                passed=True,
                notes=["Replace with real fuzzing once seed inputs and runners are available."],
            ),
            fuzz_diff_error_rate=0.0,
            verifier_coverage=1.0,
            available_signals=["compile", "tests", "fuzz", "symbolic", "functional_similarity"],
            notes=[
                "This is a placeholder semantic verification result.",
                "Wire real compile, test, fuzz, and symbolic stages into verifier/.",
            ],
        )

    def verify_manifest_sample(
        self,
        manifest_path: str | Path,
        sample_id: str,
    ) -> VerificationSummary:
        manifest = load_manifest(str(manifest_path))
        sample = find_manifest_sample(manifest, sample_id)
        default_compile = manifest.get("verification_defaults", {}).get("compile", {})
        sample_compile = sample.get("compile", {})
        compile_config = {
            "compiler": sample_compile.get("compiler", default_compile.get("compiler")),
            "compiler_flags": list(sample_compile.get("compiler_flags", default_compile.get("compiler_flags", []))),
            "timeout_sec": float(sample_compile.get("timeout_sec", default_compile.get("timeout_sec", 10.0))),
        }
        test_cases = extract_manifest_test_cases(sample)

        return self.verify_source(
            source_path=str(sample["source_path"]),
            test_cases=test_cases,
            sample_id=str(sample.get("sample_id", sample_id)),
            compiler=compile_config["compiler"],
            compiler_flags=compile_config["compiler_flags"],
            compile_timeout_sec=compile_config["timeout_sec"],
        )
