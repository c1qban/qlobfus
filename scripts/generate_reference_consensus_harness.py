from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file, project_relative, resolve_path
from verifier.compile import compile_c_source


SEED_INPUTS = [
    "",
    "0\n",
    "1\n",
    "2\n",
    "3\n",
    "10\n",
    "1 1\n",
    "1 2\n",
    "2 3\n",
    "3 4\n",
    "8 6\n",
    "7 13\n",
    "1 2 3\n",
    "1 2 3 4\n",
    "1 2 3 4 5\n",
    "1 2 3 4 5 6\n",
    "2 -1 -2 -1 -1 -5\n",
    "1 2\n3 4\n",
    "1 2 3\n4 5 6\n",
    "8 6\n50000000 30000000\n",
    "1 2 3 4 5 6\n2 -1 -2 -1 -1 -5\n",
    "10\n20\n30\n",
    "100\n1000\n10000\n",
    "10,90\n0,0\n",
    "10,0\n10,90\n10,90\n0,0\n",
    "the quick brown fox\n",
    "vjg swkem dtqyp hqz\n",
    "1.0\n2.5\n-1.0\n",
    "3.14\n2.71\n1.41\n",
    "0 0 1 1 0 1 1 2\n",
    "0 0 1 0 0 1 1 1\n",
    "2000 2020\n0 0\n",
    "1999 1999\n0 0\n",
    "2001 2003\n0 0\n",
    "1 2 3 4 5 6 7 8\n",
    "1 2 3 4 5 6 7 8 9 10 11 12\n",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20\n",
    "2 2\n00\n11\n",
    "3 3\n010\n101\n010\n",
    "4 4\n0101\n1010\n0101\n1010\n",
    "5 5\n01010\n10101\n01010\n10101\n01010\n",
]

PROBLEM_SEED_INPUTS = {
    "p00041": ["1 2 3 4\n0 0 0 0\n"],
    "p00092": ["1\n.\n0\n", "2\n..\n..\n0\n"],
    "p00177": ["0 0 0 0\n-1 -1 -1 -1\n"],
    "p00189": ["1\n0 1 5\n0\n"],
    "p00202": ["1 10\n3\n0 0\n"],
    "p00238": ["1\n1\n0 1\n0\n"],
    "p00240": ["1\n1\n1 1 1\n0\n"],
    "p00255": ["2\n1 2\n3\n0\n"],
    "p00511": ["1\n=\n", "2\n+ 3\n=\n"],
    "p00622": ["a\nb\nab\n-\n"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stdio harness files from accepted-reference consensus.")
    parser.add_argument("--processed-root", required=True, help="Processed dataset root containing split indexes.")
    parser.add_argument("--harness-root", required=True, help="Output harness root.")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], help="Splits to generate harness for.")
    parser.add_argument("--compiler", default=r"C:\Program Files\LLVM\bin\clang.exe", help="Compiler used for reference execution.")
    parser.add_argument("--max-cases-per-problem", type=int, default=3, help="Maximum consensus cases to keep per problem.")
    parser.add_argument("--min-reference-agreement", type=int, default=2, help="Minimum compiled references that must agree.")
    parser.add_argument("--timeout-sec", type=float, default=2.0, help="Compile/run timeout.")
    parser.add_argument("--progress", action="store_true", help="Print progress with elapsed time and ETA.")
    parser.add_argument("--max-problems", type=int, default=0, help="Process at most this many candidate problems; 0 means no limit.")
    parser.add_argument(
        "--target-new-samples",
        type=int,
        default=0,
        help="Stop after writing this many new harness files; 0 means no early stop.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite existing harness files.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only process problems that still have at least one missing harness file under harness-root.",
    )
    parser.add_argument(
        "--per-sample-fallback",
        action="store_true",
        help="Generate per-sample self-reference harnesses when problem-level consensus is unavailable.",
    )
    parser.add_argument(
        "--strict-seed-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter seed inputs that are obviously too weak for the source's first scanf.",
    )
    parser.add_argument(
        "--harness-audit",
        action="append",
        default=[],
        help="Optional split=path audit. Non-ok samples are treated as needing regeneration even if a harness file exists.",
    )
    parser.add_argument(
        "--audit-failures-only",
        action="store_true",
        help="Only process samples marked non-ok by --harness-audit; useful for targeted harness repair.",
    )
    parser.add_argument(
        "--existing-harness-only",
        action="store_true",
        help="Only process samples whose expected harness file already exists.",
    )
    return parser.parse_args(argv)


def load_split_samples(processed_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for split_name in splits:
        path = processed_root / "splits" / f"{split_name}.index.json"
        payload = load_structured_file(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Split index must decode to an object: {path}")
        samples.extend(dict(sample) for sample in payload.get("samples", []))
    return samples


def expected_harness_path(harness_root: Path, sample: dict[str, Any]) -> Path:
    return harness_root / Path(str(sample["relative_source_path"])).with_suffix(".tests.json")


def parse_harness_audits(values: list[str]) -> set[str]:
    regenerate: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --harness-audit split=path, got: {value}")
        _split, path_text = value.split("=", 1)
        payload = load_structured_file(resolve_path(path_text, PROJECT_ROOT))
        if not isinstance(payload, dict):
            raise ValueError(f"Harness audit must decode to an object: {path_text}")
        for item in payload.get("samples", []):
            if isinstance(item, dict) and item.get("sample_id") and str(item.get("status", "")) != "ok":
                regenerate.add(str(item["sample_id"]))
    return regenerate


def _input_tokens(input_data: str) -> list[str]:
    return re.findall(r"[^\s,]+", input_data)


def _first_scanf_conversion_count(source_text: str) -> int:
    match = re.search(r"\b(?:scanf|fscanf)\s*\(\s*\"((?:\\.|[^\"\\])*)\"", source_text)
    if not match:
        return 0
    fmt = match.group(1)
    return sum(
        1
        for spec in re.finditer(r"%(?!%)(\*)?(?:\d+)?(?:hh|h|ll|l|L|z|j|t)?[diuoxXfFeEgGaAcspn\[]", fmt)
        if spec.group(1) != "*"
    )


def _source_reads_string_rows(source_text: str) -> bool:
    return bool(re.search(r'\b(?:scanf|fscanf)\s*\(\s*"%s"', source_text))


def _looks_like_binary_grid(input_data: str) -> tuple[bool, int, int]:
    lines = [line.strip() for line in input_data.splitlines() if line.strip()]
    binary_rows = [line for line in lines if re.fullmatch(r"[01]+", line)]
    if not binary_rows:
        return False, 0, 0
    widths = {len(line) for line in binary_rows}
    return len(binary_rows) == len(lines) and len(widths) == 1, len(binary_rows), next(iter(widths))


def usable_seed_inputs_for_source(
    source_path: Path,
    *,
    strict: bool,
    problem_id: str = "",
) -> tuple[list[str], dict[str, Any]]:
    seed_inputs = list(dict.fromkeys([*PROBLEM_SEED_INPUTS.get(problem_id, []), *SEED_INPUTS]))
    if not strict:
        return seed_inputs, {
            "strict_seed_inputs": False,
            "problem_seed_count": len(PROBLEM_SEED_INPUTS.get(problem_id, [])),
            "filtered_seed_count": 0,
        }
    source_text = source_path.read_text(encoding="utf-8", errors="replace") if source_path.exists() else ""
    first_scanf_count = _first_scanf_conversion_count(source_text)
    reads_string_rows = _source_reads_string_rows(source_text)
    usable: list[str] = []
    filtered: list[dict[str, Any]] = []
    for index, seed_input in enumerate(seed_inputs):
        token_count = len(_input_tokens(seed_input))
        if first_scanf_count and token_count < first_scanf_count:
            filtered.append({"seed_index": index, "reason": "too_few_tokens_for_first_scanf", "token_count": token_count})
            continue
        if reads_string_rows:
            is_grid, rows, width = _looks_like_binary_grid(seed_input)
            if is_grid and rows < 2:
                filtered.append({"seed_index": index, "reason": "binary_grid_too_small", "rows": rows, "width": width})
                continue
        usable.append(seed_input)
    return usable, {
        "strict_seed_inputs": True,
        "problem_seed_count": len(PROBLEM_SEED_INPUTS.get(problem_id, [])),
        "first_scanf_conversion_count": first_scanf_count,
        "source_reads_string_rows": reads_string_rows,
        "filtered_seed_count": len(filtered),
        "filtered_preview": filtered[:20],
    }


def compile_reference(source_path: Path, build_root: Path, compiler: str, timeout_sec: float) -> Path | None:
    output_path = build_root / (source_path.stem + ".exe")
    result = compile_c_source(
        source_path=source_path,
        output_path=output_path,
        compiler=compiler,
        compiler_flags=["-O0"],
        timeout_sec=timeout_sec,
        workdir=source_path.parent,
    )
    return output_path if result.succeeded else None


def run_reference(executable_path: Path, input_data: str, timeout_sec: float) -> tuple[int, str, str] | None:
    try:
        completed = subprocess.run(
            [str(executable_path)],
            input=input_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    return completed.returncode, completed.stdout, completed.stderr


def consensus_cases_for_problem(
    *,
    problem_id: str,
    samples: list[dict[str, Any]],
    build_root: Path,
    compiler: str,
    max_cases: int,
    min_reference_agreement: int,
    timeout_sec: float,
    strict_seed_inputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compiled: list[Path] = []
    seed_inputs: list[str] = []
    seed_audits: list[dict[str, Any]] = []
    for sample in samples:
        source_path = resolve_path(str(sample["source_path"]), PROJECT_ROOT)
        sample_seed_inputs, sample_seed_audit = usable_seed_inputs_for_source(
            source_path,
            strict=strict_seed_inputs,
            problem_id=problem_id,
        )
        seed_audits.append({"sample_id": str(sample.get("sample_id", "")), **sample_seed_audit})
        if not seed_inputs:
            seed_inputs = sample_seed_inputs
        else:
            allowed = set(sample_seed_inputs)
            seed_inputs = [seed_input for seed_input in seed_inputs if seed_input in allowed]
        executable = compile_reference(
            source_path=source_path,
            build_root=ensure_dir(build_root / problem_id / source_path.stem),
            compiler=compiler,
            timeout_sec=timeout_sec,
        )
        if executable is not None:
            compiled.append(executable)

    cases: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for seed_index, seed_input in enumerate(seed_inputs):
        if len(cases) >= max_cases:
            break
        outputs: list[tuple[int, str, str]] = []
        for executable in compiled:
            result = run_reference(executable, seed_input, timeout_sec)
            if result is not None:
                outputs.append(result)
        if len(outputs) < min_reference_agreement:
            rejected.append({"seed_index": seed_index, "reason": "too_few_completed_references"})
            continue
        first = outputs[0]
        if first[0] != 0:
            rejected.append({"seed_index": seed_index, "reason": "nonzero_reference_returncode"})
            continue
        if not first[1] and not first[2]:
            rejected.append({"seed_index": seed_index, "reason": "empty_output"})
            continue
        if any(output != first for output in outputs[1:]):
            rejected.append({"seed_index": seed_index, "reason": "reference_outputs_disagree"})
            continue
        cases.append(
            {
                "name": f"consensus_{len(cases) + 1}",
                "input_data": seed_input,
                "expected_stdout": first[1],
                "expected_stderr": first[2],
                "expected_returncode": first[0],
                "timeout_sec": timeout_sec,
            }
        )

    audit = {
        "problem_id": problem_id,
        "sample_count": len(samples),
        "compiled_reference_count": len(compiled),
        "case_count": len(cases),
        "strict_seed_inputs": strict_seed_inputs,
        "usable_seed_count": len(seed_inputs),
        "seed_audits": seed_audits[:20],
        "rejected_preview": rejected[:20],
    }
    return cases, audit


def write_harnesses(
    harness_root: Path,
    samples: list[dict[str, Any]],
    cases_by_problem: dict[str, list[dict[str, Any]]],
    *,
    skip_existing: bool = False,
    target_remaining: int = 0,
    force_sample_ids: set[str] | None = None,
) -> int:
    written = 0
    force_sample_ids = force_sample_ids or set()
    for sample in samples:
        problem_id = str(sample.get("split_group", ""))
        cases = cases_by_problem.get(problem_id, [])
        if not cases:
            continue
        output_path = expected_harness_path(harness_root, sample)
        sample_id = str(sample.get("sample_id", ""))
        if skip_existing and output_path.exists() and sample_id not in force_sample_ids:
            continue
        if target_remaining > 0 and written >= target_remaining:
            break
        dump_json(
            output_path,
            {
                "format": "executable_stdio_v1",
                "test_cases": cases,
            },
        )
        written += 1
    return written


def fallback_cases_for_sample(
    *,
    sample: dict[str, Any],
    build_root: Path,
    compiler: str,
    max_cases: int,
    timeout_sec: float,
    strict_seed_inputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = resolve_path(str(sample["source_path"]), PROJECT_ROOT)
    seed_inputs, seed_audit = usable_seed_inputs_for_source(
        source_path,
        strict=strict_seed_inputs,
        problem_id=str(sample.get("split_group", "")),
    )
    executable = compile_reference(
        source_path=source_path,
        build_root=ensure_dir(build_root / "fallback" / source_path.stem),
        compiler=compiler,
        timeout_sec=timeout_sec,
    )
    if executable is None:
        return [], {
            "sample_id": str(sample.get("sample_id", "")),
            "source_path": project_relative(source_path),
            "case_count": 0,
            "reason": "compile_failed",
            "seed_audit": seed_audit,
        }

    cases: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for seed_index, seed_input in enumerate(seed_inputs):
        if len(cases) >= max_cases:
            break
        result = run_reference(executable, seed_input, timeout_sec)
        if result is None:
            rejected.append({"seed_index": seed_index, "reason": "timeout"})
            continue
        returncode, stdout, stderr = result
        if returncode != 0:
            rejected.append({"seed_index": seed_index, "reason": "nonzero_returncode"})
            continue
        if not stdout and not stderr:
            rejected.append({"seed_index": seed_index, "reason": "empty_output"})
            continue
        cases.append(
            {
                "name": f"self_reference_{len(cases) + 1}",
                "input_data": seed_input,
                "expected_stdout": stdout,
                "expected_stderr": stderr,
                "expected_returncode": returncode,
                "timeout_sec": timeout_sec,
            }
        )

    return cases, {
        "sample_id": str(sample.get("sample_id", "")),
        "source_path": project_relative(source_path),
        "case_count": len(cases),
        "seed_audit": seed_audit,
        "rejected_preview": rejected[:20],
        "reason": "" if cases else "no_usable_seed",
    }


def write_per_sample_fallback_harnesses(
    *,
    harness_root: Path,
    samples: list[dict[str, Any]],
    cases_by_problem: dict[str, list[dict[str, Any]]],
    build_root: Path,
    compiler: str,
    max_cases: int,
    timeout_sec: float,
    skip_existing: bool = False,
    target_remaining: int = 0,
    strict_seed_inputs: bool = True,
    force_sample_ids: set[str] | None = None,
    progress: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    written = 0
    audits: list[dict[str, Any]] = []
    force_sample_ids = force_sample_ids or set()
    candidates: list[dict[str, Any]] = []
    for sample in samples:
        if target_remaining > 0 and written >= target_remaining:
            break
        problem_id = str(sample.get("split_group", ""))
        if cases_by_problem.get(problem_id):
            continue
        output_path = expected_harness_path(harness_root, sample)
        sample_id = str(sample.get("sample_id", ""))
        if skip_existing and output_path.exists() and sample_id not in force_sample_ids:
            continue
        candidates.append(sample)

    start_time = time.time()
    total = len(candidates)
    for index, sample in enumerate(candidates, start=1):
        if target_remaining > 0 and written >= target_remaining:
            break
        sample_id = str(sample.get("sample_id", ""))
        output_path = expected_harness_path(harness_root, sample)
        if progress:
            elapsed = time.time() - start_time
            rate = max(index - 1, 0) / elapsed if elapsed > 0 else 0.0
            remaining = (total - index + 1) / rate if rate > 0 else 0.0
            print(
                f"\r[fallback] {index}/{total} sample={sample_id} "
                f"written={written} elapsed={format_duration(elapsed)} eta={format_duration(remaining)}",
                end="",
                flush=True,
            )
        cases, audit = fallback_cases_for_sample(
            sample=sample,
            build_root=build_root,
            compiler=compiler,
            max_cases=max_cases,
            timeout_sec=timeout_sec,
            strict_seed_inputs=strict_seed_inputs,
        )
        audits.append(audit)
        if not cases:
            continue
        dump_json(
            output_path,
            {
                "format": "executable_stdio_v1",
                "test_cases": cases,
                "notes": ["Generated from this sample's own accepted reference output."],
            },
        )
        written += 1
    if progress and total:
        print("", flush=True)
    return written, audits


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def print_progress(
    *,
    index: int,
    total: int,
    problem_id: str,
    start_time: float,
    harnessed_problem_count: int,
    harnessed_sample_count: int,
) -> None:
    elapsed = time.time() - start_time
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    width = 28
    filled = int(width * index / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    message = (
        f"\r[{bar}] {index}/{total} problems "
        f"current={problem_id} "
        f"harnessed_problems={harnessed_problem_count} "
        f"harnessed_samples={harnessed_sample_count} "
        f"elapsed={format_duration(elapsed)} "
        f"eta={format_duration(remaining)}"
    )
    print(message, end="", flush=True)
    if index >= total:
        print("", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    processed_root = resolve_path(args.processed_root, PROJECT_ROOT)
    harness_root = resolve_path(args.harness_root, PROJECT_ROOT)
    build_root = ensure_dir(PROJECT_ROOT / "artifacts" / "reference_consensus_harness_build")
    samples = load_split_samples(processed_root, list(args.splits))
    force_sample_ids = parse_harness_audits(list(args.harness_audit))
    if args.audit_failures_only:
        if not force_sample_ids:
            raise ValueError("--audit-failures-only requires at least one --harness-audit with non-ok samples.")
        samples = [sample for sample in samples if str(sample.get("sample_id", "")) in force_sample_ids]
    if args.existing_harness_only:
        samples = [sample for sample in samples if expected_harness_path(harness_root, sample).exists()]

    samples_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_problem[str(sample.get("split_group", ""))].append(sample)

    cases_by_problem: dict[str, list[dict[str, Any]]] = {}
    audits: list[dict[str, Any]] = []
    sorted_problem_items = sorted(samples_by_problem.items())
    start_time = time.time()
    harnessed_problem_count = 0
    harnessed_sample_count = 0
    written = 0
    skipped_existing_problem_count = 0
    if args.only_missing:
        filtered_problem_items = []
        for problem_id, problem_samples in sorted_problem_items:
            has_missing = any(not expected_harness_path(harness_root, sample).exists() for sample in problem_samples)
            has_forced = any(str(sample.get("sample_id", "")) in force_sample_ids for sample in problem_samples)
            if has_missing or has_forced:
                filtered_problem_items.append((problem_id, problem_samples))
            else:
                skipped_existing_problem_count += 1
        sorted_problem_items = filtered_problem_items
    if int(args.max_problems) > 0:
        sorted_problem_items = sorted_problem_items[: int(args.max_problems)]

    total_problems = len(sorted_problem_items)
    target_new_samples = int(args.target_new_samples)
    for index, (problem_id, problem_samples) in enumerate(sorted_problem_items, start=1):
        if target_new_samples > 0 and written >= target_new_samples:
            break
        cases, audit = consensus_cases_for_problem(
            problem_id=problem_id,
            samples=problem_samples,
            build_root=build_root,
            compiler=str(args.compiler),
            max_cases=int(args.max_cases_per_problem),
            min_reference_agreement=int(args.min_reference_agreement),
            timeout_sec=float(args.timeout_sec),
            strict_seed_inputs=bool(args.strict_seed_inputs),
        )
        cases_by_problem[problem_id] = cases
        audits.append(audit)
        if cases:
            harnessed_problem_count += 1
            remaining = max(target_new_samples - written, 0) if target_new_samples > 0 else 0
            newly_written = write_harnesses(
                harness_root,
                problem_samples,
                {problem_id: cases},
                skip_existing=bool(args.skip_existing),
                target_remaining=remaining,
                force_sample_ids=force_sample_ids,
            )
            written += newly_written
            harnessed_sample_count += newly_written if args.skip_existing else len(problem_samples)
        if args.progress:
            print_progress(
                index=index,
                total=total_problems,
                problem_id=problem_id,
                start_time=start_time,
                harnessed_problem_count=harnessed_problem_count,
                harnessed_sample_count=harnessed_sample_count,
            )

    fallback_written = 0
    fallback_audits: list[dict[str, Any]] = []
    if args.per_sample_fallback and (target_new_samples <= 0 or written < target_new_samples):
        remaining = max(target_new_samples - written, 0) if target_new_samples > 0 else 0
        fallback_written, fallback_audits = write_per_sample_fallback_harnesses(
            harness_root=harness_root,
            samples=samples,
            cases_by_problem=cases_by_problem,
            build_root=build_root,
            compiler=str(args.compiler),
            max_cases=int(args.max_cases_per_problem),
            timeout_sec=float(args.timeout_sec),
            skip_existing=bool(args.skip_existing),
            target_remaining=remaining,
            strict_seed_inputs=bool(args.strict_seed_inputs),
            force_sample_ids=force_sample_ids,
            progress=bool(args.progress),
        )
        written += fallback_written
    summary = {
        "processed_root": project_relative(processed_root),
        "harness_root": project_relative(harness_root),
        "splits": list(args.splits),
        "problem_count": len(samples_by_problem),
        "processed_problem_count": len(audits),
        "skipped_existing_problem_count": skipped_existing_problem_count,
        "sample_count": len(samples),
        "harnessed_sample_count": written,
        "harnessed_problem_count": sum(1 for cases in cases_by_problem.values() if cases),
        "fallback_harnessed_sample_count": fallback_written,
        "skip_existing": bool(args.skip_existing),
        "only_missing": bool(args.only_missing),
        "target_new_samples": target_new_samples,
        "max_problems": int(args.max_problems),
        "strict_seed_inputs": bool(args.strict_seed_inputs),
        "forced_regeneration_sample_count": len(force_sample_ids),
        "audit_failures_only": bool(args.audit_failures_only),
        "existing_harness_only": bool(args.existing_harness_only),
        "audits": audits,
        "fallback_audits": fallback_audits,
    }
    summary_path = ensure_dir(harness_root) / "_reference_consensus_summary.json"
    dump_json(summary_path, summary)
    if args.progress:
        sys.stdout.flush()
    print(f"Harnessed samples: {written}/{len(samples)}")
    print(f"Harnessed problems: {summary['harnessed_problem_count']}/{summary['problem_count']}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
