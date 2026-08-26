from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file, project_relative


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Project_codenet_c_05 validation/test stdio harness files.")
    parser.add_argument(
        "--processed-root",
        default="data/processed/project_codenet_c_05_fast",
        help="Processed dataset root containing split index files.",
    )
    parser.add_argument(
        "--harness-root",
        default="data/harness/project_codenet_c_05",
        help="Output harness root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        help="Split names to generate harness files for.",
    )
    return parser.parse_args(argv)


def multiplication_table_output() -> str:
    lines = []
    for left in range(1, 10):
        for right in range(1, 10):
            lines.append(f"{left}x{right}={left * right}")
    return "\n".join(lines) + "\n"


HARNESS_BY_PROBLEM: dict[str, dict[str, Any]] = {
    "p00000": {
        "format": "executable_stdio_v1",
        "test_cases": [
            {
                "name": "multiplication_table",
                "input_data": "",
                "expected_stdout": multiplication_table_output(),
                "expected_returncode": 0,
            }
        ],
    },
    "p00004": {
        "format": "executable_stdio_v1",
        "test_cases": [
            {
                "name": "sample_set_1",
                "input_data": "1 2 3 4 5 6\n2 -1 -2 -1 -1 -5\n",
                "expected_stdout": "-1.000 2.000\n1.000 4.000\n",
                "expected_returncode": 0,
            },
            {
                "name": "zero_rounding_cases",
                "input_data": "2 -1 -3 1 -1 -3\n2 -1 -3 -9 9 27\n",
                "expected_stdout": "0.000 3.000\n0.000 3.000\n",
                "expected_returncode": 0,
            },
        ],
    },
    "p00005": {
        "format": "executable_stdio_v1",
        "test_cases": [
            {
                "name": "sample_cases",
                "input_data": "8 6\n50000000 30000000\n",
                "expected_stdout": "2 24\n10000000 150000000\n",
                "expected_returncode": 0,
            },
            {
                "name": "coprime_and_equal",
                "input_data": "1 1\n7 13\n100 10\n",
                "expected_stdout": "1 1\n1 91\n10 100\n",
                "expected_returncode": 0,
            },
        ],
    },
}


def load_split_samples(processed_root: Path, split_name: str) -> list[dict[str, Any]]:
    index_path = processed_root / "splits" / f"{split_name}.index.json"
    payload = load_structured_file(index_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Split index must decode to an object: {index_path}")
    return [dict(sample) for sample in payload.get("samples", [])]


def harness_path_for_sample(harness_root: Path, relative_source_path: str) -> Path:
    return harness_root / Path(relative_source_path).with_suffix(".tests.json")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    processed_root = (PROJECT_ROOT / args.processed_root).resolve()
    harness_root = (PROJECT_ROOT / args.harness_root).resolve()

    written_paths: list[Path] = []
    skipped: list[dict[str, str]] = []
    split_counts: dict[str, int] = {}
    problem_counts: dict[str, int] = {}

    for split_name in args.splits:
        samples = load_split_samples(processed_root, split_name)
        split_counts[split_name] = 0
        for sample in samples:
            problem_id = str(sample.get("split_group") or sample.get("metadata", {}).get("problem_id", ""))
            harness_payload = HARNESS_BY_PROBLEM.get(problem_id)
            if harness_payload is None:
                skipped.append(
                    {
                        "sample_id": str(sample.get("sample_id", "")),
                        "reason": f"No harness template for problem {problem_id}.",
                    }
                )
                continue

            output_path = harness_path_for_sample(harness_root, str(sample["relative_source_path"]))
            dump_json(output_path, harness_payload)
            written_paths.append(output_path)
            split_counts[split_name] += 1
            problem_counts[problem_id] = problem_counts.get(problem_id, 0) + 1

    summary = {
        "harness_root": project_relative(harness_root),
        "processed_root": project_relative(processed_root),
        "splits": list(args.splits),
        "written_count": len(written_paths),
        "split_counts": split_counts,
        "problem_counts": problem_counts,
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
    }
    summary_path = ensure_dir(harness_root) / "_generation_summary.json"
    dump_json(summary_path, summary)
    print(f"Wrote harness files: {len(written_paths)}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
