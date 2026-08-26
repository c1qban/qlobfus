from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from .common import PROJECT_ROOT, copy_file, dump_json, ensure_dir, load_config, resolve_path, utc_timestamp, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small C-only Project CodeNet subset under data/raw.")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML config file.")
    return parser.parse_args(argv)


def load_problem_rows(problem_list_path: Path) -> dict[str, dict[str, str]]:
    if not problem_list_path.exists():
        return {}

    problem_rows: dict[str, dict[str, str]] = {}
    with problem_list_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {str(key): str(value) for key, value in row.items() if key is not None}
            problem_id = normalized.get("id", "").strip()
            if problem_id:
                problem_rows[problem_id] = normalized
    return problem_rows


def count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())


def normalize_values(items: list[Any]) -> set[str]:
    return {str(item).strip().lower() for item in items}


def build_source_path(
    *,
    data_root: Path,
    problem_id: str,
    submission_id: str,
    language_dir: str,
    filename_ext: str,
) -> Path:
    extension = filename_ext.lstrip(".")
    return data_root / problem_id / language_dir / f"{submission_id}.{extension}"


def select_subset(config: dict[str, Any]) -> dict[str, Any]:
    source_root = resolve_path(config.get("source_root", "data/Project_CodeNet"), PROJECT_ROOT)
    data_root = resolve_path(config.get("data_root", source_root / "data"), PROJECT_ROOT)
    metadata_root = resolve_path(config.get("metadata_root", source_root / "metadata"), PROJECT_ROOT)
    descriptions_root = resolve_path(config.get("problem_descriptions_root", source_root / "problem_descriptions"), PROJECT_ROOT)

    output_root = resolve_path(config.get("output_root", "data/raw/project_codenet_c_subset"), PROJECT_ROOT)
    output_data_root = output_root / "data"
    output_metadata_root = output_root / "metadata"
    output_descriptions_root = output_root / "problem_descriptions"

    problem_count = int(config.get("problem_count", 10))
    samples_per_problem = int(config.get("samples_per_problem", 5))
    min_lines = int(config.get("min_lines", 5))
    max_lines = int(config.get("max_lines", 400))
    accepted_values = normalize_values(list(config.get("accepted_values", ["Accepted", "AC"])))
    language_dir = str(config.get("language_dir", "C"))
    filename_ext = str(config.get("filename_ext", "c"))
    language_values = normalize_values(list(config.get("language_values", [language_dir])))
    copy_problem_descriptions = bool(config.get("copy_problem_descriptions", True))
    clear_output_root = bool(config.get("clear_output_root", True))

    if clear_output_root and output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)

    ensure_dir(output_data_root)
    ensure_dir(output_metadata_root)
    if copy_problem_descriptions:
        ensure_dir(output_descriptions_root)

    problem_rows = load_problem_rows(metadata_root / "problem_list.csv")
    selected_problems: list[dict[str, Any]] = []
    selected_problem_ids: set[str] = set()

    metadata_files = [
        path
        for path in sorted(metadata_root.glob("p*.csv"))
        if path.name.lower() != "problem_list.csv"
    ]

    for metadata_path in metadata_files:
        if len(selected_problems) >= problem_count:
            break

        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                normalized = {str(key): str(value) for key, value in row.items() if key is not None}
                submission_id = normalized.get("submission_id", "").strip()
                problem_id = normalized.get("problem_id", "").strip()
                row_language = normalized.get("language", "").strip().lower()
                verdict = normalized.get("status", "").strip().lower()
                row_ext = normalized.get("filename_ext", "").strip().lower()
                if not submission_id or not problem_id:
                    continue
                if row_language not in language_values:
                    continue
                if verdict not in accepted_values:
                    continue
                if row_ext != filename_ext.lower():
                    continue

                source_path = build_source_path(
                    data_root=data_root,
                    problem_id=problem_id,
                    submission_id=submission_id,
                    language_dir=language_dir,
                    filename_ext=filename_ext,
                )
                if not source_path.exists():
                    continue

                line_count = count_lines(source_path)
                if line_count < min_lines or line_count > max_lines:
                    continue

                rows.append(
                    {
                        "metadata_row": normalized,
                        "source_path": source_path,
                        "line_count": line_count,
                    }
                )

        if len(rows) < samples_per_problem:
            continue

        selected_rows = rows[:samples_per_problem]
        problem_id = str(selected_rows[0]["metadata_row"]["problem_id"])
        selected_problem_ids.add(problem_id)
        selected_problems.append(
            {
                "problem_id": problem_id,
                "problem_row": problem_rows.get(problem_id, {"id": problem_id}),
                "selected_rows": selected_rows,
                "metadata_path": metadata_path,
            }
        )

    copied_samples: list[dict[str, Any]] = []
    for problem_entry in selected_problems:
        problem_id = str(problem_entry["problem_id"])
        output_problem_root = ensure_dir(output_data_root / problem_id / language_dir)
        metadata_output_path = output_metadata_root / f"{problem_id}.csv"
        fieldnames: list[str] = []
        metadata_rows: list[dict[str, str]] = []
        for selected_row in problem_entry["selected_rows"]:
            row = dict(selected_row["metadata_row"])
            fieldnames = fieldnames or list(row.keys())
            metadata_rows.append(row)
            destination = output_problem_root / selected_row["source_path"].name
            copy_file(selected_row["source_path"], destination)
            copied_samples.append(
                {
                    "problem_id": problem_id,
                    "submission_id": row["submission_id"],
                    "source_path": str(destination.relative_to(output_root).as_posix()),
                    "line_count": int(selected_row["line_count"]),
                    "status": row.get("status", ""),
                }
            )

        with metadata_output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)

        if copy_problem_descriptions:
            description_source = descriptions_root / f"{problem_id}.html"
            if description_source.exists():
                copy_file(description_source, output_descriptions_root / description_source.name)

    if selected_problem_ids:
        problem_list_output_path = output_metadata_root / "problem_list.csv"
        fieldnames = []
        rows_to_write: list[dict[str, str]] = []
        if problem_rows:
            first_row = next(iter(problem_rows.values()))
            fieldnames = list(first_row.keys())
            rows_to_write = [problem_rows[problem_id] for problem_id in sorted(selected_problem_ids) if problem_id in problem_rows]
        else:
            fieldnames = ["id"]
            rows_to_write = [{"id": problem_id} for problem_id in sorted(selected_problem_ids)]

        with problem_list_output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_write)

    readme_lines = [
        "# Project CodeNet C Subset",
        "",
        f"Created at: {utc_timestamp()}",
        f"Source root: {source_root}",
        f"Problems requested: {problem_count}",
        f"Problems selected: {len(selected_problems)}",
        f"Samples per problem requested: {samples_per_problem}",
        f"Samples copied: {len(copied_samples)}",
        f"Language: {language_dir}",
        f"Accepted values: {sorted(accepted_values)}",
        f"Line count filter: [{min_lines}, {max_lines}]",
        "",
        "Selected problems:",
    ]
    readme_lines.extend(f"- {problem_entry['problem_id']}" for problem_entry in selected_problems)
    write_text(output_root / "README.md", "\n".join(readme_lines) + "\n")

    summary = {
        "created_at": utc_timestamp(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "problem_count_requested": problem_count,
        "problem_count_selected": len(selected_problems),
        "samples_per_problem_requested": samples_per_problem,
        "sample_count_copied": len(copied_samples),
        "language_dir": language_dir,
        "accepted_values": sorted(accepted_values),
        "min_lines": min_lines,
        "max_lines": max_lines,
        "problems": [
            {
                "problem_id": entry["problem_id"],
                "sample_count": len(entry["selected_rows"]),
                "submissions": [
                    {
                        "submission_id": row["metadata_row"]["submission_id"],
                        "line_count": row["line_count"],
                    }
                    for row in entry["selected_rows"]
                ],
            }
            for entry in selected_problems
        ],
        "samples": copied_samples,
    }
    dump_json(output_root / "subset_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, config = load_config(args.config)
    summary = select_subset(config)
    print(
        "Created Project CodeNet subset: "
        f"{summary['problem_count_selected']} problems, {summary['sample_count_copied']} samples."
    )


if __name__ == "__main__":
    main()
