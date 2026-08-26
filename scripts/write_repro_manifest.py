from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import dump_json, ensure_dir, project_relative, resolve_path, utc_timestamp, write_text


DEFAULT_OUTPUT_DIR = "artifacts/reproducibility"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest for paper experiments.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _command_version(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0]) or command[0]
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
        output = (completed.stdout or completed.stderr).replace("\x00", "")
        first_line = output.splitlines()[0] if output else ""
        return {"command": " ".join(command), "resolved": executable, "returncode": completed.returncode, "version": first_line}
    except Exception as exc:
        return {"command": " ".join(command), "resolved": executable, "error": str(exc)}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _existing_paths(paths: list[str]) -> dict[str, bool]:
    return {path: resolve_path(path).exists() for path in paths}


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reproducibility Manifest",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- python: `{payload['python']['version']}`",
        f"- platform: `{payload['platform']['platform']}`",
        "",
        "## Key Commands",
        "",
    ]
    for name, command in payload["paper_commands"].items():
        lines.append(f"- `{name}`: `{command}`")
    lines.extend(["", "## Toolchain", ""])
    for item in payload["toolchain"]:
        version = item.get("version", item.get("error", ""))
        lines.append(f"- `{item['command']}` -> `{version}`")
    lines.extend(["", "## Python Packages", ""])
    for name, version in payload["python_packages"].items():
        lines.append(f"- `{name}`: `{version}`")
    lines.extend(["", "## Experiment Inputs", ""])
    for path, exists in payload["paths"].items():
        lines.append(f"- `{path}`: `{exists}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    paths = [
        "data/processed/project_codenet_c_500x4_strict/protocol_v5/train_harness_clean.index.json",
        "data/processed/project_codenet_c_500x4_strict/protocol_v5/validation_harness_clean.index.json",
        "data/processed/project_codenet_c_500x4_strict/protocol_v5/test_harness_clean.index.json",
        "data/processed/project_codenet_c_500x4_strict/protocol_v5/protocol.report.json",
        "data/manifests/project_codenet_c_500x4_strict.manifest.json",
        "artifacts/paper_figures/README.md",
    ]
    payload = {
        "generated_at": utc_timestamp(),
        "python": {"version": sys.version.replace("\n", " "), "executable": sys.executable},
        "platform": {"platform": platform.platform(), "machine": platform.machine()},
        "toolchain": [
            _command_version(["clang", "--version"]),
            _command_version(["gcc", "--version"]),
            _command_version(["wsl", "--version"]),
        ],
        "python_packages": {
            "gymnasium": _package_version("gymnasium"),
            "stable-baselines3": _package_version("stable_baselines3"),
            "sb3-contrib": _package_version("sb3_contrib"),
            "matplotlib": _package_version("matplotlib"),
            "numpy": _package_version("numpy"),
        },
        "seeds": [1, 7, 13],
        "dataset": {
            "name": "Project CodeNet C 500x4 strict protocol_v5",
            "train_index": "data/processed/project_codenet_c_500x4_strict/protocol_v5/train_harness_clean.index.json",
            "validation_index": "data/processed/project_codenet_c_500x4_strict/protocol_v5/validation_harness_clean.index.json",
            "test_index": "data/processed/project_codenet_c_500x4_strict/protocol_v5/test_harness_clean.index.json",
            "manifest": "data/manifests/project_codenet_c_500x4_strict.manifest.json",
        },
        "paper_commands": {
            "harness_audit": "python -m scripts.audit_harness_quality --include-excluded",
            "corrected_summary": "python -m scripts.apply_experiment_corrections",
            "paper_figures": "python -m scripts.plot_results",
            "training_curves": "python -m scripts.plot_training_curves",
            "test_suite": "python -m unittest discover -s tests -p \"test*.py\" -v",
        },
        "paths": _existing_paths(paths),
        "repo_relative_output": project_relative(output_dir),
    }
    dump_json(output_dir / "repro_manifest.json", payload)
    write_text(output_dir / "repro_manifest.md", _markdown(payload))
    print(f"Wrote reproducibility manifest: {output_dir / 'repro_manifest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
