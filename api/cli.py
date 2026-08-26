from __future__ import annotations

import argparse
from typing import Callable

from scripts import compare_experiments, dataset_report, evaluate, prepare_dataset, report, run_baselines, sample_project_codenet_subset, train


CommandFunc = Callable[[list[str] | None], None]

COMMANDS: dict[str, CommandFunc] = {
    "prepare-dataset": prepare_dataset.main,
    "dataset-report": dataset_report.main,
    "compare-experiments": compare_experiments.main,
    "train": train.main,
    "evaluate": evaluate.main,
    "report": report.main,
    "run-baselines": run_baselines.main,
    "sample-codenet-subset": sample_project_codenet_subset.main,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CLI for the RL obfuscator scaffolding.")
    parser.add_argument("command", choices=sorted(COMMANDS), help="Subcommand to execute.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected subcommand.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    COMMANDS[args.command](args.args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
