from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .common import ensure_dir, resolve_path, write_text


DEFAULT_RUNS = [
    ("Full short seed 1", "artifacts/formal_eval_30_v2/multiseed/mppo_seed1/logs/episode_metrics.jsonl"),
    ("Full short seed 7", "artifacts/formal_eval_30_v2/mppo_run/logs/episode_metrics.jsonl"),
    ("Full short seed 13", "artifacts/formal_eval_30_v2/multiseed/mppo_seed13/logs/episode_metrics.jsonl"),
    ("Full long seed 1", "artifacts/formal_eval_30_v2/long_training/full_seed1_1536/logs/episode_metrics.jsonl"),
    ("Full long seed 7", "artifacts/formal_eval_30_v2/long_training/full_seed7_1536/logs/episode_metrics.jsonl"),
    ("Full long seed 13", "artifacts/formal_eval_30_v2/long_training/full_seed13_1536/logs/episode_metrics.jsonl"),
]


METRICS = [
    ("episode_reward", "Episode reward"),
    ("semantic_score", "Semantic score"),
    ("potency_score", "Potency score"),
    ("verifier_coverage", "Verifier coverage"),
    ("llvm_block_coverage", "LLVM block coverage"),
    ("ir_structural_delta", "IR structural delta"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO training curves from episode_metrics.jsonl files.")
    parser.add_argument("--output-dir", default="artifacts/paper_figures/training_curves")
    parser.add_argument("--formats", default="png")
    parser.add_argument("--window", type=int, default=5, help="Moving average window.")
    return parser.parse_args(argv)


def configure_matplotlib(output_dir: Path) -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir.parent / "mplconfig")))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
    return rows


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def _save(fig: Any, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    paths: list[str] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path.as_posix())
    return paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    plt = configure_matplotlib(output_dir)

    runs = [(label, resolve_path(path), _read_jsonl(resolve_path(path))) for label, path in DEFAULT_RUNS]
    runs = [(label, path, rows) for label, path, rows in runs if rows]
    generated: list[str] = []

    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for label, _path, rows in runs:
            values = [float(row.get(metric, 0.0)) for row in rows]
            x_values = list(range(1, len(values) + 1))
            ax.plot(x_values, _moving_average(values, args.window), linewidth=1.8, label=label)
        ax.set_title(f"{title} during training")
        ax.set_xlabel("Episode")
        ax.set_ylabel(title)
        ax.legend(fontsize=8, ncol=2)
        generated.extend(_save(fig, output_dir, metric, formats))
        plt.close(fig)

    lines = [
        "# Training Curves",
        "",
        f"- moving average window: `{args.window}`",
        "",
        "## Runs",
        "",
    ]
    for label, path, rows in runs:
        lines.append(f"- `{label}`: `{path.as_posix()}` ({len(rows)} episodes)")
    lines.extend(["", "## Figures", ""])
    for path in generated:
        lines.append(f"- `{path}`")
    write_text(output_dir / "README.md", "\n".join(lines) + "\n")
    print(f"Wrote training curves to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
