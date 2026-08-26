from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .common import ensure_dir, load_structured_file, resolve_path, write_text


DEFAULT_BASELINE_SUMMARY = "artifacts/formal_eval_30_v2/long_training/experiment_summary/experiment_summary.json"
DEFAULT_GROUPED_SUMMARY = "artifacts/formal_eval_30_v2/grouped_summary/grouped_summary.json"
DEFAULT_CORRECTED_SUMMARY = "artifacts/formal_eval_30_v2/corrected_summaries/formal_eval_30_v2_corrected_summary.json"
DEFAULT_TIGRESS_BREAKDOWN = "artifacts/formal_eval_30_v2/tigress_failure_breakdown/tigress_failure_breakdown.json"
DEFAULT_V3_SUMMARY = "artifacts/formal_eval_30_v3_harness_clean/experiment_summary/experiment_summary.json"
DEFAULT_V3_GROUPED_SUMMARY = "artifacts/formal_eval_30_v3_harness_clean/experiment_summary/grouped_summary.json"
DEFAULT_V3_TIGRESS_BREAKDOWN = "artifacts/formal_eval_30_v3_harness_clean/tigress_failure_breakdown/tigress_failure_breakdown.json"
DEFAULT_FE150_SUMMARY = "artifacts/formal_eval_150_harness_clean/experiment_summary_corrected/experiment_summary.json"
DEFAULT_FE150_GROUPED_SUMMARY = "artifacts/formal_eval_150_harness_clean/experiment_summary_corrected/grouped_summary.json"
DEFAULT_FE150_TIGRESS_BREAKDOWN = "artifacts/formal_eval_150_harness_clean/tigress_failure_breakdown/tigress_failure_breakdown.json"
DEFAULT_FE150_STATIC_METRICS = "artifacts/formal_eval_150_harness_clean/static_obfuscation_metrics/static_obfuscation_metrics.json"
DEFAULT_FE150_DEOBF_METRICS = "artifacts/formal_eval_150_harness_clean/deobfuscation_attacks/deobfuscation_attack_metrics.json"
DEFAULT_FE150_DEOBF_V2_METRICS = "artifacts/formal_eval_150_harness_clean/deobfuscation_attacks_v2/deobfuscation_attack_metrics.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from experiment summaries.")
    parser.add_argument("--baseline-summary", default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--grouped-summary", default=DEFAULT_GROUPED_SUMMARY)
    parser.add_argument("--corrected-summary", default=DEFAULT_CORRECTED_SUMMARY)
    parser.add_argument("--tigress-breakdown", default=DEFAULT_TIGRESS_BREAKDOWN)
    parser.add_argument("--v3-summary", default=DEFAULT_V3_SUMMARY)
    parser.add_argument("--v3-grouped-summary", default=DEFAULT_V3_GROUPED_SUMMARY)
    parser.add_argument("--v3-tigress-breakdown", default=DEFAULT_V3_TIGRESS_BREAKDOWN)
    parser.add_argument("--fe150-summary", default=DEFAULT_FE150_SUMMARY)
    parser.add_argument("--fe150-grouped-summary", default=DEFAULT_FE150_GROUPED_SUMMARY)
    parser.add_argument("--fe150-tigress-breakdown", default=DEFAULT_FE150_TIGRESS_BREAKDOWN)
    parser.add_argument("--fe150-static-metrics", default=DEFAULT_FE150_STATIC_METRICS)
    parser.add_argument("--fe150-deobf-metrics", default=DEFAULT_FE150_DEOBF_METRICS)
    parser.add_argument("--fe150-deobf-v2-metrics", default=DEFAULT_FE150_DEOBF_V2_METRICS)
    parser.add_argument("--output-dir", default="artifacts/paper_figures")
    parser.add_argument("--formats", default="png", help="Comma-separated figure formats, e.g. png,pdf.")
    return parser.parse_args(argv)


def configure_matplotlib(output_dir: Path) -> Any:
    mpl_config = ensure_dir(output_dir.parent / "mplconfig")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
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
            "grid.alpha": 0.22,
            "axes.titleweight": "bold",
        }
    )
    return plt


def _load_dict(path_value: str | Path) -> dict[str, Any]:
    path = resolve_path(path_value)
    data = load_structured_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def _metric(group: dict[str, Any], metric_name: str, stat: str = "mean") -> float:
    value = group.get(metric_name, {})
    if isinstance(value, dict):
        return float(value.get(stat, 0.0))
    return float(value or 0.0)


def _method_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in summary.get("methods", []) if isinstance(item, dict)]


def _group_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in summary.get("groups", []) if isinstance(item, dict)]


def _group_variant(group: dict[str, Any]) -> str:
    return str(dict(group.get("group", {})).get("variant", "unknown"))


def _save(fig: Any, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    paths: list[str] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path.as_posix())
    return paths


def _annotate_bars(ax: Any, bars: Any, *, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_main_baselines(plt: Any, summary: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    rows = _method_rows(summary)
    wanted = ["random", "rule_based", "maskable_ppo", "maskable_ppo#2", "tigress"]
    label_map = {
        "random": "Random",
        "rule_based": "Rule-based",
        "maskable_ppo": "PPO short",
        "maskable_ppo#2": "PPO long",
        "tigress": "Tigress",
    }
    rows = [row for name in wanted for row in rows if row.get("method") == name]
    labels = [label_map.get(str(row.get("method")), str(row.get("method"))) for row in rows]
    metrics = [
        ("semantic_pass_rate", "Semantic pass rate"),
        ("mean_potency_score", "Potency"),
        ("mean_verifier_coverage", "Verifier coverage"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(13.5, 4.2), constrained_layout=True)
    colors = ["#3D5A80", "#98C1D9", "#EE6C4D", "#E0A458", "#293241"]
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(row.get(metric, 0.0)) for row in rows]
        bars = ax.bar(labels, values, color=colors[: len(values)])
        ax.set_title(title)
        ax.set_ylim(0, max(1.05, max(values or [0]) * 1.15))
        ax.tick_params(axis="x", rotation=30)
        _annotate_bars(ax, bars)
    fig.suptitle("Formal Eval 30 v2: Baseline Comparison", fontsize=14, fontweight="bold")
    paths = _save(fig, output_dir, "main_baseline_comparison", formats)
    plt.close(fig)
    return paths


def plot_short_vs_long(plt: Any, grouped: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    by_variant = {_group_variant(group): group for group in _group_rows(grouped)}
    variants = ["full_short", "full_long_1536"]
    labels = ["PPO 288 steps", "PPO 1536 steps"]
    metrics = [
        ("mean_potency_score", "Potency"),
        ("mean_verifier_coverage", "Verifier coverage"),
        ("mean_llvm_block_coverage", "LLVM block coverage"),
        ("mean_ir_structural_delta", "IR structural delta"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.2), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        means = [_metric(by_variant[variant], metric, "mean") for variant in variants]
        stds = [_metric(by_variant[variant], metric, "std") for variant in variants]
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=["#52796F", "#CAD2C5"])
        ax.set_title(title)
        ax.set_ylim(0, max(means) * 1.25 if means else 1)
        ax.tick_params(axis="x", rotation=20)
        _annotate_bars(ax, bars, fmt="{:.3f}")
    fig.suptitle("Maskable PPO: Short vs Long Training", fontsize=14, fontweight="bold")
    paths = _save(fig, output_dir, "ppo_short_vs_long", formats)
    plt.close(fig)
    return paths


def plot_ablation(plt: Any, grouped: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    by_variant = {_group_variant(group): group for group in _group_rows(grouped)}
    variants = ["full_short", "no_action_safety", "no_fuzz_verifier", "no_llvm_reward"]
    labels = ["Full", "No safety", "No fuzz", "No LLVM"]
    metrics = [
        ("semantic_pass_rate", "Semantic pass rate"),
        ("mean_verifier_coverage", "Verifier coverage"),
        ("mean_potency_score", "Potency"),
        ("failure_count", "Failure count"),
    ]
    colors = ["#2A9D8F", "#E76F51", "#F4A261", "#6D597A"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.2), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        means = [_metric(by_variant[variant], metric, "mean") for variant in variants]
        stds = [_metric(by_variant[variant], metric, "std") for variant in variants]
        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, max(means) * 1.25 if means else 1)
        ax.tick_params(axis="x", rotation=25)
        _annotate_bars(ax, bars, fmt="{:.2f}")
    fig.suptitle("Ablation Study: Component Necessity", fontsize=14, fontweight="bold")
    paths = _save(fig, output_dir, "ablation_components", formats)
    plt.close(fig)
    return paths


def plot_tigress_breakdown(plt: Any, breakdown: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    by_category = dict(dict(breakdown.get("summary", {})).get("by_category", {}))
    order = ["tool_parse_failure", "tool_compile_failure", "semantic_failure", "tool_timeout"]
    labels = [name for name in order if int(by_category.get(name, 0))]
    values = [int(by_category[name]) for name in labels]
    pretty = {
        "tool_parse_failure": "Tool parse",
        "tool_compile_failure": "Tool compile",
        "semantic_failure": "Semantic",
        "tool_timeout": "Timeout",
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    bars = ax.bar([pretty.get(label, label) for label in labels], values, color=["#8AB17D", "#E9C46A", "#E76F51", "#6D597A"])
    ax.set_title("Tigress Failure Breakdown")
    ax.set_ylabel("Failed samples")
    ax.set_ylim(0, max(values or [1]) + 1)
    _annotate_bars(ax, bars, fmt="{:.0f}")
    paths = _save(fig, output_dir, "tigress_failure_breakdown", formats)
    plt.close(fig)
    return paths


def plot_corrected_semantics(plt: Any, corrected: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    aggregates = [dict(item) for item in corrected.get("aggregates", []) if isinstance(item, dict)]
    wanted = ["full_short", "full_long_1536", "no_action_safety", "no_fuzz_verifier", "no_llvm_reward"]
    by_variant = {str(item.get("variant")): item for item in aggregates}
    labels = ["Full short", "Full long", "No safety", "No fuzz", "No LLVM"]
    values = [float(by_variant[name].get("semantic_pass_rate_mean", 0.0)) for name in wanted if name in by_variant]
    stds = [float(by_variant[name].get("semantic_pass_rate_std", 0.0)) for name in wanted if name in by_variant]
    labels = [label for label, name in zip(labels, wanted) if name in by_variant]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    bars = ax.bar(labels, values, yerr=stds, capsize=5, color=["#2A9D8F", "#52796F", "#E76F51", "#F4A261", "#6D597A"])
    ax.set_title("Semantic Pass Rate After p00067 Harness Correction")
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="x", rotation=25)
    _annotate_bars(ax, bars, fmt="{:.3f}")
    paths = _save(fig, output_dir, "corrected_semantic_pass_rate", formats)
    plt.close(fig)
    return paths


def plot_v3_clean_main(
    plt: Any,
    summary: dict[str, Any],
    grouped: dict[str, Any],
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    method_rows = _method_rows(summary)
    by_method = {str(row.get("method")): row for row in method_rows}
    ppo_group = next(
        (
            group
            for group in _group_rows(grouped)
            if dict(group.get("group", {})).get("method") == "maskable_ppo"
            and dict(group.get("group", {})).get("variant") == "full_short"
        ),
        {},
    )
    rows = [
        ("Random", by_method.get("random", {})),
        ("Rule-based", by_method.get("rule_based", {})),
        (
            "Maskable PPO\n3 seeds",
            {
                "semantic_pass_rate": _metric(ppo_group, "semantic_pass_rate", "mean"),
                "mean_potency_score": _metric(ppo_group, "mean_potency_score", "mean"),
                "mean_verifier_coverage": _metric(ppo_group, "mean_verifier_coverage", "mean"),
                "mean_ir_structural_delta": _metric(ppo_group, "mean_ir_structural_delta", "mean"),
            },
        ),
        ("Tigress", by_method.get("tigress", {})),
    ]
    metrics = [
        ("semantic_pass_rate", "Semantic pass"),
        ("mean_potency_score", "Potency"),
        ("mean_verifier_coverage", "Verifier coverage"),
        ("mean_ir_structural_delta", "IR structural delta"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15.2, 4.4), constrained_layout=True)
    colors = ["#355070", "#6D597A", "#2A9D8F", "#E76F51"]
    labels = [label for label, _row in rows]
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(row.get(metric, 0.0)) for _label, row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, max(1.05, max(values or [0]) * 1.18))
        ax.tick_params(axis="x", rotation=25)
        _annotate_bars(ax, bars, fmt="{:.3f}")
    fig.suptitle("Harness-Clean Formal Eval v3", fontsize=14, fontweight="bold")
    paths = _save(fig, output_dir, "formal_eval_v3_clean_main", formats)
    plt.close(fig)
    return paths


def plot_fe150_main(
    plt: Any,
    summary: dict[str, Any],
    grouped: dict[str, Any],
    output_dir: Path,
    formats: list[str],
) -> list[str]:
    method_rows = _method_rows(summary)
    by_method = {str(row.get("method")): row for row in method_rows}
    ppo_group = next(
        (
            group
            for group in _group_rows(grouped)
            if dict(group.get("group", {})).get("method") == "maskable_ppo"
            and dict(group.get("group", {})).get("variant") == "full_short"
        ),
        {},
    )
    rows = [
        ("Random", by_method.get("random", {})),
        ("Rule-based", by_method.get("rule_based", {})),
        (
            "Maskable PPO\n3 seeds",
            {
                "semantic_pass_rate": _metric(ppo_group, "semantic_pass_rate", "mean"),
                "mean_potency_score": _metric(ppo_group, "mean_potency_score", "mean"),
                "mean_verifier_coverage": _metric(ppo_group, "mean_verifier_coverage", "mean"),
                "mean_ir_structural_delta": _metric(ppo_group, "mean_ir_structural_delta", "mean"),
                "failure_count": _metric(ppo_group, "failure_count", "mean"),
            },
        ),
        ("Tigress", by_method.get("tigress", {})),
    ]
    metrics = [
        ("semantic_pass_rate", "Semantic pass"),
        ("mean_potency_score", "Potency"),
        ("mean_verifier_coverage", "Verifier coverage"),
        ("failure_count", "Failures"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15.2, 4.4), constrained_layout=True)
    colors = ["#355070", "#6D597A", "#2A9D8F", "#E76F51"]
    labels = [label for label, _row in rows]
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(row.get(metric, 0.0)) for _label, row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, max(1.05, max(values or [0]) * 1.18))
        ax.tick_params(axis="x", rotation=25)
        _annotate_bars(ax, bars, fmt="{:.3f}" if metric != "failure_count" else "{:.1f}")
    fig.suptitle("Expanded Harness-Clean Formal Eval 150", fontsize=14, fontweight="bold")
    paths = _save(fig, output_dir, "formal_eval_150_clean_main", formats)
    plt.close(fig)
    return paths


def _aggregate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in payload.get("aggregates", []) if isinstance(item, dict)]


def _aggregate_key(row: dict[str, Any]) -> str:
    method = str(row.get("method", ""))
    seed = str(row.get("seed", ""))
    if method == "maskable_ppo":
        return f"PPO {seed}"
    if method == "rule_based":
        return "Rule-based"
    return method.title()


def plot_fe150_static_metrics(plt: Any, payload: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    rows = _aggregate_rows(payload)
    order = ["random", "rule_based", "maskable_ppo", "tigress"]
    rows = sorted(rows, key=lambda row: (order.index(str(row.get("method"))) if str(row.get("method")) in order else 99, str(row.get("seed", ""))))
    labels = [_aggregate_key(row) for row in rows]
    metrics = [
        ("mean_static_obfuscation_score", "Static score"),
        ("mean_attacker_resistance_score", "Attacker resistance"),
        ("mean_token_edit_distance_ratio", "Token edit"),
        ("mean_identifier_jaccard_distance", "Identifier distance"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5), constrained_layout=True)
    colors = ["#355070" if "PPO" not in label else "#2A9D8F" for label in labels]
    colors = ["#E76F51" if label == "Tigress" else color for label, color in zip(labels, colors)]
    colors = ["#6D597A" if label == "Rule-based" else color for label, color in zip(labels, colors)]
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(row.get(metric, 0.0)) for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, max(1.05, max(values or [0]) * 1.15))
        ax.tick_params(axis="x", rotation=35)
        _annotate_bars(ax, bars, fmt="{:.3f}")
    fig.suptitle("Expanded Eval 150: Static Obfuscation Strength", fontsize=14, fontweight="bold")
    paths = _save(fig, output_dir, "formal_eval_150_static_metrics", formats)
    plt.close(fig)
    return paths


def plot_fe150_deobf_metrics(plt: Any, payload: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    rows = [row for row in _aggregate_rows(payload) if str(row.get("attack")) == "normalize"]
    order = ["random", "rule_based", "maskable_ppo", "tigress"]
    rows = sorted(rows, key=lambda row: (order.index(str(row.get("method"))) if str(row.get("method")) in order else 99, str(row.get("seed", ""))))
    labels = [_aggregate_key(row) for row in rows]
    metrics = [
        ("mean_pre_attack_static_obfuscation_score", "Pre-attack score"),
        ("mean_post_attack_static_obfuscation_score", "Post-normalize score"),
        ("mean_score_retention_ratio", "Retention ratio"),
        ("mean_residual_attacker_resistance_score", "Residual resistance"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5), constrained_layout=True)
    colors = ["#355070" if "PPO" not in label else "#2A9D8F" for label in labels]
    colors = ["#E76F51" if label == "Tigress" else color for label, color in zip(labels, colors)]
    colors = ["#6D597A" if label == "Rule-based" else color for label, color in zip(labels, colors)]
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(row.get(metric, 0.0)) for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylim(0, max(1.05, max(values or [0]) * 1.15))
        ax.tick_params(axis="x", rotation=35)
        _annotate_bars(ax, bars, fmt="{:.3f}")
    fig.suptitle("Expanded Eval 150: Lightweight Deobfuscation Attack", fontsize=14, fontweight="bold")
    paths = _save(fig, output_dir, "formal_eval_150_deobfuscation_metrics", formats)
    plt.close(fig)
    return paths


def plot_fe150_deobf_v2_metrics(plt: Any, payload: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    rows = [row for row in _aggregate_rows(payload) if str(row.get("attack")) in {"cfg_cleanup", "llvm_o2_ir"}]
    order = ["random", "rule_based", "maskable_ppo", "tigress"]
    rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("attack")),
            order.index(str(row.get("method"))) if str(row.get("method")) in order else 99,
            str(row.get("seed", "")),
        ),
    )
    labels = [f"{_aggregate_key(row)}\n{row.get('attack')}" for row in rows]
    values = [float(row.get("mean_post_attack_static_obfuscation_score", 0.0)) for row in rows]
    colors = ["#457B9D" if str(row.get("attack")) == "cfg_cleanup" else "#E76F51" for row in rows]

    fig, ax = plt.subplots(figsize=(15.5, 5.0), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Expanded Eval 150: CFG Cleanup vs LLVM O2 IR Residual Strength")
    ax.set_ylabel("Post-attack static score")
    ax.set_ylim(0, max(1.05, max(values or [0]) * 1.15))
    ax.tick_params(axis="x", rotation=45)
    _annotate_bars(ax, bars, fmt="{:.3f}")
    paths = _save(fig, output_dir, "formal_eval_150_deobfuscation_v2_metrics", formats)
    plt.close(fig)
    return paths


def plot_v3_tigress_breakdown(plt: Any, breakdown: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    paths = plot_tigress_breakdown(plt, breakdown, output_dir, formats)
    renamed: list[str] = []
    for path_text in paths:
        path = Path(path_text)
        target = path.with_name(f"formal_eval_v3_{path.name}")
        path.replace(target)
        renamed.append(target.as_posix())
    return renamed


def plot_fe150_tigress_breakdown(plt: Any, breakdown: dict[str, Any], output_dir: Path, formats: list[str]) -> list[str]:
    paths = plot_tigress_breakdown(plt, breakdown, output_dir, formats)
    renamed: list[str] = []
    for path_text in paths:
        path = Path(path_text)
        target = path.with_name(f"formal_eval_150_{path.name}")
        path.replace(target)
        renamed.append(target.as_posix())
    return renamed


def render_index(generated: dict[str, list[str]]) -> str:
    lines = [
        "# Paper Figures",
        "",
        "Generated figures from formal_eval_30_v2, formal_eval_30_v3, and expanded formal_eval_150 harness-clean summaries.",
        "",
    ]
    for title, paths in generated.items():
        lines.append(f"## {title}")
        lines.extend(f"- `{path}`" for path in paths)
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    formats = [part.strip().lower() for part in str(args.formats).split(",") if part.strip()]
    plt = configure_matplotlib(output_dir)

    baseline_summary = _load_dict(args.baseline_summary)
    grouped_summary = _load_dict(args.grouped_summary)
    corrected_summary = _load_dict(args.corrected_summary)
    tigress_breakdown = _load_dict(args.tigress_breakdown)
    v3_summary = _load_dict(args.v3_summary)
    v3_grouped_summary = _load_dict(args.v3_grouped_summary)
    v3_tigress_breakdown = _load_dict(args.v3_tigress_breakdown)
    fe150_summary = _load_dict(args.fe150_summary)
    fe150_grouped_summary = _load_dict(args.fe150_grouped_summary)
    fe150_tigress_breakdown = _load_dict(args.fe150_tigress_breakdown)
    fe150_static_metrics = _load_dict(args.fe150_static_metrics)
    fe150_deobf_metrics = _load_dict(args.fe150_deobf_metrics)
    fe150_deobf_v2_metrics = _load_dict(args.fe150_deobf_v2_metrics)

    generated = {
        "Expanded Harness-Clean 150 Main Results": plot_fe150_main(
            plt, fe150_summary, fe150_grouped_summary, output_dir, formats
        ),
        "Expanded Harness-Clean 150 Static Metrics": plot_fe150_static_metrics(
            plt, fe150_static_metrics, output_dir, formats
        ),
        "Expanded Harness-Clean 150 Deobfuscation Metrics": plot_fe150_deobf_metrics(
            plt, fe150_deobf_metrics, output_dir, formats
        ),
        "Expanded Harness-Clean 150 Deobfuscation v2 Metrics": plot_fe150_deobf_v2_metrics(
            plt, fe150_deobf_v2_metrics, output_dir, formats
        ),
        "Expanded Harness-Clean 150 Tigress Breakdown": plot_fe150_tigress_breakdown(
            plt, fe150_tigress_breakdown, output_dir, formats
        ),
        "Harness-Clean v3 Main Results": plot_v3_clean_main(
            plt, v3_summary, v3_grouped_summary, output_dir, formats
        ),
        "Harness-Clean v3 Tigress Breakdown": plot_v3_tigress_breakdown(
            plt, v3_tigress_breakdown, output_dir, formats
        ),
        "Main Baseline Comparison": plot_main_baselines(plt, baseline_summary, output_dir, formats),
        "PPO Short vs Long": plot_short_vs_long(plt, grouped_summary, output_dir, formats),
        "Ablation Components": plot_ablation(plt, grouped_summary, output_dir, formats),
        "Corrected Semantic Pass Rate": plot_corrected_semantics(plt, corrected_summary, output_dir, formats),
        "Tigress Failure Breakdown": plot_tigress_breakdown(plt, tigress_breakdown, output_dir, formats),
    }
    write_text(output_dir / "README.md", render_index(generated))
    print(f"Wrote paper figures to: {output_dir}")
    for paths in generated.values():
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
