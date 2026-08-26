from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def aggregate_trace_resilience_metrics(run_dir: Path) -> dict[str, Any]:
    trace_paths = sorted((run_dir / "artifacts").glob("env_workers/**/*.jsonl"))
    episode_summaries: list[dict[str, Any]] = []

    for trace_path in trace_paths:
        if trace_path.name != "transform_trace.jsonl":
            continue
        events = _load_trace_events(trace_path)
        if not events:
            continue

        reset_event = next((event for event in events if event.get("event") == "reset"), None)
        step_events = [event for event in events if event.get("event") == "step"]
        if reset_event is None or not step_events:
            continue

        final_step = step_events[-1]
        operator_names = [
            str(event.get("operator_metadata", {}).get("operator_name", "")).strip()
            for event in step_events
            if str(event.get("operator_metadata", {}).get("operator_name", "")).strip()
        ]
        applied_action_count = sum(
            1
            for event in step_events
            if bool(event.get("action_valid"))
            and str(event.get("action", "")) != "stop"
            and bool(event.get("operator_metadata", {}))
        )
        unique_operator_count = len(set(operator_names))
        operator_diversity_ratio = (
            unique_operator_count / float(max(applied_action_count, 1))
            if applied_action_count > 0
            else 0.0
        )

        reward_breakdown = dict(final_step.get("reward_breakdown", {}))
        ir_metrics = dict(reward_breakdown.get("ir_metrics", {}))
        flatten_cfg_used = "flatten_cfg" in operator_names
        branch_rewrite_ratio = _coerce_float(ir_metrics.get("branch_rewrite_ratio"))
        ir_structural_delta = _coerce_float(ir_metrics.get("ir_structural_delta"))
        llvm_block_coverage = _coerce_float(ir_metrics.get("llvm_block_coverage"))
        action_depth_score = min(applied_action_count / 8.0, 1.0)
        proxy_resilience_score = min(
            1.0,
            (0.30 * operator_diversity_ratio)
            + (0.20 * float(flatten_cfg_used))
            + (0.20 * branch_rewrite_ratio)
            + (0.15 * ir_structural_delta)
            + (0.10 * llvm_block_coverage)
            + (0.05 * action_depth_score),
        )

        episode_summaries.append(
            {
                "trace_path": str(trace_path),
                "sample_id": str(reset_event.get("sample_id", "")),
                "applied_action_count": applied_action_count,
                "unique_operator_count": unique_operator_count,
                "operator_diversity_ratio": operator_diversity_ratio,
                "flatten_cfg_used": flatten_cfg_used,
                "branch_rewrite_ratio": branch_rewrite_ratio,
                "ir_structural_delta": ir_structural_delta,
                "llvm_block_coverage": llvm_block_coverage,
                "proxy_resilience_score": proxy_resilience_score,
                "semantic_gate_passed": bool(reward_breakdown.get("semantic_gate_passed")),
            }
        )

    if not episode_summaries:
        return {
            "status": "unavailable",
            "proxy_based": True,
            "episode_count": 0,
            "episodes": [],
            "notes": [
                "Resilience metrics are currently unavailable because no transform traces were found.",
            ],
        }

    def _mean(metric_name: str) -> float:
        return sum(float(episode.get(metric_name, 0.0)) for episode in episode_summaries) / float(len(episode_summaries))

    return {
        "status": "ok",
        "proxy_based": True,
        "episode_count": len(episode_summaries),
        "flatten_cfg_episode_rate": sum(1 for episode in episode_summaries if episode["flatten_cfg_used"]) / float(len(episode_summaries)),
        "semantic_gate_pass_rate": sum(1 for episode in episode_summaries if episode["semantic_gate_passed"]) / float(len(episode_summaries)),
        "mean_operator_diversity_ratio": _mean("operator_diversity_ratio"),
        "mean_branch_rewrite_ratio": _mean("branch_rewrite_ratio"),
        "mean_ir_structural_delta": _mean("ir_structural_delta"),
        "mean_llvm_block_coverage": _mean("llvm_block_coverage"),
        "mean_proxy_resilience_score": _mean("proxy_resilience_score"),
        "max_proxy_resilience_score": max((float(episode.get("proxy_resilience_score", 0.0)) for episode in episode_summaries), default=0.0),
        "episodes": episode_summaries,
        "notes": [
            "These resilience metrics are proxy-based and summarize structural obfuscation signals, not true deobfuscation resistance.",
            "A stronger resilience evaluation still requires external recovery/deobfuscation baselines or dedicated robustness experiments.",
        ],
    }


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_trace_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events
