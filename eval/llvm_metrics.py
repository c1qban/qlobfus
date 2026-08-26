from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_block_id(value: object) -> str:
    return str(value or "").strip()


def build_ir_metrics_snapshot(
    *,
    baseline_llvm_blocks: set[str],
    current_llvm_blocks: set[str],
    touched_llvm_blocks: set[str],
    baseline_branch_blocks: set[str],
    rewritten_branch_blocks: set[str],
) -> dict[str, Any]:
    baseline_count = len(baseline_llvm_blocks)
    branch_count = len(baseline_branch_blocks)
    symmetric_delta = baseline_llvm_blocks.symmetric_difference(current_llvm_blocks)

    llvm_block_coverage = (
        len(touched_llvm_blocks) / float(baseline_count)
        if baseline_count > 0
        else 0.0
    )
    branch_rewrite_ratio = (
        len(rewritten_branch_blocks) / float(branch_count)
        if branch_count > 0
        else 0.0
    )
    ir_structural_delta = (
        len(symmetric_delta) / float(max(baseline_count, 1))
        if baseline_count > 0
        else 0.0
    )

    return {
        "llvm_metric_available": baseline_count > 0,
        "baseline_llvm_block_count": baseline_count,
        "current_llvm_block_count": len(current_llvm_blocks),
        "touched_llvm_block_count": len(touched_llvm_blocks),
        "baseline_branch_block_count": branch_count,
        "rewritten_branch_block_count": len(rewritten_branch_blocks),
        "llvm_block_coverage": llvm_block_coverage,
        "branch_rewrite_ratio": branch_rewrite_ratio,
        "ir_structural_delta": ir_structural_delta,
        "llvm_block_ids": sorted(current_llvm_blocks),
        "touched_llvm_block_ids": sorted(touched_llvm_blocks),
        "rewritten_branch_block_ids": sorted(rewritten_branch_blocks),
        "symmetric_delta_block_ids": sorted(symmetric_delta),
    }


def aggregate_trace_ir_metrics(run_dir: Path) -> dict[str, Any]:
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
        if reset_event is None:
            continue

        baseline_metrics = dict(reset_event.get("ir_metrics", {}))
        step_metrics = [
            dict(event.get("reward_breakdown", {}).get("ir_metrics", {}))
            for event in step_events
            if isinstance(event.get("reward_breakdown", {}).get("ir_metrics"), dict)
        ]
        final_metrics = step_metrics[-1] if step_metrics else baseline_metrics
        episode_summaries.append(
            {
                "trace_path": str(trace_path),
                "sample_id": str(reset_event.get("sample_id", "")),
                "baseline_metrics": baseline_metrics,
                "final_metrics": final_metrics,
                "step_count": len(step_events),
            }
        )

    available_episodes = [
        episode for episode in episode_summaries
        if bool(episode.get("final_metrics", {}).get("llvm_metric_available"))
    ]
    if not episode_summaries:
        return {
            "status": "unavailable",
            "episode_count": 0,
            "llvm_metric_episode_count": 0,
            "episodes": [],
        }

    def _mean(metric_name: str) -> float:
        if not available_episodes:
            return 0.0
        return sum(float(episode["final_metrics"].get(metric_name, 0.0)) for episode in available_episodes) / float(len(available_episodes))

    return {
        "status": "ok",
        "episode_count": len(episode_summaries),
        "llvm_metric_episode_count": len(available_episodes),
        "mean_llvm_block_coverage": _mean("llvm_block_coverage"),
        "mean_branch_rewrite_ratio": _mean("branch_rewrite_ratio"),
        "mean_ir_structural_delta": _mean("ir_structural_delta"),
        "max_llvm_block_coverage": max((float(episode["final_metrics"].get("llvm_block_coverage", 0.0)) for episode in available_episodes), default=0.0),
        "max_branch_rewrite_ratio": max((float(episode["final_metrics"].get("branch_rewrite_ratio", 0.0)) for episode in available_episodes), default=0.0),
        "max_ir_structural_delta": max((float(episode["final_metrics"].get("ir_structural_delta", 0.0)) for episode in available_episodes), default=0.0),
        "episodes": episode_summaries,
    }


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
