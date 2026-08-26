from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def aggregate_trace_obfuscation_metrics(run_dir: Path) -> dict[str, Any]:
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
        applied_actions = [
            event
            for event in step_events
            if bool(event.get("action_valid"))
            and str(event.get("action", "")) != "stop"
            and bool(event.get("operator_metadata", {}))
        ]

        reward_breakdown = dict(final_step.get("reward_breakdown", {}))
        verification = dict(final_step.get("verification", {}))

        episode_summaries.append(
            {
                "trace_path": str(trace_path),
                "sample_id": str(reset_event.get("sample_id", "")),
                "step_count": len(step_events),
                "applied_action_count": len(applied_actions),
                "unique_operator_count": len(set(operator_names)),
                "operator_names": operator_names,
                "potency_score": _coerce_float(final_step.get("potency_score")),
                "cost_penalty": _coerce_float(final_step.get("cost_penalty")),
                "semantic_score": _coerce_float(verification.get("semantic_score")),
                "verifier_coverage": _coerce_float(verification.get("verifier_coverage")),
                "semantic_gate_passed": bool(reward_breakdown.get("semantic_gate_passed")),
                "compile_succeeded": bool(verification.get("compile", {}).get("succeeded", False)),
                "termination_reason": str(final_step.get("termination_reason", "")),
            }
        )

    if not episode_summaries:
        return {
            "status": "unavailable",
            "episode_count": 0,
            "episodes": [],
        }

    gate_passed_episodes = [episode for episode in episode_summaries if episode["semantic_gate_passed"]]

    def _mean(metric_name: str) -> float:
        return sum(float(episode.get(metric_name, 0.0)) for episode in episode_summaries) / float(len(episode_summaries))

    def _max(metric_name: str) -> float:
        return max((float(episode.get(metric_name, 0.0)) for episode in episode_summaries), default=0.0)

    return {
        "status": "ok",
        "episode_count": len(episode_summaries),
        "semantic_gate_pass_rate": len(gate_passed_episodes) / float(len(episode_summaries)),
        "compile_success_rate": sum(1 for episode in episode_summaries if episode["compile_succeeded"]) / float(len(episode_summaries)),
        "mean_potency_score": _mean("potency_score"),
        "max_potency_score": _max("potency_score"),
        "mean_cost_penalty": _mean("cost_penalty"),
        "max_cost_penalty": _max("cost_penalty"),
        "mean_semantic_score": _mean("semantic_score"),
        "mean_verifier_coverage": _mean("verifier_coverage"),
        "mean_step_count": _mean("step_count"),
        "mean_applied_action_count": _mean("applied_action_count"),
        "mean_unique_operator_count": _mean("unique_operator_count"),
        "episodes": episode_summaries,
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
