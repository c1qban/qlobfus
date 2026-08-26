from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.llvm_metrics import aggregate_trace_ir_metrics
from eval.obfuscation_metrics import aggregate_trace_obfuscation_metrics
from eval.resilience_metrics import aggregate_trace_resilience_metrics
from verifier.models import VerificationSummary


def build_evaluation_summary(
    job_name: str,
    benchmark: dict[str, Any],
    metrics: list[str],
    training_context: dict[str, Any] | None = None,
    verification: VerificationSummary | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    summary = {
        "job_name": job_name,
        "benchmark": benchmark,
        "metrics": metrics,
        "training_context": training_context or {},
        "notes": [
            "Populate potency metrics from eval/potency.",
            "Populate resilience metrics from eval/resilience.",
            "Populate cost metrics from eval/cost and runtime instrumentation.",
        ],
    }

    if verification is not None:
        summary["semantic_verification"] = verification.to_dict()
        summary["derived_flags"] = {
            "passes_semantic_gate": verification.semantic_score >= verification.semantic_threshold,
        }

    if run_dir is not None:
        ir_metrics = aggregate_trace_ir_metrics(run_dir)
        obfuscation_metrics = aggregate_trace_obfuscation_metrics(run_dir)
        resilience_metrics = aggregate_trace_resilience_metrics(run_dir)
        summary["llvm_ir_metrics"] = ir_metrics
        summary["obfuscation_metrics"] = obfuscation_metrics
        summary["resilience_metrics"] = resilience_metrics
        summary.setdefault("derived_flags", {})
        summary["derived_flags"].update(
            {
                "llvm_metrics_available": ir_metrics.get("llvm_metric_episode_count", 0) > 0,
                "obfuscation_metrics_available": obfuscation_metrics.get("episode_count", 0) > 0,
                "resilience_metrics_available": resilience_metrics.get("episode_count", 0) > 0,
            }
        )

    return summary
