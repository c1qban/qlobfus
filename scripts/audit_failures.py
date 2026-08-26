from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit semantic/compile failures from baseline and PPO experiment outputs.")
    parser.add_argument("--baseline-summary", action="append", default=[], help="Path to reports/baseline_summary.json.")
    parser.add_argument("--ppo-run-dir", action="append", default=[], help="Path to a Maskable PPO run directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for failure_audit.json and failure_audit.md.")
    parser.add_argument("--semantic-threshold", type=float, default=0.97, help="Minimum semantic score treated as passing.")
    return parser.parse_args(argv)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _stderr_tail(stderr: str, *, max_lines: int = 10) -> str:
    lines = [line.rstrip() for line in stderr.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _failure_reasons(episode: dict[str, Any], semantic_threshold: float) -> list[str]:
    reasons: list[str] = []
    status = str(episode.get("status", "")).lower()
    stderr = str(episode.get("transform_stderr", "")).lower()
    stdout = str(episode.get("transform_stdout", "")).lower()
    if "timeout" in status or "timed out" in stderr or "timed out" in stdout:
        reasons.append("tool_timeout")
    if status in {"transform_failed", "skipped_tool_missing"} or "parsing error" in stderr or "syntax error" in stderr:
        reasons.append("tool_parse_failure")
    if not bool(episode.get("compile_succeeded", True)):
        reasons.append("compile_failed")
    if _safe_float(episode.get("semantic_score", 1.0), 0.0) < semantic_threshold:
        reasons.append("semantic_below_threshold")
    termination_reason = str(episode.get("termination_reason", ""))
    if termination_reason in {"semantic_gate_failed", "operator_not_applicable", "invalid_action", "missing_operator"}:
        reasons.append(termination_reason)
    return sorted(set(reasons))


def _select_failure_event(trace_events: list[dict[str, Any]], semantic_threshold: float) -> dict[str, Any]:
    step_events = [event for event in trace_events if event.get("event") == "step"]
    for event in step_events:
        verification = dict(event.get("verification", {}))
        compile_info = dict(verification.get("compile", {}))
        reward_breakdown = dict(event.get("reward_breakdown", {}))
        if (
            not bool(compile_info.get("succeeded", True))
            or _safe_float(verification.get("semantic_score", event.get("semantic_score", 1.0)), 1.0) < semantic_threshold
            or str(event.get("termination_reason", "")) == "semantic_gate_failed"
            or reward_breakdown.get("semantic_gate_passed") is False
        ):
            return event
    return step_events[-1] if step_events else (trace_events[-1] if trace_events else {})


def _trace_failure_details(trace_path_value: str, semantic_threshold: float) -> dict[str, Any]:
    if not trace_path_value:
        return {"trace_found": False}
    trace_path = resolve_path(trace_path_value)
    if not trace_path.exists():
        return {"trace_found": False, "trace_path": trace_path_value}

    events = _read_jsonl(trace_path)
    event = _select_failure_event(events, semantic_threshold)
    operator_metadata = dict(event.get("operator_metadata", {}))
    parameters = dict(operator_metadata.get("parameters", {}))
    verification = dict(event.get("verification", {}))
    compile_info = dict(verification.get("compile", {}))
    tests_info = dict(verification.get("tests", {}))
    diff_info = dict(verification.get("diff", {}))
    fuzz_info = dict(verification.get("fuzz", {}))
    return {
        "trace_found": True,
        "trace_path": project_relative(trace_path),
        "event": str(event.get("event", "")),
        "action": str(event.get("action", "")),
        "requested_action": str(event.get("requested_action", "")),
        "termination_reason": str(event.get("termination_reason", "")),
        "operator_name": str(operator_metadata.get("operator_name", "")),
        "location_tag": str(operator_metadata.get("location_tag", "")),
        "parameter_tag": str(operator_metadata.get("parameter_tag", "")),
        "node_id": str(operator_metadata.get("node_id", parameters.get("node_id", ""))),
        "risk_score": _safe_float(parameters.get("risk_score", operator_metadata.get("risk_score", 0.0))),
        "risk_level": str(parameters.get("risk_level", operator_metadata.get("risk_level", ""))),
        "risk_reasons": list(parameters.get("risk_reasons", operator_metadata.get("risk_reasons", [])) or []),
        "compile_succeeded": bool(compile_info.get("succeeded", True)),
        "compile_returncode": compile_info.get("returncode", ""),
        "compile_stderr_tail": _stderr_tail(str(compile_info.get("stderr", ""))),
        "tests_passed": tests_info.get("passed", ""),
        "tests_failed_cases": tests_info.get("failed_cases", ""),
        "diff_executed": diff_info.get("executed", ""),
        "diff_passed": diff_info.get("passed", ""),
        "fuzz_executed": fuzz_info.get("executed", ""),
        "fuzz_passed": fuzz_info.get("passed", ""),
    }


def _audit_episode(
    *,
    method: str,
    episode: dict[str, Any],
    semantic_threshold: float,
) -> dict[str, Any] | None:
    reasons = _failure_reasons(episode, semantic_threshold)
    trace_details = _trace_failure_details(str(episode.get("trace_path", "")), semantic_threshold)
    if not bool(trace_details.get("compile_succeeded", episode.get("compile_succeeded", True))):
        reasons.append("compile_failed")
    if str(trace_details.get("termination_reason", "")) == "semantic_gate_failed":
        reasons.append("semantic_gate_failed")
    reasons = sorted(set(reasons))
    if not reasons:
        return None
    return {
        "method": method,
        "sample_id": str(episode.get("sample_id", "")),
        "split": str(episode.get("split", "")),
        "failure_reasons": reasons,
        "semantic_score": _safe_float(episode.get("semantic_score", 0.0)),
        "compile_succeeded": bool(episode.get("compile_succeeded", trace_details.get("compile_succeeded", True))),
        "termination_reason": str(episode.get("termination_reason", trace_details.get("termination_reason", ""))),
        "reward": _safe_float(episode.get("reward", episode.get("episode_reward", 0.0))),
        "episode_length": int(episode.get("step_index", episode.get("episode_length", 0)) or 0),
        "trace": trace_details,
    }


def audit_baseline_summary(path: Path, semantic_threshold: float) -> list[dict[str, Any]]:
    summary = load_structured_file(path)
    failures: list[dict[str, Any]] = []
    for method, result in sorted(dict(summary.get("results", {})).items()):
        for episode in dict(result).get("episodes", []):
            failure = _audit_episode(method=str(method), episode=dict(episode), semantic_threshold=semantic_threshold)
            if failure is not None:
                failures.append(failure)
    return failures


def audit_ppo_run(run_dir: Path, semantic_threshold: float) -> list[dict[str, Any]]:
    episode_metrics_path = run_dir / "logs" / "episode_metrics.jsonl"
    failures: list[dict[str, Any]] = []
    for episode in _read_jsonl(episode_metrics_path):
        failure = _audit_episode(method="maskable_ppo", episode=episode, semantic_threshold=semantic_threshold)
        if failure is not None:
            failures.append(failure)
    return failures


def summarize_failures(failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_operator: Counter[str] = Counter()
    by_risk_level: Counter[str] = Counter()
    by_sample: Counter[str] = Counter()
    for failure in failures:
        by_method[str(failure.get("method", ""))] += 1
        by_sample[str(failure.get("sample_id", ""))] += 1
        by_reason.update(str(reason) for reason in failure.get("failure_reasons", []))
        trace = dict(failure.get("trace", {}))
        operator_name = str(trace.get("operator_name", "")) or "unknown"
        risk_level = str(trace.get("risk_level", "")) or "unlabeled"
        by_operator[operator_name] += 1
        by_risk_level[risk_level] += 1
    return {
        "failure_count": len(failures),
        "by_method": dict(sorted(by_method.items())),
        "by_reason": dict(sorted(by_reason.items())),
        "by_operator": dict(sorted(by_operator.items())),
        "by_risk_level": dict(sorted(by_risk_level.items())),
        "repeated_samples": {
            sample_id: count
            for sample_id, count in sorted(by_sample.items())
            if count > 1
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict(report.get("summary", {}))
    failures = [dict(item) for item in report.get("failures", [])]
    lines = [
        "# Failure Audit Report",
        "",
        f"- Created at: `{report.get('created_at', '')}`",
        f"- Baseline summary: `{report.get('baseline_summary_path', '')}`",
        f"- PPO run: `{report.get('ppo_run_dir', '')}`",
        f"- Semantic threshold: `{report.get('semantic_threshold', '')}`",
        "",
        "## Summary",
        f"- Failure count: `{summary.get('failure_count', 0)}`",
        f"- By method: `{summary.get('by_method', {})}`",
        f"- By reason: `{summary.get('by_reason', {})}`",
        f"- By operator: `{summary.get('by_operator', {})}`",
        f"- By risk level: `{summary.get('by_risk_level', {})}`",
        "",
        "## Failed Episodes",
    ]
    if not failures:
        lines.append("- No failures found.")
    else:
        headers = ["method", "sample_id", "reasons", "operator", "action", "risk", "semantic", "compile", "termination"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for failure in failures:
            trace = dict(failure.get("trace", {}))
            risk = trace.get("risk_level", "")
            risk_score = trace.get("risk_score", 0.0)
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{failure.get('method', '')}`",
                        f"`{failure.get('sample_id', '')}`",
                        f"`{','.join(failure.get('failure_reasons', []))}`",
                        f"`{trace.get('operator_name', '')}`",
                        f"`{trace.get('action', '')}`",
                        f"`{risk}:{risk_score}`",
                        f"`{failure.get('semantic_score', 0.0)}`",
                        f"`{failure.get('compile_succeeded', '')}`",
                        f"`{failure.get('termination_reason', '')}`",
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Compile Error Tails"])
    compile_failures = [
        failure
        for failure in failures
        if not bool(dict(failure.get("trace", {})).get("compile_succeeded", failure.get("compile_succeeded", True)))
    ]
    if not compile_failures:
        lines.append("- No compile failures found.")
    for failure in compile_failures:
        trace = dict(failure.get("trace", {}))
        stderr_tail = str(trace.get("compile_stderr_tail", "")).strip()
        lines.append(f"### `{failure.get('method', '')}` / `{failure.get('sample_id', '')}`")
        lines.append(f"- Action: `{trace.get('action', '')}`")
        lines.append(f"- Risk: `{trace.get('risk_level', '')}:{trace.get('risk_score', 0.0)}` `{trace.get('risk_reasons', [])}`")
        lines.append("```text")
        lines.append(stderr_tail or "(empty stderr)")
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    baseline_paths = [resolve_path(path) for path in args.baseline_summary]
    ppo_run_dirs = [resolve_path(path) for path in args.ppo_run_dir]
    for baseline_path in baseline_paths:
        failures.extend(audit_baseline_summary(baseline_path, float(args.semantic_threshold)))
    for ppo_run_dir in ppo_run_dirs:
        failures.extend(audit_ppo_run(ppo_run_dir, float(args.semantic_threshold)))
    return {
        "created_at": utc_timestamp(),
        "semantic_threshold": float(args.semantic_threshold),
        "baseline_summary_path": ", ".join(project_relative(path) for path in baseline_paths),
        "ppo_run_dir": ", ".join(project_relative(path) for path in ppo_run_dirs),
        "summary": summarize_failures(failures),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    report = build_report(args)
    json_path = output_dir / "failure_audit.json"
    md_path = output_dir / "failure_audit.md"
    dump_json(json_path, report)
    write_text(md_path, render_markdown(report))
    print(f"Wrote failure audit JSON: {json_path}")
    print(f"Wrote failure audit Markdown: {md_path}")


if __name__ == "__main__":
    main()
