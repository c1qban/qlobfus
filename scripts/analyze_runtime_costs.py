from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


DEFAULT_OUTPUT_DIR = "artifacts/formal_eval_150_harness_clean/runtime_costs"
DEFAULT_BASELINE_SUMMARIES = [
    "artifacts/formal_eval_150_harness_clean/fe150_baselines_20260519_201421/reports/baseline_summary.json",
    "artifacts/formal_eval_150_harness_clean/fe150_tigress_20260519_220111/reports/baseline_summary.json",
]
DEFAULT_PPO_RUN_DIRS = [
    "artifacts/formal_eval_150_harness_clean/multiseed/mppo_seed1",
    "artifacts/formal_eval_150_harness_clean/multiseed/mppo_seed7",
    "artifacts/formal_eval_150_harness_clean/multiseed/mppo_seed13",
]


CSV_FIELDS = [
    "method",
    "variant",
    "seed",
    "source_path",
    "sample_count",
    "episode_count",
    "completed_count",
    "total_steps",
    "wall_clock_span_sec",
    "seconds_per_episode",
    "seconds_per_sample",
    "verification_calls",
    "cache_hits",
    "cache_hit_rate",
    "cache_file_count",
    "compile_attempts",
    "compile_successes",
    "test_case_executions",
    "diff_case_executions",
    "fuzz_case_executions",
    "semantic_failures",
    "tool_failures",
    "notes",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze runtime, verifier, and cache costs for experiment artifacts.")
    parser.add_argument("--baseline-summary", action="append", default=list(DEFAULT_BASELINE_SUMMARIES))
    parser.add_argument("--ppo-run-dir", action="append", default=list(DEFAULT_PPO_RUN_DIRS))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--semantic-threshold", type=float, default=0.97)
    parser.add_argument("--job-name", default="formal_eval_150_runtime_costs")
    return parser.parse_args(argv)


def _load_dict(path: Path) -> dict[str, Any]:
    data = load_structured_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON/YAML object in {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _parse_utc(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_span_seconds(root: Path) -> float:
    if not root.exists():
        return 0.0
    mtimes: list[float] = []
    if root.is_file():
        mtimes.append(root.stat().st_mtime)
    else:
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    mtimes.append(path.stat().st_mtime)
                except OSError:
                    continue
    if len(mtimes) < 2:
        return 0.0
    return max(mtimes) - min(mtimes)


def _timestamp_from_text(value: str) -> datetime | None:
    match = re.search(r"(20\d{6})_(\d{6})(?:_\d+)?", value)
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _trace_timestamp_span_seconds(episodes: list[dict[str, Any]]) -> float:
    timestamps: list[datetime] = []
    for episode in episodes:
        trace_path = str(episode.get("trace_path", ""))
        parsed = _timestamp_from_text(trace_path)
        if parsed is not None:
            timestamps.append(parsed)
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)).total_seconds()


def _wall_span_with_source(root: Path, episodes: list[dict[str, Any]]) -> tuple[float, str]:
    trace_span = _trace_timestamp_span_seconds(episodes)
    if trace_span > 0:
        return trace_span, "trace_path timestamps"
    file_span = _file_span_seconds(root)
    if file_span > 1.0:
        return file_span, "artifact file mtimes"
    return 0.0, "unavailable"


def _verification_from_event(event: dict[str, Any]) -> dict[str, Any]:
    verification = event.get("verification", {})
    return dict(verification) if isinstance(verification, dict) else {}


def _verification_stats_from_summary(summary: dict[str, Any]) -> dict[str, int]:
    if not summary:
        return {
            "verification_calls": 0,
            "cache_hits": 0,
            "compile_attempts": 0,
            "compile_successes": 0,
            "test_case_executions": 0,
            "diff_case_executions": 0,
            "fuzz_case_executions": 0,
        }

    notes = [str(note) for note in summary.get("notes", [])]
    compile_result = dict(summary.get("compile", {}))
    tests = dict(summary.get("tests", {}))
    diff = dict(summary.get("diff", {}))
    fuzz = dict(summary.get("fuzz", {}))
    return {
        "verification_calls": 1,
        "cache_hits": 1 if any("Loaded verification result from cache." in note for note in notes) else 0,
        "compile_attempts": 1 if compile_result else 0,
        "compile_successes": 1 if bool(compile_result.get("succeeded", False)) else 0,
        "test_case_executions": len(tests.get("cases", [])),
        "diff_case_executions": len(diff.get("cases", [])) if bool(diff.get("executed", False)) else 0,
        "fuzz_case_executions": len(fuzz.get("cases", [])) if bool(fuzz.get("executed", False)) else 0,
    }


def _sum_stats(items: list[dict[str, int]]) -> dict[str, int]:
    keys = [
        "verification_calls",
        "cache_hits",
        "compile_attempts",
        "compile_successes",
        "test_case_executions",
        "diff_case_executions",
        "fuzz_case_executions",
    ]
    return {key: sum(int(item.get(key, 0)) for item in items) for key in keys}


def _trace_verification_stats(trace_path: str | Path) -> dict[str, int]:
    path = resolve_path(trace_path)
    if not path.exists():
        return _sum_stats([])
    stats: list[dict[str, int]] = []
    for event in _read_jsonl(path):
        verification = _verification_from_event(event)
        if verification:
            stats.append(_verification_stats_from_summary(verification))
    return _sum_stats(stats)


def _episode_trace_stats(episodes: list[dict[str, Any]]) -> dict[str, int]:
    stats = []
    seen: set[str] = set()
    for episode in episodes:
        trace_path = str(episode.get("trace_path", ""))
        if not trace_path or trace_path in seen:
            continue
        seen.add(trace_path)
        stats.append(_trace_verification_stats(trace_path))
    return _sum_stats(stats)


def _baseline_verification_stats(episodes: list[dict[str, Any]]) -> dict[str, int]:
    stats: list[dict[str, int]] = []
    for episode in episodes:
        summary = episode.get("verification_summary", {})
        if isinstance(summary, dict) and summary:
            stats.append(_verification_stats_from_summary(dict(summary)))
            continue
        trace_path = str(episode.get("trace_path", ""))
        if trace_path:
            stats.append(_trace_verification_stats(trace_path))
    return _sum_stats(stats)


def _cache_file_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("_verifier_cache/*.json"))


def _completed_count(episodes: list[dict[str, Any]]) -> int:
    return sum(1 for episode in episodes if str(episode.get("status", "completed")) == "completed")


def _semantic_failures(episodes: list[dict[str, Any]], threshold: float) -> int:
    return sum(1 for episode in episodes if float(episode.get("semantic_score", 0.0)) < threshold)


def _tool_failures(episodes: list[dict[str, Any]]) -> int:
    return sum(1 for episode in episodes if str(episode.get("status", "")) not in {"", "completed"})


def _with_rates(row: dict[str, Any]) -> dict[str, Any]:
    episode_count = int(row.get("episode_count", 0))
    sample_count = int(row.get("sample_count", 0))
    wall = float(row.get("wall_clock_span_sec", 0.0))
    verification_calls = int(row.get("verification_calls", 0))
    cache_hits = int(row.get("cache_hits", 0))
    row["seconds_per_episode"] = _round(wall / episode_count) if episode_count and wall else 0.0
    row["seconds_per_sample"] = _round(wall / sample_count) if sample_count and wall else 0.0
    row["cache_hit_rate"] = _round(cache_hits / verification_calls) if verification_calls else 0.0
    return row


def _summarize_ppo_run(run_dir: Path, semantic_threshold: float) -> dict[str, Any]:
    runtime = _load_dict(run_dir / "training_runtime.json")
    callback = dict(runtime.get("callback_summary", {}))
    corrected_path = run_dir / "logs" / "episode_metrics.corrected.jsonl"
    episode_path = corrected_path if corrected_path.exists() else resolve_path(str(callback.get("episode_metrics_path", "")))
    episodes = _read_jsonl(episode_path)
    trace_stats = _episode_trace_stats(episodes)
    wall, wall_source = _wall_span_with_source(run_dir, episodes)
    row = {
        "method": str(runtime.get("algorithm", "maskable_ppo")),
        "variant": "full_short",
        "seed": str(runtime.get("seed", "")),
        "source_path": project_relative(run_dir),
        "sample_count": int(runtime.get("sample_count", 0)),
        "episode_count": len(episodes) or int(callback.get("episode_count", 0)),
        "completed_count": len(episodes) or int(callback.get("episode_count", 0)),
        "total_steps": int(callback.get("observed_step_count", runtime.get("total_timesteps", 0)) or 0),
        "wall_clock_span_sec": _round(wall),
        "cache_file_count": _cache_file_count(run_dir),
        "semantic_failures": _semantic_failures(episodes, semantic_threshold),
        "tool_failures": _tool_failures(episodes),
        "notes": f"episode_metrics={project_relative(episode_path)}; wall_clock_span_source={wall_source}",
        **trace_stats,
    }
    return _with_rates(row)


def _summarize_baseline_result(
    *,
    summary_path: Path,
    method: str,
    result: dict[str, Any],
    semantic_threshold: float,
) -> dict[str, Any]:
    run_dir = summary_path.parent.parent
    episodes = [dict(item) for item in result.get("episodes", []) if isinstance(item, dict)]
    wall, wall_source = _wall_span_with_source(run_dir, episodes)
    verification_stats = _baseline_verification_stats(episodes)
    row = {
        "method": method,
        "variant": "baseline",
        "seed": "",
        "source_path": project_relative(summary_path),
        "sample_count": int(result.get("episode_count", len(episodes)) or len(episodes)),
        "episode_count": len(episodes),
        "completed_count": int(result.get("completed_episode_count", _completed_count(episodes))),
        "total_steps": sum(int(episode.get("step_index", 0) or episode.get("episode_length", 0) or 0) for episode in episodes),
        "wall_clock_span_sec": _round(max(wall, 0.0)),
        "cache_file_count": _cache_file_count(run_dir),
        "semantic_failures": _semantic_failures(episodes, semantic_threshold),
        "tool_failures": _tool_failures(episodes),
        "notes": f"wall_clock_span_source={wall_source}",
        **verification_stats,
    }
    return _with_rates(row)


def build_report(
    *,
    baseline_summaries: list[str],
    ppo_run_dirs: list[str],
    semantic_threshold: float,
    job_name: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir_text in ppo_run_dirs:
        run_dir = resolve_path(run_dir_text)
        if (run_dir / "training_runtime.json").exists():
            rows.append(_summarize_ppo_run(run_dir, semantic_threshold))

    for summary_text in baseline_summaries:
        summary_path = resolve_path(summary_text)
        if not summary_path.exists():
            continue
        summary = _load_dict(summary_path)
        for method, raw_result in sorted(dict(summary.get("results", {})).items()):
            if isinstance(raw_result, dict):
                rows.append(
                    _summarize_baseline_result(
                        summary_path=summary_path,
                        method=str(method),
                        result=dict(raw_result),
                        semantic_threshold=semantic_threshold,
                    )
                )

    totals = {
        "row_count": len(rows),
        "total_wall_clock_span_sec": _round(sum(float(row.get("wall_clock_span_sec", 0.0)) for row in rows)),
        "total_verification_calls": sum(int(row.get("verification_calls", 0)) for row in rows),
        "total_cache_hits": sum(int(row.get("cache_hits", 0)) for row in rows),
        "total_compile_attempts": sum(int(row.get("compile_attempts", 0)) for row in rows),
        "total_test_case_executions": sum(int(row.get("test_case_executions", 0)) for row in rows),
        "total_diff_case_executions": sum(int(row.get("diff_case_executions", 0)) for row in rows),
        "total_fuzz_case_executions": sum(int(row.get("fuzz_case_executions", 0)) for row in rows),
        "mean_seconds_per_episode": _round(_safe_mean([float(row.get("seconds_per_episode", 0.0)) for row in rows])),
    }
    totals["overall_cache_hit_rate"] = _round(totals["total_cache_hits"] / totals["total_verification_calls"]) if totals["total_verification_calls"] else 0.0

    return {
        "job_name": job_name,
        "generated_at": utc_timestamp(),
        "semantic_threshold": semantic_threshold,
        "methodology": {
            "wall_clock_span_sec": "Estimated from trace-path timestamps when available; otherwise from artifact mtimes or marked unavailable.",
            "verification_calls": "Counted from transform traces or embedded verification_summary objects.",
            "cache_hits": "Counted when verification notes contain 'Loaded verification result from cache.'.",
        },
        "summary": totals,
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _markdown(report: dict[str, Any]) -> str:
    summary = dict(report["summary"])
    lines = [
        "# Runtime Cost Report",
        "",
        f"- job: `{report['job_name']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- rows: `{summary['row_count']}`",
        f"- total_wall_clock_span_sec: `{summary['total_wall_clock_span_sec']}`",
        f"- total_verification_calls: `{summary['total_verification_calls']}`",
        f"- total_compile_attempts: `{summary['total_compile_attempts']}`",
        f"- total_test_case_executions: `{summary['total_test_case_executions']}`",
        f"- total_diff_case_executions: `{summary['total_diff_case_executions']}`",
        f"- total_fuzz_case_executions: `{summary['total_fuzz_case_executions']}`",
        f"- overall_cache_hit_rate: `{summary['overall_cache_hit_rate']}`",
        "",
        "## Methodology",
        "",
        "- `wall_clock_span_sec` is estimated from trace-path timestamps when available; otherwise from artifact mtimes or marked unavailable.",
        "- `verification_calls` are counted from transform traces or embedded `verification_summary` objects.",
        "- `cache_hits` are counted from verifier notes that explicitly say the result was loaded from cache.",
        "",
        "## Rows",
        "",
        "| method | seed | episodes | span sec | sec/episode | verifier calls | cache hit rate | compile | tests | diff | fuzz | semantic failures | tool failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {method} | {seed} | {episode_count} | {wall_clock_span_sec:.3f} | {seconds_per_episode:.3f} | "
            "{verification_calls} | {cache_hit_rate:.3f} | {compile_attempts} | {test_case_executions} | "
            "{diff_case_executions} | {fuzz_case_executions} | {semantic_failures} | {tool_failures} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    report = build_report(
        baseline_summaries=list(args.baseline_summary),
        ppo_run_dirs=list(args.ppo_run_dir),
        semantic_threshold=float(args.semantic_threshold),
        job_name=str(args.job_name),
    )
    dump_json(output_dir / "runtime_cost_report.json", report)
    _write_csv(output_dir / "runtime_cost_rows.csv", list(report["rows"]))
    write_text(output_dir / "runtime_cost_report.md", _markdown(report))
    print(f"Wrote runtime cost report: {output_dir / 'runtime_cost_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
