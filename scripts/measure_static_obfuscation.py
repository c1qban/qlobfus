from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import zlib
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text


C_KEYWORDS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Bool",
    "_Complex",
    "_Imaginary",
}


TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*'|"
    r"\b[A-Za-z_][A-Za-z0-9_]*\b|"
    r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|"
    r"==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|->|"
    r"[{}()\[\];,?:+\-*/%<>=!&|^~.]"
)


ROW_FIELDS = [
    "method",
    "variant",
    "seed",
    "sample_id",
    "source_kind",
    "original_source",
    "candidate_source",
    "token_count_original",
    "token_count_candidate",
    "token_count_ratio",
    "line_count_ratio",
    "char_count_ratio",
    "token_edit_distance_ratio",
    "normalized_attacker_similarity",
    "attacker_resistance_score",
    "identifier_jaccard_distance",
    "identifier_entropy_delta",
    "cyclomatic_delta",
    "max_nesting_delta",
    "compression_ratio_delta",
    "static_obfuscation_score",
]


MAX_SEQUENCE_MATCHER_CELLS = 250_000
MAX_METRIC_SOURCE_CHARS = 1_000_000


def _read_metric_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(MAX_METRIC_SOURCE_CHARS)


AGG_FIELDS = [
    "method",
    "variant",
    "seed",
    "row_count",
    "mean_static_obfuscation_score",
    "std_static_obfuscation_score",
    "mean_attacker_resistance_score",
    "mean_token_edit_distance_ratio",
    "mean_identifier_jaccard_distance",
    "mean_cyclomatic_delta",
    "mean_max_nesting_delta",
    "mean_compression_ratio_delta",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure source-level obfuscation strength beyond LLVM proxy metrics.")
    parser.add_argument("--baseline-summary", action="append", default=[], help="Path to run_baselines baseline_summary.json.")
    parser.add_argument("--ppo-run-dir", action="append", default=[], help="Path to a PPO run directory.")
    parser.add_argument("--output-dir", default="artifacts/static_obfuscation_metrics")
    parser.add_argument("--job-name", default="static_obfuscation_metrics")
    return parser.parse_args(argv)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//.*", " ", text)
    return text


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(_strip_comments(text))


def _identifiers(tokens: list[str]) -> list[str]:
    return [token for token in tokens if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) and token not in C_KEYWORDS]


def _identifier_entropy(identifiers: list[str]) -> float:
    if not identifiers:
        return 0.0
    counts = Counter(identifiers)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - (len(left & right) / len(left | right))


def _normalize_for_attacker(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    identifier_ids: dict[str, str] = {}
    for token in tokens:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            if token in C_KEYWORDS:
                normalized.append(token)
            else:
                if token not in identifier_ids:
                    identifier_ids[token] = f"ID{len(identifier_ids)}"
                normalized.append(identifier_ids[token])
        elif re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", token):
            normalized.append("NUM")
        elif token.startswith('"') or token.startswith("'"):
            normalized.append("STR")
        else:
            normalized.append(token)
    return normalized


def _bounded_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) * len(right) <= MAX_SEQUENCE_MATCHER_CELLS:
        return SequenceMatcher(a=left, b=right, autojunk=False).ratio()
    left_counts = Counter(left)
    right_counts = Counter(right)
    overlap = sum(min(count, right_counts.get(token, 0)) for token, count in left_counts.items())
    return (2.0 * overlap) / (len(left) + len(right))


def _cyclomatic_proxy(tokens: list[str]) -> int:
    decision_tokens = {"if", "for", "while", "case", "&&", "||", "?"}
    return 1 + sum(1 for token in tokens if token in decision_tokens)


def _max_brace_nesting(tokens: list[str]) -> int:
    depth = 0
    max_depth = 0
    for token in tokens:
        if token == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif token == "}":
            depth = max(0, depth - 1)
    return max_depth


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="replace")
    if not raw:
        return 0.0
    return len(zlib.compress(raw)) / len(raw)


def _safe_ratio(candidate: float, original: float) -> float:
    if original <= 0:
        return 0.0 if candidate <= 0 else 1.0
    return candidate / original


def _clamp01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def measure_pair(original_text: str, candidate_text: str) -> dict[str, float]:
    original_tokens = _tokens(original_text)
    candidate_tokens = _tokens(candidate_text)
    original_ids = _identifiers(original_tokens)
    candidate_ids = _identifiers(candidate_tokens)
    token_similarity = _bounded_similarity(original_tokens, candidate_tokens)
    normalized_similarity = _bounded_similarity(
        _normalize_for_attacker(original_tokens),
        _normalize_for_attacker(candidate_tokens),
    )
    token_edit_distance_ratio = 1.0 - token_similarity
    attacker_resistance_score = 1.0 - normalized_similarity
    identifier_jaccard_distance = _jaccard_distance(set(original_ids), set(candidate_ids))
    cyclomatic_delta = abs(_cyclomatic_proxy(candidate_tokens) - _cyclomatic_proxy(original_tokens))
    max_nesting_delta = abs(_max_brace_nesting(candidate_tokens) - _max_brace_nesting(original_tokens))
    compression_ratio_delta = abs(_compression_ratio(candidate_text) - _compression_ratio(original_text))
    score = (
        0.30 * _clamp01(token_edit_distance_ratio)
        + 0.25 * _clamp01(attacker_resistance_score)
        + 0.20 * _clamp01(identifier_jaccard_distance)
        + 0.15 * _clamp01(cyclomatic_delta / 5.0)
        + 0.05 * _clamp01(max_nesting_delta / 3.0)
        + 0.05 * _clamp01(compression_ratio_delta)
    )
    return {
        "token_count_original": float(len(original_tokens)),
        "token_count_candidate": float(len(candidate_tokens)),
        "token_count_ratio": _safe_ratio(len(candidate_tokens), len(original_tokens)),
        "line_count_ratio": _safe_ratio(len(candidate_text.splitlines()), len(original_text.splitlines())),
        "char_count_ratio": _safe_ratio(len(candidate_text), len(original_text)),
        "token_edit_distance_ratio": token_edit_distance_ratio,
        "normalized_attacker_similarity": normalized_similarity,
        "attacker_resistance_score": attacker_resistance_score,
        "identifier_jaccard_distance": identifier_jaccard_distance,
        "identifier_entropy_delta": _identifier_entropy(candidate_ids) - _identifier_entropy(original_ids),
        "cyclomatic_delta": float(cyclomatic_delta),
        "max_nesting_delta": float(max_nesting_delta),
        "compression_ratio_delta": compression_ratio_delta,
        "static_obfuscation_score": score,
    }


def _load_dict(path_value: str | Path) -> dict[str, Any]:
    data = load_structured_file(path_value)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path_value}")
    return data


def _source_lookup_from_cache(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    data = _load_dict(path)
    lookup: dict[str, Path] = {}
    for sample in data.get("samples", []):
        if isinstance(sample, dict) and sample.get("sample_id") and sample.get("source_path"):
            lookup[str(sample["sample_id"])] = resolve_path(str(sample["source_path"]))
    return lookup


def _final_source_from_trace(trace_path: Path) -> Path | None:
    if not trace_path.exists():
        return None
    final_source = None
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        source_artifact = event.get("source_artifact")
        if source_artifact:
            final_source = resolve_path(str(source_artifact))
    return final_source if final_source and final_source.exists() else None


def _episode_rows(
    *,
    method: str,
    variant: str,
    seed: str,
    source_kind: str,
    episodes: list[dict[str, Any]],
    source_lookup: dict[str, Path],
    candidate_lookup: dict[str, Path] | None = None,
    progress_label: str = "",
    progress_every: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start_time = time.time()
    total = len(episodes)
    for index, episode in enumerate(episodes, start=1):
        sample_id = str(episode.get("sample_id", ""))
        trace_text = str(episode.get("trace_path", ""))
        if not sample_id:
            continue
        if progress_label and progress_every > 0 and (index == 1 or index % progress_every == 1):
            print(
                f"[row-load] {progress_label} processing episode={index}/{total} sample={sample_id}",
                flush=True,
            )
        original_path = source_lookup.get(sample_id)
        candidate_path = _final_source_from_trace(resolve_path(trace_text)) if trace_text else None
        if candidate_path is None and candidate_lookup:
            candidate_path = candidate_lookup.get(sample_id)
        if original_path is None or candidate_path is None or not original_path.exists() or not candidate_path.exists():
            continue
        metrics = measure_pair(
            _read_metric_text(original_path),
            _read_metric_text(candidate_path),
        )
        row = {
            "method": method,
            "variant": variant,
            "seed": seed,
            "sample_id": sample_id,
            "source_kind": source_kind,
            "original_source": project_relative(original_path),
            "candidate_source": project_relative(candidate_path),
            **{key: round(value, 6) for key, value in metrics.items()},
        }
        rows.append(row)
        if progress_label and progress_every > 0 and (index % progress_every == 0 or index == total):
            elapsed = time.time() - start_time
            rate = index / elapsed if elapsed > 0 else 0.0
            remaining = (total - index) / rate if rate > 0 else 0.0
            print(
                f"[row-load] {progress_label} episodes={index}/{total} rows={len(rows)} "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                flush=True,
            )
    return rows


def _external_candidate_lookup(run_dir: Path, method: str, episodes: list[dict[str, Any]]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    root = run_dir / "artifacts" / "baselines" / method
    if not root.exists():
        return lookup
    for episode in episodes:
        sample_id = str(episode.get("sample_id", ""))
        if not sample_id:
            continue
        sample_dir = root / sample_id.replace("/", "_")
        if not sample_dir.exists():
            continue
        candidates = sorted(sample_dir.glob(f"*.{method}.c"))
        if not candidates:
            candidates = sorted(sample_dir.glob("*.c"))
        if candidates:
            lookup[sample_id] = candidates[0].resolve()
    return lookup


def _infer_seed_from_run(run_dir: Path) -> str:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = _load_dict(metadata_path)
        training = dict(metadata.get("training", {}))
        if "seed" in training:
            return str(training["seed"])
    match = re.search(r"seed(\d+)", run_dir.as_posix(), flags=re.I)
    return match.group(1) if match else ""


def _infer_variant_from_run(run_dir: Path) -> str:
    text = run_dir.as_posix().lower()
    if "no_action_safety" in text:
        return "no_action_safety"
    if "no_fuzz_verifier" in text:
        return "no_fuzz_verifier"
    if "no_llvm_reward" in text:
        return "no_llvm_reward"
    if "long" in text or "1536" in text:
        return "full_long"
    return "full_short"


def _collect_baseline_rows(summary_path: Path, *, progress_every: int = 0) -> list[dict[str, Any]]:
    summary = _load_dict(summary_path)
    run_dir = resolve_path(str(summary.get("run_dir", summary_path.parent.parent)))
    source_lookup = _source_lookup_from_cache(run_dir / "data" / "baseline_env_input_cache.json")
    rows: list[dict[str, Any]] = []
    for method, result in dict(summary.get("results", {})).items():
        episodes = [dict(item) for item in dict(result).get("episodes", []) if isinstance(item, dict)]
        if progress_every > 0:
            print(
                f"[row-load] baseline={method} episodes={len(episodes)} "
                f"summary={project_relative(summary_path)}",
                flush=True,
            )
        rows.extend(
            _episode_rows(
                method=str(method),
                variant="baseline",
                seed="",
                source_kind="baseline_summary",
                episodes=episodes,
                source_lookup=source_lookup,
                candidate_lookup=_external_candidate_lookup(run_dir, str(method), episodes),
                progress_label=f"baseline:{method}",
                progress_every=progress_every,
            )
        )
    return rows


def _collect_ppo_rows(run_dir: Path, *, progress_every: int = 0) -> list[dict[str, Any]]:
    source_lookup = _source_lookup_from_cache(run_dir / "data" / "env_input_cache.json")
    log_path = run_dir / "logs" / "episode_metrics.jsonl"
    frozen_episodes_path = run_dir / "reports" / "episodes.jsonl"
    episodes: list[dict[str, Any]] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    episodes.append(item)
    elif frozen_episodes_path.exists():
        for line in frozen_episodes_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    episodes.append(item)
    if progress_every > 0:
        print(
            f"[row-load] ppo episodes={len(episodes)} run_dir={project_relative(run_dir)}",
            flush=True,
        )
    return _episode_rows(
        method="maskable_ppo",
        variant=_infer_variant_from_run(run_dir),
        seed=_infer_seed_from_run(run_dir),
        source_kind="ppo_frozen_episodes" if frozen_episodes_path.exists() and not log_path.exists() else "ppo_episode_metrics",
        episodes=episodes,
        source_lookup=source_lookup,
        progress_label="ppo:maskable_ppo",
        progress_every=progress_every,
    )


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["variant"]), str(row["seed"]))].append(row)
    aggregates: list[dict[str, Any]] = []
    for (method, variant, seed), items in sorted(grouped.items()):
        def values(name: str) -> list[float]:
            return [float(item.get(name, 0.0)) for item in items]

        aggregates.append(
            {
                "method": method,
                "variant": variant,
                "seed": seed,
                "row_count": len(items),
                "mean_static_obfuscation_score": round(_safe_mean(values("static_obfuscation_score")), 6),
                "std_static_obfuscation_score": round(_safe_std(values("static_obfuscation_score")), 6),
                "mean_attacker_resistance_score": round(_safe_mean(values("attacker_resistance_score")), 6),
                "mean_token_edit_distance_ratio": round(_safe_mean(values("token_edit_distance_ratio")), 6),
                "mean_identifier_jaccard_distance": round(_safe_mean(values("identifier_jaccard_distance")), 6),
                "mean_cyclomatic_delta": round(_safe_mean(values("cyclomatic_delta")), 6),
                "mean_max_nesting_delta": round(_safe_mean(values("max_nesting_delta")), 6),
                "mean_compression_ratio_delta": round(_safe_mean(values("compression_ratio_delta")), 6),
            }
        )
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Static Obfuscation Metrics",
        "",
        f"- job: `{payload['job_name']}`",
        f"- generated_at: `{payload['generated_at']}`",
        f"- row_count: `{payload['row_count']}`",
        "",
        "## Aggregates",
        "",
        "| method | variant | seed | rows | static score | attacker resistance | token edit | identifier distance | cyclomatic delta | nesting delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["aggregates"]:
        lines.append(
            "| {method} | {variant} | {seed} | {row_count} | {mean_static_obfuscation_score:.6f} +/- {std_static_obfuscation_score:.6f} | "
            "{mean_attacker_resistance_score:.6f} | {mean_token_edit_distance_ratio:.6f} | {mean_identifier_jaccard_distance:.6f} | "
            "{mean_cyclomatic_delta:.6f} | {mean_max_nesting_delta:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `static_obfuscation_score` combines token edit distance, simple normalization-resistant distance, identifier-set distance, cyclomatic delta, nesting delta, and compression-ratio delta.",
            "- `attacker_resistance_score` is a lightweight normalization attacker proxy: identifiers, literals, and strings are normalized before similarity is computed.",
            "- These metrics are stronger than a single LLVM proxy, but they are still not a full real-world reverse-engineering attack.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    rows: list[dict[str, Any]] = []
    for summary in args.baseline_summary:
        rows.extend(_collect_baseline_rows(resolve_path(summary)))
    for run_dir in args.ppo_run_dir:
        rows.extend(_collect_ppo_rows(resolve_path(run_dir)))
    aggregates = _aggregate(rows)
    payload = {
        "job_name": args.job_name,
        "generated_at": utc_timestamp(),
        "row_count": len(rows),
        "rows": rows,
        "aggregates": aggregates,
    }
    dump_json(output_dir / "static_obfuscation_metrics.json", payload)
    _write_csv(output_dir / "static_obfuscation_rows.csv", rows, ROW_FIELDS)
    _write_csv(output_dir / "static_obfuscation_aggregates.csv", aggregates, AGG_FIELDS)
    write_text(output_dir / "static_obfuscation_metrics.md", _markdown(payload))
    print(f"Wrote static obfuscation metrics: {output_dir / 'static_obfuscation_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
