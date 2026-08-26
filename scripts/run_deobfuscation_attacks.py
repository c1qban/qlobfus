from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .common import dump_json, ensure_dir, load_structured_file, project_relative, resolve_path, utc_timestamp, write_text
from .measure_static_obfuscation import (
    AGG_FIELDS,
    ROW_FIELDS,
    _collect_baseline_rows,
    _collect_ppo_rows,
    MAX_METRIC_SOURCE_CHARS,
    _tokens,
    measure_pair,
)


ATTACK_ROW_FIELDS = ROW_FIELDS + [
    "attack",
    "attack_status",
    "attack_skip_reason",
    "pre_attack_static_obfuscation_score",
    "post_attack_static_obfuscation_score",
    "score_retention_ratio",
    "residual_attacker_resistance_score",
]


ATTACK_AGG_FIELDS = [
    "method",
    "variant",
    "seed",
    "attack",
    "attempted_count",
    "row_count",
    "skipped_count",
    "mean_pre_attack_static_obfuscation_score",
    "mean_post_attack_static_obfuscation_score",
    "std_post_attack_static_obfuscation_score",
    "mean_score_retention_ratio",
    "mean_residual_attacker_resistance_score",
    "mean_token_edit_distance_ratio",
    "mean_identifier_jaccard_distance",
]


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight deobfuscation attacks and measure residual strength.")
    parser.add_argument("--baseline-summary", action="append", default=[], help="Path to run_baselines baseline_summary.json.")
    parser.add_argument("--ppo-run-dir", action="append", default=[], help="Path to a PPO run directory.")
    parser.add_argument("--output-dir", default="artifacts/deobfuscation_attacks")
    parser.add_argument("--job-name", default="lightweight_deobfuscation_attacks")
    parser.add_argument(
        "--attack",
        action="append",
        choices=[
            "normalize",
            "strip_dead_blocks",
            "normalize_strip",
            "cfg_cleanup",
            "normalize_cfg_cleanup",
            "llvm_o2_ir",
        ],
        default=[],
        help="Attack to run. Defaults to normalize and normalize_strip.",
    )
    parser.add_argument("--compiler", default="", help="Compiler for llvm_o2_ir attack. Defaults to clang/clang.exe lookup.")
    parser.add_argument("--compiler-timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--llvm-max-source-bytes",
        type=int,
        default=5_000_000,
        help="Skip llvm_o2_ir for candidates larger than this many bytes. Use 0 to disable.",
    )
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N source rows. Use 0 to disable.")
    return parser.parse_args(argv)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//.*", " ", text)
    return text


def _canonicalize_tokens(text: str) -> str:
    tokens = _tokens(text)
    identifiers: dict[str, str] = {}
    out: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            if token in C_KEYWORDS:
                out.append(token)
            else:
                if token not in identifiers:
                    identifiers[token] = f"id_{len(identifiers)}"
                out.append(identifiers[token])
        elif re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", token):
            out.append("0")
        elif token.startswith('"'):
            out.append('"S"')
        elif token.startswith("'"):
            out.append("'c'")
        else:
            out.append(token)
    return " ".join(out)


def _strip_dead_blocks(text: str) -> str:
    text = _strip_comments(text)
    patterns = [
        r"\bif\s*\(\s*0\s*\)\s*\{[^{}]*\}",
        r"\bwhile\s*\(\s*0\s*\)\s*\{[^{}]*\}",
        r"\bfor\s*\(\s*;\s*0\s*;\s*\)\s*\{[^{}]*\}",
    ]
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            text, count = re.subn(pattern, " ", text, flags=re.S)
            changed = changed or count > 0
    return text


def _cleanup_cfg_patterns(text: str) -> str:
    text = _strip_dead_blocks(text)
    text = re.sub(r"\bvolatile\s+", " ", text)
    text = re.sub(r"\bgoto\s+[A-Za-z_][A-Za-z0-9_]*\s*;", " ", text)
    text = re.sub(r"^[ \t]*[A-Za-z_][A-Za-z0-9_]*\s*:\s*", " ", text, flags=re.M)
    text = re.sub(r"\bswitch\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{", r"switch(\1){", text)
    text = re.sub(r"\bif\s*\(\s*(?:1|true)\s*\)\s*\{([^{}]*)\}", r" \1 ", text, flags=re.S)
    text = re.sub(r"\bif\s*\(\s*(?:0|false)\s*\)\s*\{[^{}]*\}\s*else\s*\{([^{}]*)\}", r" \1 ", text, flags=re.S)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def apply_attack(text: str, attack: str) -> str:
    if attack == "normalize":
        return _canonicalize_tokens(text)
    if attack == "strip_dead_blocks":
        return _strip_dead_blocks(text)
    if attack == "normalize_strip":
        return _canonicalize_tokens(_strip_dead_blocks(text))
    if attack == "cfg_cleanup":
        return _cleanup_cfg_patterns(text)
    if attack == "normalize_cfg_cleanup":
        return _canonicalize_tokens(_cleanup_cfg_patterns(text))
    raise ValueError(f"Unsupported attack: {attack}")


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in args.baseline_summary:
        summary_path = resolve_path(summary)
        print(f"[row-load] loading baseline summary={project_relative(summary_path)}", flush=True)
        before = len(rows)
        rows.extend(_collect_baseline_rows(summary_path, progress_every=args.progress_every))
        print(f"[row-load] loaded baseline rows={len(rows) - before} total_rows={len(rows)}", flush=True)
    for run_dir in args.ppo_run_dir:
        resolved_run_dir = resolve_path(run_dir)
        print(f"[row-load] loading ppo run_dir={project_relative(resolved_run_dir)}", flush=True)
        before = len(rows)
        rows.extend(_collect_ppo_rows(resolved_run_dir, progress_every=args.progress_every))
        print(f"[row-load] loaded ppo rows={len(rows) - before} total_rows={len(rows)}", flush=True)
    return rows


def _read(path_value: str, *, budgeted: bool = True) -> str:
    path = resolve_path(path_value)
    if not budgeted:
        return path.read_text(encoding="utf-8", errors="replace")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(MAX_METRIC_SOURCE_CHARS)


def _source_too_large(path_value: str, max_bytes: int) -> bool:
    if max_bytes <= 0:
        return False
    return resolve_path(path_value).stat().st_size > max_bytes


def _default_compiler(explicit: str) -> str:
    if explicit:
        return explicit
    env = os.environ.get("CC", "")
    if env:
        return env
    return "clang"


def _normalize_ir(ir_text: str) -> str:
    text = re.sub(r";.*", " ", ir_text)
    text = re.sub(r"source_filename\s*=\s*\"[^\"]*\"", "source_filename = \"SOURCE\"", text)
    text = re.sub(r"target\s+(?:triple|datalayout)\s*=\s*\"[^\"]*\"", " ", text)
    text = re.sub(r"![0-9]+", "!N", text)
    text = re.sub(r"%[A-Za-z0-9_.$-]+", "%v", text)
    text = re.sub(r"@[A-Za-z0-9_.$-]+", "@g", text)
    text = re.sub(r"\b[0-9]+\b", "0", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compile_to_llvm_ir(
    *,
    source_path: str,
    source_text: str,
    compiler: str,
    timeout_sec: float,
    cache_dir: Path,
) -> tuple[str, str, str]:
    digest = hashlib.sha256((compiler + "\0" + source_text).encode("utf-8", errors="replace")).hexdigest()
    cache_path = cache_dir / f"{digest}.ll"
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8", errors="replace")
        if text.startswith("LLVM_O2_IR_UNAVAILABLE"):
            return text, "skipped", text.split(" ", 1)[0]
        if text.startswith("LLVM_O2_IR_COMPILE_FAILED"):
            return text, "skipped", "compile_failed"
        return text, "ok", ""
    ensure_dir(cache_dir)
    suffix = Path(source_path).suffix or ".c"
    tmp_dir = ensure_dir(cache_dir / f"tmp_{digest[:16]}")
    tmp_source = tmp_dir / f"input{suffix}"
    tmp_source.write_text(source_text, encoding="utf-8", errors="replace")
    command = [
        compiler,
        "-O2",
        "-S",
        "-emit-llvm",
        "-x",
        "c",
        str(tmp_source),
        "-o",
        "-",
        "-Wno-everything",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        fallback = f"LLVM_O2_IR_UNAVAILABLE {type(exc).__name__}"
        cache_path.write_text(fallback, encoding="utf-8")
        return fallback, "skipped", type(exc).__name__
    if completed.returncode != 0:
        text = "LLVM_O2_IR_COMPILE_FAILED " + completed.stderr[-1000:]
        status = "skipped"
        reason = "compile_failed"
    else:
        text = _normalize_ir(completed.stdout)
        status = "ok"
        reason = ""
    cache_path.write_text(text, encoding="utf-8")
    return text, status, reason


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def run_attacks(
    rows: list[dict[str, Any]],
    attacks: list[str],
    *,
    compiler: str = "",
    compiler_timeout_sec: float = 10.0,
    llvm_max_source_bytes: int = 5_000_000,
    cache_dir: Path | None = None,
    progress_every: int = 25,
) -> list[dict[str, Any]]:
    attacked_rows: list[dict[str, Any]] = []
    ir_cache_dir = ensure_dir(cache_dir or resolve_path("artifacts/deobfuscation_attacks/_llvm_o2_cache"))
    total_rows = len(rows)
    total_outputs = total_rows * len(attacks)
    started_at = time.monotonic()
    if progress_every:
        print(
            f"[deobfuscation] running attacks={','.join(attacks)} rows={total_rows} outputs={total_outputs}",
            flush=True,
        )
    for row_index, row in enumerate(rows, start=1):
        budgeted_read = "llvm_o2_ir" not in attacks
        original_text = _read(str(row["original_source"]), budgeted=budgeted_read)
        candidate_text = _read(str(row["candidate_source"]), budgeted=budgeted_read)
        pre_score = float(row.get("static_obfuscation_score", 0.0) or 0.0)
        for attack in attacks:
            status = "ok"
            skip_reason = ""
            if attack == "llvm_o2_ir":
                if _source_too_large(str(row["candidate_source"]), llvm_max_source_bytes):
                    status = "skipped"
                    skip_reason = "candidate_source_too_large"
                    metrics = {key: 0.0 for key in ROW_FIELDS if key not in {"method", "variant", "seed", "sample_id", "source_kind", "original_source", "candidate_source"}}
                    post_score = 0.0
                    attacked_rows.append(
                        {
                            **row,
                            **metrics,
                            "attack": attack,
                            "attack_status": status,
                            "attack_skip_reason": skip_reason,
                            "pre_attack_static_obfuscation_score": round(pre_score, 6),
                            "post_attack_static_obfuscation_score": post_score,
                            "score_retention_ratio": 0.0,
                            "residual_attacker_resistance_score": 0.0,
                        }
                    )
                    continue
                selected_compiler = _default_compiler(compiler)
                attacked_candidate, candidate_status, candidate_reason = _compile_to_llvm_ir(
                    source_path=str(row["candidate_source"]),
                    source_text=candidate_text,
                    compiler=selected_compiler,
                    timeout_sec=compiler_timeout_sec,
                    cache_dir=ir_cache_dir,
                )
                attacked_original, original_status, original_reason = _compile_to_llvm_ir(
                    source_path=str(row["original_source"]),
                    source_text=original_text,
                    compiler=selected_compiler,
                    timeout_sec=compiler_timeout_sec,
                    cache_dir=ir_cache_dir,
                )
                if candidate_status != "ok" or original_status != "ok":
                    status = "skipped"
                    skip_reason = candidate_reason or original_reason or "llvm_ir_unavailable"
            else:
                attacked_candidate = apply_attack(candidate_text, attack)
                attacked_original = (
                    apply_attack(original_text, attack)
                    if attack.startswith("normalize") or attack == "cfg_cleanup"
                    else original_text
                )
            if status == "ok":
                metrics = measure_pair(attacked_original, attacked_candidate)
                post_score = float(metrics["static_obfuscation_score"])
            else:
                metrics = {key: 0.0 for key in ROW_FIELDS if key not in {"method", "variant", "seed", "sample_id", "source_kind", "original_source", "candidate_source"}}
                post_score = 0.0
            attacked_rows.append(
                {
                    **row,
                    **{key: round(value, 6) for key, value in metrics.items()},
                    "attack": attack,
                    "attack_status": status,
                    "attack_skip_reason": skip_reason,
                    "pre_attack_static_obfuscation_score": round(pre_score, 6),
                    "post_attack_static_obfuscation_score": round(post_score, 6),
                    "score_retention_ratio": round(_safe_ratio(post_score, pre_score), 6),
                    "residual_attacker_resistance_score": round(float(metrics["attacker_resistance_score"]), 6),
                }
            )
        if progress_every and (row_index == 1 or row_index % progress_every == 0 or row_index == total_rows):
            elapsed = time.monotonic() - started_at
            rate = row_index / elapsed if elapsed > 0 else 0.0
            remaining = (total_rows - row_index) / rate if rate > 0 else 0.0
            print(
                "[deobfuscation] "
                f"rows={row_index}/{total_rows} "
                f"outputs={len(attacked_rows)}/{total_outputs} "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s "
                f"method={row.get('method','')} sample={row.get('sample_id','')}",
                flush=True,
            )
    return attacked_rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["variant"]), str(row["seed"]), str(row["attack"]))
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (method, variant, seed, attack), items in sorted(grouped.items()):
        ok_items = [item for item in items if str(item.get("attack_status", "ok")) == "ok"]

        def values(name: str) -> list[float]:
            return [float(item.get(name, 0.0) or 0.0) for item in ok_items]

        aggregates.append(
            {
                "method": method,
                "variant": variant,
                "seed": seed,
                "attack": attack,
                "attempted_count": len(items),
                "row_count": len(ok_items),
                "skipped_count": len(items) - len(ok_items),
                "mean_pre_attack_static_obfuscation_score": round(_safe_mean(values("pre_attack_static_obfuscation_score")), 6),
                "mean_post_attack_static_obfuscation_score": round(_safe_mean(values("post_attack_static_obfuscation_score")), 6),
                "std_post_attack_static_obfuscation_score": round(_safe_std(values("post_attack_static_obfuscation_score")), 6),
                "mean_score_retention_ratio": round(_safe_mean(values("score_retention_ratio")), 6),
                "mean_residual_attacker_resistance_score": round(_safe_mean(values("residual_attacker_resistance_score")), 6),
                "mean_token_edit_distance_ratio": round(_safe_mean(values("token_edit_distance_ratio")), 6),
                "mean_identifier_jaccard_distance": round(_safe_mean(values("identifier_jaccard_distance")), 6),
            }
        )
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Lightweight Deobfuscation Attack Metrics",
        "",
        f"- job: `{payload['job_name']}`",
        f"- generated_at: `{payload['generated_at']}`",
        f"- row_count: `{payload['row_count']}`",
        f"- attacks: `{', '.join(payload['attacks'])}`",
        "",
        "## Aggregates",
        "",
        "| method | variant | seed | attack | ok/attempted | skipped | pre score | post score | retention | residual resistance | token edit | identifier distance |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["aggregates"]:
        lines.append(
            "| {method} | {variant} | {seed} | {attack} | {row_count}/{attempted_count} | {skipped_count} | "
            "{mean_pre_attack_static_obfuscation_score:.6f} | {mean_post_attack_static_obfuscation_score:.6f} +/- {std_post_attack_static_obfuscation_score:.6f} | "
            "{mean_score_retention_ratio:.6f} | {mean_residual_attacker_resistance_score:.6f} | "
            "{mean_token_edit_distance_ratio:.6f} | {mean_identifier_jaccard_distance:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `normalize` canonicalizes identifiers, literals, strings, comments, and whitespace before measuring residual distance.",
            "- `strip_dead_blocks` removes simple syntactic dead blocks such as `if (0) { ... }` before measuring residual distance.",
            "- `cfg_cleanup` removes simple dead blocks, gotos, labels, volatile markers, and simple opaque predicates.",
            "- `llvm_o2_ir` compiles original and candidate sources to normalized LLVM IR using `clang -O2 -S -emit-llvm`, then measures residual IR-level distance.",
            "- These attacks are lightweight and deterministic. They do not replace a full human reverse-engineering or mature deobfuscation tool evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    attacks = list(args.attack) or ["normalize", "normalize_strip"]
    output_dir = ensure_dir(resolve_path(args.output_dir))
    print(f"[deobfuscation] output_dir={output_dir}", flush=True)
    print("[deobfuscation] loading source/candidate rows...", flush=True)
    static_rows = _load_rows(args)
    manifest = {
        "job_name": args.job_name,
        "started_at": utc_timestamp(),
        "attacks": attacks,
        "input_row_count": len(static_rows),
        "baseline_summary": args.baseline_summary,
        "ppo_run_dir": args.ppo_run_dir,
        "llvm_max_source_bytes": args.llvm_max_source_bytes,
        "progress_every": args.progress_every,
    }
    dump_json(output_dir / "deobfuscation_attack_manifest.json", manifest)
    print(
        f"[deobfuscation] loaded rows={len(static_rows)} manifest={output_dir / 'deobfuscation_attack_manifest.json'}",
        flush=True,
    )
    if not static_rows:
        print("[deobfuscation] warning: no rows were loaded; outputs will be empty.", flush=True)
    attacked_rows = run_attacks(
        static_rows,
        attacks,
        compiler=str(args.compiler),
        compiler_timeout_sec=float(args.compiler_timeout_sec),
        llvm_max_source_bytes=int(args.llvm_max_source_bytes),
        cache_dir=output_dir / "_llvm_o2_cache",
        progress_every=int(args.progress_every),
    )
    aggregates = aggregate(attacked_rows)
    payload = {
        "job_name": args.job_name,
        "generated_at": utc_timestamp(),
        "attacks": attacks,
        "row_count": len(attacked_rows),
        "rows": attacked_rows,
        "aggregates": aggregates,
    }
    dump_json(output_dir / "deobfuscation_attack_metrics.json", payload)
    _write_csv(output_dir / "deobfuscation_attack_rows.csv", attacked_rows, ATTACK_ROW_FIELDS)
    _write_csv(output_dir / "deobfuscation_attack_aggregates.csv", aggregates, ATTACK_AGG_FIELDS)
    write_text(output_dir / "deobfuscation_attack_metrics.md", _markdown(payload))
    print(f"Wrote deobfuscation attack metrics: {output_dir / 'deobfuscation_attack_metrics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
