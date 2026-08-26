from __future__ import annotations

import argparse

from eval.reporting import build_evaluation_summary
from .common import PROJECT_ROOT, copy_file, create_timestamped_dir, dump_json, ensure_dir, load_config, load_json_if_exists, project_relative, resolve_path, utc_timestamp
from verifier.pipeline import VerificationPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an evaluation scaffold for held-out security benchmarks.")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML config file.")
    parser.add_argument("--run-dir", help="Optional run directory to attach evaluation artifacts to.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path, config = load_config(args.config)

    job_name = str(config.get("job_name", "evaluation_run"))
    benchmark = config.get("benchmark", {})
    metrics = list(config.get("metrics", []))
    verification_config = config.get("verification", {})

    if args.run_dir:
        run_dir = resolve_path(args.run_dir, PROJECT_ROOT)
        reports_dir = ensure_dir(run_dir / "reports")
        metadata = load_json_if_exists(run_dir / "metadata.json")
    else:
        reports_dir = create_timestamped_dir(resolve_path("eval/reports", PROJECT_ROOT), job_name)
        run_dir = reports_dir
        metadata = None

    copy_file(config_path, reports_dir / f"evaluation.config{config_path.suffix}")

    semantic_threshold = 0.97
    if metadata is not None:
        semantic_threshold = float(metadata.get("environment", {}).get("semantic_threshold", semantic_threshold))

    pipeline = VerificationPipeline(
        semantic_threshold=semantic_threshold,
        compiler=verification_config.get("compiler"),
        compiler_flags=list(verification_config.get("compiler_flags", [])),
        compile_timeout_sec=float(verification_config.get("compile_timeout_sec", 10.0)),
    )
    if verification_config.get("manifest_path") and verification_config.get("sample_id"):
        verification = pipeline.verify_manifest_sample(
            manifest_path=verification_config["manifest_path"],
            sample_id=str(verification_config["sample_id"]),
        )
    elif verification_config.get("source_path"):
        verification = pipeline.verify_source(
            source_path=verification_config["source_path"],
            test_cases=list(verification_config.get("test_cases", [])),
            sample_id=str(verification_config.get("sample_id", job_name)),
            compiler=verification_config.get("compiler"),
            compiler_flags=list(verification_config.get("compiler_flags", [])),
            compile_timeout_sec=float(verification_config.get("compile_timeout_sec", 10.0)),
            build_dir=verification_config.get("build_dir"),
            workdir=verification_config.get("workdir"),
        )
    else:
        verification = pipeline.run_stub(sample_id=job_name)
    evaluation_summary = build_evaluation_summary(
        job_name=job_name,
        benchmark=benchmark,
        metrics=metrics,
        training_context=metadata.get("policy", {}) if metadata else {},
        verification=verification,
        run_dir=run_dir if args.run_dir else None,
    )
    evaluation_summary["created_at"] = utc_timestamp()
    evaluation_summary["status"] = "scaffolded"
    evaluation_summary["attached_run_dir"] = project_relative(run_dir)

    dump_json(reports_dir / "evaluation_summary.json", evaluation_summary)

    print(f"Created evaluation scaffold: {reports_dir}")


if __name__ == "__main__":
    main()
