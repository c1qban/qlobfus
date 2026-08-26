from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rl.trainers import detect_maskable_ppo_dependencies, run_maskable_ppo_training
from .common import (
    PROJECT_ROOT,
    copy_file,
    create_timestamped_dir,
    dump_json,
    ensure_dir,
    load_config,
    load_structured_file,
    project_relative,
    resolve_path,
    utc_timestamp,
    write_text,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a training run scaffold for the RL obfuscator.")
    parser.add_argument("--config", required=True, help="Path to a JSON-compatible YAML config file.")
    parser.add_argument("--run-dir", help="Optional explicit run directory.")
    parser.add_argument(
        "--split-index",
        help="Optional path to a prepared train/validation/test split index JSON file.",
    )
    return parser.parse_args(argv)


def load_split_index(path_value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_path(path_value, PROJECT_ROOT)
    payload = load_structured_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Split index must decode to an object: {path}")
    if not isinstance(payload.get("samples", []), list):
        raise ValueError(f"Split index samples must be a list: {path}")
    return path, payload


def load_harness_test_cases(harness_path: str) -> list[dict[str, Any]]:
    payload = load_structured_file(harness_path)
    if isinstance(payload, dict):
        test_cases = payload.get("test_cases", [])
    elif isinstance(payload, list):
        test_cases = payload
    else:
        raise ValueError(f"Harness payload must decode to an object or list: {harness_path}")

    if not isinstance(test_cases, list):
        raise ValueError(f"Harness test_cases must be a list: {harness_path}")

    return [dict(case) for case in test_cases]


def build_training_task(sample: dict[str, Any], position: int) -> dict[str, Any]:
    return {
        "task_id": f"task_{position:06d}_{sample.get('sample_id', 'unknown')}",
        "queue_position": position,
        "sample_id": str(sample.get("sample_id", "")),
        "split": str(sample.get("split", "")),
        "split_group": str(sample.get("split_group", "")),
        "source_path": str(sample.get("source_path", "")),
        "relative_source_path": str(sample.get("relative_source_path", "")),
        "source_stats": dict(sample.get("source_stats", {})),
        "compile": dict(sample.get("compile", {})),
        "harness": dict(sample.get("harness", {})),
    }


def build_env_cache_sample(sample: dict[str, Any]) -> dict[str, Any]:
    harness = dict(sample.get("harness", {}))
    harness_path = str(harness.get("path", ""))
    harness_cases = load_harness_test_cases(harness_path) if harness.get("exists") and harness_path else []

    return {
        "sample_id": str(sample.get("sample_id", "")),
        "split": str(sample.get("split", "")),
        "split_group": str(sample.get("split_group", "")),
        "source_path": str(sample.get("source_path", "")),
        "relative_source_path": str(sample.get("relative_source_path", "")),
        "source_stats": dict(sample.get("source_stats", {})),
        "compile": dict(sample.get("compile", {})),
        "verification_input": {
            "harness_format": str(harness.get("format", "")),
            "harness_path": harness_path,
            "case_count": int(harness.get("case_count", 0)),
            "test_cases": harness_cases,
        },
    }


def materialize_training_inputs(run_dir: Path, split_index: dict[str, Any]) -> dict[str, Any]:
    data_dir = ensure_dir(run_dir / "data")
    snapshot_path = data_dir / "split_index.snapshot.json"
    dump_json(snapshot_path, split_index)

    samples = [dict(sample) for sample in split_index.get("samples", [])]
    sample_ids = [str(sample.get("sample_id", "")) for sample in samples]
    write_text(data_dir / "sample_ids.txt", "\n".join(sample_ids) + ("\n" if sample_ids else ""))

    task_queue = [build_training_task(sample, position=index) for index, sample in enumerate(samples)]
    task_queue_path = data_dir / "train_tasks.jsonl"
    task_queue_text = "".join(json.dumps(task, sort_keys=True) + "\n" for task in task_queue)
    write_text(task_queue_path, task_queue_text)

    env_cache = {
        "job_name": str(split_index.get("job_name", "unknown")),
        "created_at": utc_timestamp(),
        "split": str(split_index.get("split", "unspecified")),
        "sample_count": int(split_index.get("sample_count", len(samples))),
        "samples": [build_env_cache_sample(sample) for sample in samples],
    }
    env_cache_path = data_dir / "env_input_cache.json"
    dump_json(env_cache_path, env_cache)

    return {
        "split_index_snapshot_path": project_relative(snapshot_path),
        "sample_ids_path": project_relative(data_dir / "sample_ids.txt"),
        "task_queue_path": project_relative(task_queue_path),
        "env_input_cache_path": project_relative(env_cache_path),
        "split": str(split_index.get("split", "unspecified")),
        "sample_count": int(split_index.get("sample_count", len(samples))),
        "sample_preview": sample_ids[:10],
        "manifest_job_name": str(split_index.get("job_name", "unknown")),
    }


def resolve_training_mode(training: dict[str, Any]) -> str:
    raw_mode = str(training.get("execution_mode", "auto")).lower()
    if raw_mode not in {"auto", "train", "scaffold"}:
        raise ValueError(f"Unsupported training.execution_mode: {raw_mode}")
    return raw_mode


def build_training_runtime_placeholder(reason: str) -> dict[str, Any]:
    return {
        "status": "scaffold_only",
        "reason": reason,
        "algorithm": "",
        "dependency_status": detect_maskable_ppo_dependencies(),
        "total_timesteps": 0,
        "seed": 0,
        "sample_count": 0,
        "vector_envs": 0,
        "observation_dim": 12,
        "model_path": "",
        "best_model_path": "",
        "tensorboard_log_dir": "",
        "callback_summary_path": "",
        "callback_summary": {},
        "runtime_summary_path": "",
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path, config = load_config(args.config)

    job_name = str(config.get("job_name", "training_run"))
    environment = dict(config.get("environment", {}))
    policy = dict(config.get("policy", {}))
    reward = dict(config.get("reward", {}))
    training = dict(config.get("training", {}))
    run_root = resolve_path(training.get("run_root", "runs"), PROJECT_ROOT)
    run_dir = resolve_path(args.run_dir, PROJECT_ROOT) if args.run_dir else create_timestamped_dir(run_root, job_name)
    split_index_value = args.split_index or training.get("split_index_path")
    execution_mode = resolve_training_mode(training)

    ensure_dir(run_dir)
    ensure_dir(run_dir / "artifacts")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "reports")

    copy_file(config_path, run_dir / f"config.snapshot{config_path.suffix}")

    dataset_metadata: dict[str, Any] = {
        "split_index_path": "",
        "split_index_snapshot_path": "",
        "sample_ids_path": "",
        "task_queue_path": "",
        "env_input_cache_path": "",
        "split": "",
        "sample_count": 0,
        "sample_preview": [],
        "manifest_job_name": "",
    }
    if split_index_value:
        split_index_path, split_index = load_split_index(split_index_value)
        dataset_metadata = {
            "split_index_path": project_relative(split_index_path),
            **materialize_training_inputs(run_dir, split_index),
        }
    else:
        dataset_metadata["notes"] = ["No split index was supplied. Training scaffold was created without dataset binding."]

    training_runtime = build_training_runtime_placeholder(
        "Training backend was not invoked yet."
    )
    algorithm = str(policy.get("algorithm", "")).lower()
    if execution_mode == "scaffold":
        training_runtime = build_training_runtime_placeholder(
            "training.execution_mode=scaffold, so only scaffolding artifacts were generated."
        )
    elif not dataset_metadata.get("env_input_cache_path"):
        training_runtime = build_training_runtime_placeholder(
            "No env_input_cache.json was generated, so Maskable PPO training was skipped."
        )
    elif algorithm != "maskable_ppo":
        training_runtime = build_training_runtime_placeholder(
            f"Unsupported policy algorithm for this trainer: {algorithm or 'unspecified'}."
        )
    else:
        env_cache_path = resolve_path(str(dataset_metadata["env_input_cache_path"]), PROJECT_ROOT)
        dependency_status = detect_maskable_ppo_dependencies()
        if execution_mode == "train" and not all(dependency_status.values()):
            missing = ", ".join(name for name, present in dependency_status.items() if not present)
            raise RuntimeError(f"Maskable PPO training requires missing dependencies: {missing}")
        if execution_mode == "auto" and not all(dependency_status.values()):
            training_runtime = run_maskable_ppo_training(
                run_dir=run_dir,
                env_cache_path=env_cache_path,
                environment_config=environment,
                policy_config=policy,
                reward_config=reward,
                training_config=training,
            )
        else:
            training_runtime = run_maskable_ppo_training(
                run_dir=run_dir,
                env_cache_path=env_cache_path,
                environment_config=environment,
                policy_config=policy,
                reward_config=reward,
                training_config=training,
            )

    runtime_status = str(training_runtime.get("status", ""))
    if runtime_status == "completed":
        status = "trained"
    elif runtime_status == "completed_no_feasible_checkpoint":
        status = "trained_no_feasible_checkpoint"
    else:
        status = "scaffolded"
    metadata = {
        "job_name": job_name,
        "created_at": utc_timestamp(),
        "status": status,
        "run_dir": project_relative(run_dir),
        "config_path": project_relative(config_path),
        "dataset": dataset_metadata,
        "environment": environment,
        "policy": policy,
        "reward": reward,
        "training": training,
        "training_runtime": training_runtime,
    }
    train_plan = {
        "phase": "fit" if status.startswith("trained") else "bootstrap",
        "steps": [
            "Expand parameterized operator locations and argument choices beyond the first deterministic action library.",
            "Cache verifier outputs and add stronger fuzz/symbolic backends to reduce training-time verification cost.",
            "Add resumable training, multi-worker log aggregation, and benchmark comparison hooks.",
        ],
    }

    dump_json(run_dir / "metadata.json", metadata)
    dump_json(run_dir / "train_plan.json", train_plan)
    reward_breakdown_path = run_dir / "logs" / "reward_breakdown.csv"
    if not reward_breakdown_path.exists():
        write_text(
            reward_breakdown_path,
            "step,total_reward,semantic_component,potency_component,diversity_component,llvm_block_component,branch_rewrite_component,ir_delta_component,cost_component\n",
        )
    action_mask_stats_path = run_dir / "logs" / "action_mask_stats.json"
    if not action_mask_stats_path.exists():
        write_text(
            action_mask_stats_path,
            '{\n  "status": "placeholder",\n  "message": "Populate with per-step mask statistics during training."\n}\n',
        )
    write_text(run_dir / "logs" / "seed.txt", f"{training.get('seed', 0)}\n")

    if status == "trained":
        print(f"Created training run: {run_dir}")
    elif status == "trained_no_feasible_checkpoint":
        print(f"Created training run with no feasible checkpoint: {run_dir}")
    else:
        print(f"Created training scaffold: {run_dir}")


if __name__ == "__main__":
    main()
