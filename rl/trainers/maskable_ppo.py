from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from rl.envs import GymObfuscationEnv, load_env_cache
from rl.trainers.callbacks import LLVMAwareTrainingCallback
from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file, project_relative, resolve_path


MASKABLE_PPO_DEPENDENCIES = [
    "gymnasium",
    "stable_baselines3",
    "sb3_contrib",
]


def detect_maskable_ppo_dependencies() -> dict[str, bool]:
    return {name: bool(importlib.util.find_spec(name)) for name in MASKABLE_PPO_DEPENDENCIES}


def _tensorboard_available() -> bool:
    return bool(importlib.util.find_spec("tensorboard"))


def _dependencies_ready(status: dict[str, bool]) -> bool:
    return all(status.values())


def _build_env_cache_from_split_index(path_value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_path(path_value, PROJECT_ROOT)
    split_index = load_structured_file(path)
    if not isinstance(split_index, dict):
        raise ValueError(f"Validation split index must decode to an object: {path}")
    if str(split_index.get("split", "")) != "validation":
        raise ValueError(f"Checkpoint selection index must be a validation split: {path}")
    samples: list[dict[str, Any]] = []
    for raw_sample in split_index.get("samples", []):
        sample = dict(raw_sample)
        harness = dict(sample.get("harness", {}))
        harness_path = str(harness.get("path", ""))
        harness_payload = load_structured_file(resolve_path(harness_path, PROJECT_ROOT)) if harness_path else {}
        cases = list(harness_payload.get("test_cases", [])) if isinstance(harness_payload, dict) else []
        samples.append(
            {
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
                    "case_count": int(harness.get("case_count", len(cases)) or len(cases)),
                    "test_cases": [dict(case) for case in cases if isinstance(case, dict)],
                },
            }
        )
    return path, {
        "job_name": str(split_index.get("job_name", "unknown")),
        "split": "validation",
        "sample_count": len(samples),
        "samples": samples,
    }


def _build_validation_evaluator(
    *,
    validation_env: GymObfuscationEnv,
    output_dir: Path,
    seed: int,
    use_action_masking: bool,
) -> Any:
    def evaluate(model: Any, episode_count: int) -> dict[str, Any]:
        episodes: list[dict[str, Any]] = []
        sample_ids = validation_env.core_env.available_sample_ids()
        for position, sample_id in enumerate(sample_ids, start=1):
            if position == 1 or position % 10 == 0 or position == len(sample_ids):
                print(
                    f"[validation-selection] episode={episode_count} sample={position}/{len(sample_ids)} id={sample_id}",
                    flush=True,
                )
            observation, _info = validation_env.reset(seed=seed + position - 1, options={"sample_id": sample_id})
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action_masks = validation_env.action_masks() if use_action_masking else None
                action, _state = model.predict(
                    observation,
                    deterministic=True,
                    action_masks=action_masks,
                )
                action_index = int(action.item()) if hasattr(action, "item") else int(action)
                observation, _reward, terminated, truncated, info = validation_env.step(action_index)
            episodes.append(
                {
                    "sample_id": sample_id,
                    "semantic_score": float(info.get("semantic_score", 0.0)),
                    "semantic_passed": float(info.get("semantic_score", 0.0)) >= validation_env.core_env.semantic_threshold,
                    "transformation_applied": bool(info.get("action_history", [])),
                    "termination_reason": str(info.get("termination_reason", "")),
                    "verifier_coverage": float(dict(info.get("verification_summary", {})).get("verifier_coverage", 0.0)),
                    "potency_score": float(info.get("potency_score", 0.0)),
                    "cost_penalty": float(info.get("cost_penalty", 0.0)),
                    "llvm_block_coverage": float(info.get("llvm_block_coverage", 0.0)),
                    "branch_rewrite_ratio": float(info.get("branch_rewrite_ratio", 0.0)),
                    "ir_structural_delta": float(info.get("ir_structural_delta", 0.0)),
                }
            )
        dump_json(output_dir / f"episode_{episode_count:06d}.validation_episodes.json", {"episodes": episodes})

        def aggregate(key: str) -> float:
            return mean(float(item.get(key, 0.0)) for item in episodes) if episodes else 0.0

        return {
            "sample_count": len(episodes),
            "semantic_score": aggregate("semantic_score"),
            "semantic_pass_rate": aggregate("semantic_passed"),
            "transformation_rate": aggregate("transformation_applied"),
            "nonzero_potency_rate": mean(1.0 if float(item["potency_score"]) > 0.0 else 0.0 for item in episodes) if episodes else 0.0,
            "nonzero_ir_delta_rate": mean(1.0 if float(item["ir_structural_delta"]) > 0.0 else 0.0 for item in episodes) if episodes else 0.0,
            "branch_rewrite_rate": mean(1.0 if float(item["branch_rewrite_ratio"]) > 0.0 else 0.0 for item in episodes) if episodes else 0.0,
            "verifier_coverage": aggregate("verifier_coverage"),
            "potency_score": aggregate("potency_score"),
            "cost_penalty": aggregate("cost_penalty"),
            "llvm_block_coverage": aggregate("llvm_block_coverage"),
            "branch_rewrite_ratio": aggregate("branch_rewrite_ratio"),
            "ir_structural_delta": aggregate("ir_structural_delta"),
            "episodes_path": project_relative(output_dir / f"episode_{episode_count:06d}.validation_episodes.json"),
        }

    return evaluate


def _build_runtime_summary(
    *,
    status: str,
    run_dir: Path,
    dependency_status: dict[str, bool],
    reason: str,
    algorithm: str,
    total_timesteps: int,
    seed: int,
    sample_count: int,
    vector_envs: int,
    model_path: Path | None = None,
    tensorboard_log_dir: Path | None = None,
    best_model_path: Path | None = None,
    best_infeasible_model_path: Path | None = None,
    callback_summary_path: Path | None = None,
    callback_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "status": status,
        "reason": reason,
        "algorithm": algorithm,
        "dependency_status": dependency_status,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "sample_count": sample_count,
        "vector_envs": vector_envs,
        "observation_dim": 12,
        "model_path": project_relative(model_path) if model_path else "",
        "best_model_path": project_relative(best_model_path) if best_model_path else "",
        "best_infeasible_model_path": (
            project_relative(best_infeasible_model_path) if best_infeasible_model_path else ""
        ),
        "tensorboard_log_dir": project_relative(tensorboard_log_dir) if tensorboard_log_dir else "",
        "callback_summary_path": project_relative(callback_summary_path) if callback_summary_path else "",
        "callback_summary": callback_summary or {},
        "runtime_summary_path": project_relative(run_dir / "training_runtime.json"),
    }
    dump_json(run_dir / "training_runtime.json", summary)
    return summary


def run_maskable_ppo_training(
    *,
    run_dir: Path,
    env_cache_path: Path,
    environment_config: dict[str, Any],
    policy_config: dict[str, Any],
    reward_config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    dependency_status = detect_maskable_ppo_dependencies()
    algorithm = str(policy_config.get("algorithm", "maskable_ppo"))
    total_timesteps = int(training_config.get("total_timesteps", 0))
    seed = int(training_config.get("seed", 0))
    vector_envs = max(1, int(environment_config.get("parallel_envs", 1)))
    use_action_masking = bool(policy_config.get("action_masking", True))

    env_cache = load_env_cache(env_cache_path)
    sample_count = int(env_cache.get("sample_count", len(env_cache.get("samples", []))))

    if sample_count <= 0:
        return _build_runtime_summary(
            status="skipped_empty_dataset",
            run_dir=run_dir,
            dependency_status=dependency_status,
            reason="No samples were available in env_input_cache.json.",
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            seed=seed,
            sample_count=sample_count,
            vector_envs=vector_envs,
        )

    if not _dependencies_ready(dependency_status):
        return _build_runtime_summary(
            status="skipped_missing_dependencies",
            run_dir=run_dir,
            dependency_status=dependency_status,
            reason="Install gymnasium, stable-baselines3, and sb3-contrib to enable Maskable PPO training.",
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            seed=seed,
            sample_count=sample_count,
            vector_envs=vector_envs,
        )

    mplconfig_dir = ensure_dir(run_dir / "artifacts" / "mplconfig")
    os.environ["MPLCONFIGDIR"] = str(mplconfig_dir)
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv

    set_random_seed(seed)
    tensorboard_log_dir = ensure_dir(run_dir / "logs" / "tensorboard") if _tensorboard_available() else None
    model_path = run_dir / "checkpoints" / "maskable_ppo_model.zip"
    validation_env: GymObfuscationEnv | None = None
    validation_evaluator = None
    validation_metadata: dict[str, Any] = {}
    selection_config = dict(training_config.get("checkpoint_selection", {}))
    validation_index_value = selection_config.get("validation_split_index_path")
    if validation_index_value:
        validation_index_path, validation_env_cache = _build_env_cache_from_split_index(str(validation_index_value))
        validation_env = GymObfuscationEnv(
            env_cache=validation_env_cache,
            max_steps=int(environment_config.get("max_steps", 8)),
            semantic_threshold=float(environment_config.get("semantic_threshold", 0.97)),
            legal_actions=list(policy_config.get("action_vocab", [])) or None,
            reward_config=reward_config,
            action_safety_config=dict(environment_config.get("action_safety", {})),
            verification_config=dict(environment_config.get("verification", {})),
            early_stop_config=dict(environment_config.get("early_stop", {})),
            artifact_root=run_dir / "artifacts" / "validation_selection_env",
        )
        validation_evaluator = _build_validation_evaluator(
            validation_env=validation_env,
            output_dir=ensure_dir(run_dir / "reports" / "validation_selection"),
            seed=seed,
            use_action_masking=use_action_masking,
        )
        validation_metadata = {
            "split_index_path": project_relative(validation_index_path),
            "sample_count": int(validation_env_cache["sample_count"]),
            "split": "validation",
            "deterministic": True,
            "action_masking": use_action_masking,
        }
    callback_wrapper = LLVMAwareTrainingCallback(
        run_dir=run_dir,
        selection_config=selection_config,
        periodic_evaluation_config=dict(training_config.get("periodic_evaluation", {})),
        validation_evaluator=validation_evaluator,
        validation_metadata=validation_metadata,
    )
    training_callback = callback_wrapper.build()

    max_steps = int(environment_config.get("max_steps", 8))
    semantic_threshold = float(environment_config.get("semantic_threshold", 0.97))
    action_safety_config = dict(environment_config.get("action_safety", {}))
    verification_config = dict(environment_config.get("verification", {}))
    early_stop_config = dict(environment_config.get("early_stop", {}))
    action_vocab = list(policy_config.get("action_vocab", [])) or None
    env_artifact_root = ensure_dir(run_dir / "artifacts" / "env_workers")

    def make_env(worker_index: int) -> Any:
        def _init() -> Any:
            env = GymObfuscationEnv.from_env_cache(
                env_cache_path,
                max_steps=max_steps,
                semantic_threshold=semantic_threshold,
                legal_actions=action_vocab,
                initial_cursor=worker_index,
                reward_config=reward_config,
                action_safety_config=action_safety_config,
                verification_config=verification_config,
                early_stop_config=early_stop_config,
                artifact_root=env_artifact_root / f"worker_{worker_index:02d}",
            )
            return Monitor(env)

        return _init

    vec_env = DummyVecEnv([make_env(index) for index in range(vector_envs)])
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        batch_size=int(training_config.get("batch_size", 32)),
        device=str(training_config.get("device", "auto")),
        gamma=float(training_config.get("gamma", 0.99)),
        learning_rate=float(training_config.get("learning_rate", 3e-4)),
        n_steps=int(training_config.get("n_steps", 64)),
        seed=seed,
        tensorboard_log=str(tensorboard_log_dir) if tensorboard_log_dir else None,
        verbose=int(training_config.get("verbose", 0)),
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=False,
            callback=training_callback,
            use_masking=use_action_masking,
        )
        model.save(str(model_path))
    finally:
        vec_env.close()
        if validation_env is not None:
            validation_env.close()

    callback_summary = {}
    if callback_wrapper.summary_path.exists():
        callback_summary = json.loads(callback_wrapper.summary_path.read_text(encoding="utf-8"))

    feasible_checkpoint_found = bool(callback_summary.get("feasible_checkpoint_found", True))
    return _build_runtime_summary(
        status="completed" if feasible_checkpoint_found else "completed_no_feasible_checkpoint",
        run_dir=run_dir,
        dependency_status=dependency_status,
        reason=(
            "Maskable PPO training finished successfully."
            if feasible_checkpoint_found
            else "Training finished, but no validation checkpoint satisfied the configured feasibility constraints."
        ),
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        seed=seed,
        sample_count=sample_count,
        vector_envs=vector_envs,
        model_path=model_path,
        tensorboard_log_dir=tensorboard_log_dir,
        best_model_path=callback_wrapper.best_model_path if callback_wrapper.best_model_path.exists() else None,
        best_infeasible_model_path=(
            callback_wrapper.best_infeasible_model_path
            if callback_wrapper.best_infeasible_model_path.exists()
            else None
        ),
        callback_summary_path=callback_wrapper.summary_path if callback_wrapper.summary_path.exists() else None,
        callback_summary=callback_summary,
    )
