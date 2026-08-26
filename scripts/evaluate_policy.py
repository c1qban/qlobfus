from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rl.envs import GymObfuscationEnv
from scripts.common import (
    PROJECT_ROOT,
    copy_file,
    dump_json,
    ensure_dir,
    load_config,
    project_relative,
    resolve_path,
    utc_timestamp,
    write_text,
)
from scripts.run_baselines import build_env_cache, build_episode_result, format_duration, summarize_baseline
from scripts.train import load_split_index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Maskable PPO checkpoint without further training.")
    parser.add_argument("--config", required=True, help="Training-compatible config containing environment/policy/reward.")
    parser.add_argument("--checkpoint", required=True, help="Frozen best_maskable_ppo_model.zip or model checkpoint.")
    parser.add_argument("--split-index", required=True, help="Independent validation or test harness-clean index.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _as_action_index(action: Any) -> int:
    if hasattr(action, "item"):
        return int(action.item())
    if isinstance(action, (list, tuple)):
        return int(action[0])
    return int(action)


def evaluate_model(
    *,
    model: Any,
    env: GymObfuscationEnv,
    deterministic: bool,
    seed: int,
    use_action_masking: bool,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    sample_ids = env.core_env.available_sample_ids()
    started_at = time.monotonic()
    for position, sample_id in enumerate(sample_ids, start=1):
        elapsed = time.monotonic() - started_at
        eta = (elapsed / max(position - 1, 1)) * max(len(sample_ids) - position + 1, 0) if position > 1 else 0.0
        print(
            f"[frozen-ppo] {position}/{len(sample_ids)} sample={sample_id} "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
            flush=True,
        )
        observation, info = env.reset(seed=seed + position - 1, options={"sample_id": sample_id})
        terminated = truncated = False
        reward = 0.0
        while not (terminated or truncated):
            mask = env.action_masks() if use_action_masking else None
            action, _state = model.predict(observation, deterministic=deterministic, action_masks=mask)
            observation, reward, terminated, truncated, info = env.step(_as_action_index(action))
        state = env.core_env.current_state()
        if state is None:
            raise RuntimeError(f"Environment did not retain terminal state for sample: {sample_id}")
        episode = build_episode_result(state, reward, dict(info), baseline_name="maskable_ppo_frozen")
        episode["deterministic"] = deterministic
        episode["action_history"] = list(state.action_history)
        episodes.append(episode)
    return summarize_baseline("maskable_ppo_frozen", episodes)


def _render_markdown(summary: dict[str, Any]) -> str:
    result = summary["result"]
    lines = [
        "# Frozen Maskable PPO Evaluation",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- split_index: `{summary['split_index_path']}`",
        f"- split: `{summary['split']}`",
        f"- deterministic: `{summary['deterministic']}`",
        f"- requested_samples: `{summary['sample_count']}`",
        f"- completed_episodes: `{result['completed_episode_count']}`",
        f"- mean_reward: `{result['mean_reward']:.6f}`",
        f"- mean_semantic_score: `{result['mean_semantic_score']:.6f}`",
        f"- mean_verifier_coverage: `{result['mean_verifier_coverage']:.6f}`",
        f"- mean_llvm_block_coverage: `{result['mean_llvm_block_coverage']:.6f}`",
        f"- mean_branch_rewrite_ratio: `{result['mean_branch_rewrite_ratio']:.6f}`",
        f"- mean_ir_structural_delta: `{result['mean_ir_structural_delta']:.6f}`",
        "",
        "This report was produced by checkpoint inference only; no training update was performed.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path, config = load_config(args.config)
    checkpoint_path = resolve_path(args.checkpoint, PROJECT_ROOT)
    split_index_path, split_index = load_split_index(args.split_index)
    output_dir = ensure_dir(resolve_path(args.output_dir, PROJECT_ROOT))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if str(split_index.get("split", "")) == "train":
        raise ValueError("Frozen policy evaluation refuses a train split. Use validation or test.")

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError("Install sb3-contrib to evaluate Maskable PPO checkpoints.") from exc

    environment_config = dict(config.get("environment", {}))
    policy_config = dict(config.get("policy", {}))
    reward_config = dict(config.get("reward", {}))
    use_action_masking = bool(policy_config.get("action_masking", True))
    env_cache = build_env_cache(split_index)
    env = GymObfuscationEnv(
        env_cache=env_cache,
        max_steps=int(environment_config.get("max_steps", 8)),
        semantic_threshold=float(environment_config.get("semantic_threshold", 0.97)),
        legal_actions=list(policy_config.get("action_vocab", [])) or None,
        reward_config=reward_config,
        action_safety_config=dict(environment_config.get("action_safety", {})),
        verification_config=dict(environment_config.get("verification", {})),
        early_stop_config=dict(environment_config.get("early_stop", {})),
        artifact_root=output_dir / "artifacts" / "env",
    )
    model = MaskablePPO.load(str(checkpoint_path), device=str(args.device))
    model_action_count = int(getattr(getattr(model, "action_space", None), "n", -1))
    if model_action_count != int(env.action_space.n):
        raise ValueError(
            f"Checkpoint action space ({model_action_count}) does not match configured environment ({env.action_space.n})."
        )

    try:
        result = evaluate_model(
            model=model,
            env=env,
            deterministic=bool(args.deterministic),
            seed=int(args.seed),
            use_action_masking=use_action_masking,
        )
    finally:
        env.close()

    copy_file(config_path, output_dir / f"evaluation.config{config_path.suffix}")
    dump_json(output_dir / "data" / "split_index.snapshot.json", split_index)
    dump_json(output_dir / "data" / "env_input_cache.json", env_cache)
    summary = {
        "job_name": str(config.get("job_name", "frozen_ppo_evaluation")),
        "created_at": utc_timestamp(),
        "status": "completed",
        "checkpoint_path": project_relative(checkpoint_path),
        "config_path": project_relative(config_path),
        "split_index_path": project_relative(split_index_path),
        "split": str(split_index.get("split", "")),
        "sample_count": int(split_index.get("sample_count", len(split_index.get("samples", [])))),
        "deterministic": bool(args.deterministic),
        "action_masking": use_action_masking,
        "seed": int(args.seed),
        "action_vocab": list(env.core_env.action_vocab),
        "early_stop": dict(env.core_env.early_stop_config),
        "result": result,
    }
    dump_json(output_dir / "reports" / "frozen_policy_evaluation.json", summary)
    write_text(output_dir / "reports" / "frozen_policy_evaluation.md", _render_markdown(summary))
    write_text(
        output_dir / "reports" / "episodes.jsonl",
        "".join(json.dumps(episode, sort_keys=True) + "\n" for episode in result["episodes"]),
    )
    print(f"Wrote frozen policy evaluation: {output_dir / 'reports' / 'frozen_policy_evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
