from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from eval.reporting import build_evaluation_summary
from scripts.common import dump_json, ensure_dir, project_relative
from scripts.common import utc_timestamp


class LLVMAwareTrainingCallback:
    def __init__(
        self,
        *,
        run_dir: Path,
        selection_config: dict[str, Any] | None = None,
        periodic_evaluation_config: dict[str, Any] | None = None,
        validation_evaluator: Callable[[Any, int], dict[str, Any]] | None = None,
        validation_metadata: dict[str, Any] | None = None,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _CallbackImpl(BaseCallback):
            def __init__(self, outer: "LLVMAwareTrainingCallback") -> None:
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                self.outer.on_step(self)
                return True

            def _on_training_end(self) -> None:
                self.outer.on_training_end(self)

        self._callback_type = _CallbackImpl
        self.run_dir = run_dir
        self.logs_dir = ensure_dir(run_dir / "logs")
        self.reports_dir = ensure_dir(run_dir / "reports")
        self.checkpoints_dir = ensure_dir(run_dir / "checkpoints")
        self.episode_log_path = self.logs_dir / "episode_metrics.jsonl"
        self.summary_path = self.reports_dir / "training_callback_summary.json"
        self.action_mask_stats_path = self.logs_dir / "action_mask_stats.json"
        self.best_model_path = self.checkpoints_dir / "best_maskable_ppo_model.zip"
        self.best_infeasible_model_path = self.checkpoints_dir / "best_infeasible_maskable_ppo_model.zip"
        self.periodic_eval_dir = ensure_dir(self.reports_dir / "periodic_evaluations")
        self.periodic_eval_index_path = self.periodic_eval_dir / "index.json"
        self.periodic_eval_latest_path = self.periodic_eval_dir / "latest.evaluation_summary.json"
        self.episode_count = 0
        self.llvm_metric_episode_count = 0
        self.best_selection_score = float("-inf")
        self.best_episode_metrics: dict[str, Any] = {}
        self.best_validation_metrics: dict[str, Any] = {}
        self.best_infeasible_validation_metrics: dict[str, Any] = {}
        self.selection_config = self._resolve_selection_config(selection_config or {})
        self.selection_metric = str(self.selection_config["metric_name"])
        self.validation_evaluator = validation_evaluator
        self.validation_metadata = dict(validation_metadata or {})
        self.validation_eval_dir = ensure_dir(self.reports_dir / "validation_selection")
        self.validation_eval_index_path = self.validation_eval_dir / "index.json"
        self.validation_eval_latest_path = self.validation_eval_dir / "latest.validation_summary.json"
        self._validation_evaluations: list[dict[str, Any]] = []
        self.periodic_evaluation_config = self._resolve_periodic_evaluation_config(periodic_evaluation_config or {})
        self._episode_metrics: list[dict[str, Any]] = []
        self._periodic_evaluations: list[dict[str, Any]] = []
        self._action_mask_sum = 0.0
        self._action_mask_count = 0
        self._action_mask_width_sum = 0.0
        self._step_count = 0
        self._started_at = time.monotonic()

    def build(self) -> Any:
        return self._callback_type(self)

    def on_step(self, callback: Any) -> None:
        infos = list(callback.locals.get("infos", []))
        dones = list(callback.locals.get("dones", []))
        if not dones:
            dones = [False for _ in infos]

        for info in infos:
            if not isinstance(info, dict):
                continue
            action_mask = info.get("action_mask")
            if isinstance(action_mask, list) and action_mask:
                self._action_mask_sum += sum(int(value) for value in action_mask) / float(len(action_mask))
                self._action_mask_width_sum += float(len(action_mask))
                self._action_mask_count += 1
            self._step_count += 1

        for done, info in zip(dones, infos):
            if not done or not isinstance(info, dict):
                continue
            metrics = self._extract_episode_metrics(info)
            self._episode_metrics.append(metrics)
            self.episode_count += 1
            if bool(metrics.get("llvm_metric_available", False)):
                self.llvm_metric_episode_count += 1
            self._append_episode_metrics(metrics)
            if self.validation_evaluator is None and float(metrics["selection_score"]) > self.best_selection_score:
                self.best_selection_score = float(metrics["selection_score"])
                self.best_episode_metrics = dict(metrics)
                callback.model.save(str(self.best_model_path))
            self._maybe_run_validation_selection(callback.model)
            self._maybe_write_periodic_evaluation_snapshot()
            self._write_summary()
            progress_every = int(self.selection_config["progress_every_n_episodes"])
            if self.episode_count % progress_every == 0:
                elapsed = time.monotonic() - self._started_at
                completed_steps = int(getattr(callback, "num_timesteps", self._step_count))
                total_steps = int(getattr(callback.model, "_total_timesteps", 0) or 0)
                eta = elapsed * max(total_steps - completed_steps, 0) / completed_steps if completed_steps > 0 else 0.0
                print(
                    f"[train] episodes={self.episode_count} timesteps={completed_steps}/{total_steps or '?'} "
                    f"elapsed={int(elapsed)}s eta={int(eta)}s",
                    flush=True,
                )

    def on_training_end(self, callback: Any) -> None:
        self._run_validation_selection(callback.model, reason="training_end", force=True)
        if self.periodic_evaluation_config["enabled"] and self.episode_count > 0:
            if not self._periodic_evaluations or self._periodic_evaluations[-1]["episode_count"] != self.episode_count:
                self._write_periodic_evaluation_snapshot(reason="training_end")
        self._write_summary()

    def _extract_episode_metrics(self, info: dict[str, Any]) -> dict[str, Any]:
        episode = dict(info.get("episode", {}))
        reward_breakdown = dict(info.get("reward_breakdown", {}))
        llvm_metric_available = bool(info.get("llvm_metric_available", False))
        metrics = {
            "sample_id": str(info.get("sample_id", "")),
            "split": str(info.get("split", "")),
            "trace_path": str(info.get("trace_path", "")),
            "termination_reason": str(info.get("termination_reason", "")),
            "episode_reward": float(episode.get("r", reward_breakdown.get("total_reward", 0.0))),
            "episode_length": int(episode.get("l", 0)),
            "semantic_score": float(info.get("semantic_score", 0.0)),
            "verifier_coverage": float(info.get("verification_summary", {}).get("verifier_coverage", 0.0)),
            "potency_score": float(info.get("potency_score", 0.0)),
            "cost_penalty": float(info.get("cost_penalty", 0.0)),
            "llvm_block_coverage": float(info.get("llvm_block_coverage", 0.0)),
            "branch_rewrite_ratio": float(info.get("branch_rewrite_ratio", 0.0)),
            "ir_structural_delta": float(info.get("ir_structural_delta", 0.0)),
            "llvm_metric_available": llvm_metric_available,
            "reward_breakdown": reward_breakdown,
        }
        metrics["selection_score"] = self._compute_selection_score(metrics)
        return metrics

    def _compute_selection_score(self, metrics: dict[str, Any]) -> float:
        return (
            float(self.selection_config["semantic_weight"]) * float(metrics.get("semantic_score", 0.0))
            + float(self.selection_config["potency_weight"]) * float(metrics.get("potency_score", 0.0))
            + float(self.selection_config["llvm_block_weight"]) * float(metrics.get("llvm_block_coverage", 0.0))
            + float(self.selection_config["branch_rewrite_weight"]) * float(metrics.get("branch_rewrite_ratio", 0.0))
            + float(self.selection_config["ir_delta_weight"]) * float(metrics.get("ir_structural_delta", 0.0))
            + float(self.selection_config["verifier_weight"]) * float(metrics.get("verifier_coverage", 0.0))
            - float(self.selection_config["cost_weight"]) * float(metrics.get("cost_penalty", 0.0))
        )

    def _append_episode_metrics(self, metrics: dict[str, Any]) -> None:
        with self.episode_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    def _write_summary(self) -> None:
        llvm_episodes = [
            item for item in self._episode_metrics
            if bool(item.get("llvm_metric_available", False))
        ]

        def _mean(items: list[dict[str, Any]], key: str) -> float:
            if not items:
                return 0.0
            return sum(float(item.get(key, 0.0)) for item in items) / float(len(items))

        action_mask_mean = self._action_mask_sum / float(self._action_mask_count) if self._action_mask_count else 0.0
        action_mask_width_mean = self._action_mask_width_sum / float(self._action_mask_count) if self._action_mask_count else 0.0
        summary = {
            "status": "ok",
            "selection_metric": self.selection_metric,
            "selection_config": dict(self.selection_config),
            "episode_count": self.episode_count,
            "llvm_metric_episode_count": self.llvm_metric_episode_count,
            "best_selection_score": self.best_selection_score if (self.best_episode_metrics or self.best_validation_metrics) else 0.0,
            "best_model_path": project_relative(self.best_model_path) if self.best_model_path.exists() else "",
            "best_infeasible_model_path": (
                project_relative(self.best_infeasible_model_path)
                if self.best_infeasible_model_path.exists()
                else ""
            ),
            "best_episode_metrics": self.best_episode_metrics,
            "checkpoint_selection_source": "validation_aggregate" if self.validation_evaluator is not None else "training_episode",
            "feasible_checkpoint_found": bool(self.best_validation_metrics) if self.validation_evaluator is not None else bool(self.best_episode_metrics),
            "best_validation_metrics": self.best_validation_metrics,
            "best_infeasible_validation_metrics": self.best_infeasible_validation_metrics,
            "validation_selection": {
                "enabled": self.validation_evaluator is not None,
                "every_n_episodes": int(self.selection_config["validation_every_n_episodes"]),
                "evaluation_count": len(self._validation_evaluations),
                "metadata": dict(self.validation_metadata),
                "latest_path": project_relative(self.validation_eval_latest_path) if self.validation_eval_latest_path.exists() else "",
                "index_path": project_relative(self.validation_eval_index_path) if self.validation_eval_index_path.exists() else "",
            },
            "mean_episode_reward": _mean(self._episode_metrics, "episode_reward"),
            "mean_semantic_score": _mean(self._episode_metrics, "semantic_score"),
            "mean_potency_score": _mean(self._episode_metrics, "potency_score"),
            "mean_verifier_coverage": _mean(self._episode_metrics, "verifier_coverage"),
            "mean_llvm_block_coverage": _mean(llvm_episodes, "llvm_block_coverage"),
            "mean_branch_rewrite_ratio": _mean(llvm_episodes, "branch_rewrite_ratio"),
            "mean_ir_structural_delta": _mean(llvm_episodes, "ir_structural_delta"),
            "action_mask_mean_active_ratio": action_mask_mean,
            "action_mask_mean_width": action_mask_width_mean,
            "observed_step_count": self._step_count,
            "episode_metrics_path": project_relative(self.episode_log_path),
            "periodic_evaluation": {
                "enabled": bool(self.periodic_evaluation_config["enabled"]),
                "every_n_episodes": int(self.periodic_evaluation_config["every_n_episodes"]),
                "snapshot_count": len(self._periodic_evaluations),
                "latest_path": project_relative(self.periodic_eval_latest_path) if self.periodic_eval_latest_path.exists() else "",
                "index_path": project_relative(self.periodic_eval_index_path) if self.periodic_eval_index_path.exists() else "",
            },
        }
        dump_json(self.summary_path, summary)
        dump_json(
            self.action_mask_stats_path,
            {
                "status": "ok",
                "observed_step_count": self._step_count,
                "mean_active_ratio": action_mask_mean,
                "mean_action_vocab_width": action_mask_width_mean,
            },
        )

    def _maybe_run_validation_selection(self, model: Any) -> None:
        if self.validation_evaluator is None:
            return
        every_n_episodes = int(self.selection_config["validation_every_n_episodes"])
        if self.episode_count <= 0 or self.episode_count % every_n_episodes != 0:
            return
        self._run_validation_selection(model, reason="episode_interval")

    def _run_validation_selection(self, model: Any, *, reason: str, force: bool = False) -> None:
        if self.validation_evaluator is None or self.episode_count <= 0:
            return
        if self._validation_evaluations and self._validation_evaluations[-1]["episode_count"] == self.episode_count:
            return
        if not force and self.episode_count % int(self.selection_config["validation_every_n_episodes"]) != 0:
            return
        metrics = dict(self.validation_evaluator(model, self.episode_count))
        metrics["selection_score"] = self._compute_selection_score(metrics)
        metrics["selection_eligible"] = self._selection_eligible(metrics)
        metrics["constraint_violations"] = self._constraint_violations(metrics)
        metrics["episode_count"] = self.episode_count
        metrics["reason"] = reason
        metrics["created_at"] = utc_timestamp()
        improved = self._validation_candidate_improves(metrics)
        metrics["improved_best"] = improved
        improved_infeasible = self._infeasible_candidate_improves(metrics)
        metrics["improved_infeasible_fallback"] = improved_infeasible
        if improved:
            self.best_selection_score = float(metrics["selection_score"])
            self.best_validation_metrics = dict(metrics)
            model.save(str(self.best_model_path))
        if improved_infeasible:
            self.best_infeasible_validation_metrics = dict(metrics)
            model.save(str(self.best_infeasible_model_path))
        snapshot_path = self.validation_eval_dir / f"episode_{self.episode_count:06d}.validation_summary.json"
        dump_json(snapshot_path, metrics)
        dump_json(self.validation_eval_latest_path, metrics)
        self._validation_evaluations.append(
            {
                "episode_count": self.episode_count,
                "selection_score": float(metrics["selection_score"]),
                "improved_best": improved,
                "improved_infeasible_fallback": improved_infeasible,
                "selection_eligible": bool(metrics["selection_eligible"]),
                "total_constraint_violation": float(metrics["constraint_violations"]["total"]),
                "path": project_relative(snapshot_path),
                "created_at": metrics["created_at"],
            }
        )
        dump_json(
            self.validation_eval_index_path,
            {
                "status": "ok",
                "selection_metric": self.selection_metric,
                "metadata": dict(self.validation_metadata),
                "evaluation_count": len(self._validation_evaluations),
                "evaluations": list(self._validation_evaluations),
            },
        )
        print(
            f"[validation-selection] episode={self.episode_count} samples={metrics.get('sample_count', 0)} "
            f"score={float(metrics['selection_score']):.6f} improved={improved} "
            f"infeasible_fallback={improved_infeasible}",
            flush=True,
        )

    def _maybe_write_periodic_evaluation_snapshot(self) -> None:
        if not self.periodic_evaluation_config["enabled"]:
            return
        every_n_episodes = int(self.periodic_evaluation_config["every_n_episodes"])
        if self.episode_count <= 0 or self.episode_count % every_n_episodes != 0:
            return
        if self._periodic_evaluations and self._periodic_evaluations[-1]["episode_count"] == self.episode_count:
            return
        self._write_periodic_evaluation_snapshot(reason="episode_interval")

    def _write_periodic_evaluation_snapshot(self, *, reason: str) -> None:
        benchmark = dict(self.periodic_evaluation_config["benchmark"])
        datasets = benchmark.get("datasets", [])
        benchmark["datasets"] = list(datasets) if isinstance(datasets, list) else [str(datasets)]
        summary = build_evaluation_summary(
            job_name=str(benchmark.get("name", self.run_dir.name)),
            benchmark=benchmark,
            metrics=list(self.periodic_evaluation_config["metrics"]),
            training_context={
                "source": "training_callback",
                "run_dir": project_relative(self.run_dir),
                "selection_metric": self.selection_metric,
                "episode_count": self.episode_count,
                "observed_step_count": self._step_count,
            },
            run_dir=self.run_dir,
        )
        summary["periodic_context"] = {
            "reason": reason,
            "created_at": utc_timestamp(),
            "episode_count": self.episode_count,
            "observed_step_count": self._step_count,
            "best_selection_score": self.best_selection_score if (self.best_episode_metrics or self.best_validation_metrics) else 0.0,
            "best_model_path": project_relative(self.best_model_path) if self.best_model_path.exists() else "",
        }
        snapshot_path = self.periodic_eval_dir / f"episode_{self.episode_count:06d}.evaluation_summary.json"
        dump_json(snapshot_path, summary)
        dump_json(self.periodic_eval_latest_path, summary)
        snapshot_record = {
            "episode_count": self.episode_count,
            "observed_step_count": self._step_count,
            "reason": reason,
            "path": project_relative(snapshot_path),
            "created_at": summary["periodic_context"]["created_at"],
        }
        self._periodic_evaluations.append(snapshot_record)
        dump_json(
            self.periodic_eval_index_path,
            {
                "status": "ok",
                "run_dir": project_relative(self.run_dir),
                "snapshot_count": len(self._periodic_evaluations),
                "snapshots": list(self._periodic_evaluations),
            },
        )

    def _resolve_selection_config(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "metric_name": str(
                raw_config.get(
                    "metric_name",
                    "semantic+potency+llvm_coverage+branch_rewrite+ir_delta+verifier-cost",
                )
            ),
            "semantic_weight": float(raw_config.get("semantic_weight", 0.35)),
            "potency_weight": float(raw_config.get("potency_weight", 0.20)),
            "llvm_block_weight": float(raw_config.get("llvm_block_weight", 0.15)),
            "branch_rewrite_weight": float(raw_config.get("branch_rewrite_weight", 0.10)),
            "ir_delta_weight": float(raw_config.get("ir_delta_weight", 0.10)),
            "verifier_weight": float(raw_config.get("verifier_weight", 0.15)),
            "cost_weight": float(raw_config.get("cost_weight", 0.05)),
            "min_semantic_score": float(raw_config.get("min_semantic_score", 0.0)),
            "min_semantic_pass_rate": float(raw_config.get("min_semantic_pass_rate", 0.0)),
            "min_transformation_rate": float(raw_config.get("min_transformation_rate", 0.0)),
            "validation_every_n_episodes": max(1, int(raw_config.get("validation_every_n_episodes", 100))),
            "progress_every_n_episodes": max(1, int(raw_config.get("progress_every_n_episodes", 10))),
        }

    def _selection_eligible(self, metrics: dict[str, Any]) -> bool:
        return (
            float(metrics.get("semantic_score", 0.0)) >= float(self.selection_config["min_semantic_score"])
            and float(metrics.get("semantic_pass_rate", 1.0)) >= float(self.selection_config["min_semantic_pass_rate"])
            and float(metrics.get("transformation_rate", 1.0)) >= float(self.selection_config["min_transformation_rate"])
        )

    def _constraint_violations(self, metrics: dict[str, Any]) -> dict[str, float]:
        semantic_score_shortfall = max(
            float(self.selection_config["min_semantic_score"]) - float(metrics.get("semantic_score", 0.0)),
            0.0,
        )
        semantic_pass_rate_shortfall = max(
            float(self.selection_config["min_semantic_pass_rate"])
            - float(metrics.get("semantic_pass_rate", 1.0)),
            0.0,
        )
        transformation_rate_shortfall = max(
            float(self.selection_config["min_transformation_rate"])
            - float(metrics.get("transformation_rate", 1.0)),
            0.0,
        )
        return {
            "semantic_score_shortfall": semantic_score_shortfall,
            "semantic_pass_rate_shortfall": semantic_pass_rate_shortfall,
            "transformation_rate_shortfall": transformation_rate_shortfall,
            "total": semantic_score_shortfall + semantic_pass_rate_shortfall + transformation_rate_shortfall,
        }

    def _validation_candidate_improves(self, metrics: dict[str, Any]) -> bool:
        candidate_eligible = bool(metrics.get("selection_eligible", False))
        constraints_enabled = any(
            float(self.selection_config[key]) > 0.0
            for key in ("min_semantic_score", "min_semantic_pass_rate", "min_transformation_rate")
        )
        if not self.best_validation_metrics:
            return candidate_eligible or not constraints_enabled
        best_eligible = bool(self.best_validation_metrics.get("selection_eligible", False))
        if candidate_eligible != best_eligible:
            return candidate_eligible
        return float(metrics["selection_score"]) > self.best_selection_score

    def _infeasible_candidate_improves(self, metrics: dict[str, Any]) -> bool:
        if bool(metrics.get("selection_eligible", False)):
            return False
        if not self.best_infeasible_validation_metrics:
            return True
        candidate_violation = float(metrics.get("constraint_violations", {}).get("total", float("inf")))
        best_violation = float(
            self.best_infeasible_validation_metrics.get("constraint_violations", {}).get("total", float("inf"))
        )
        if candidate_violation != best_violation:
            return candidate_violation < best_violation
        return float(metrics["selection_score"]) > float(
            self.best_infeasible_validation_metrics.get("selection_score", float("-inf"))
        )

    def _resolve_periodic_evaluation_config(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        benchmark = raw_config.get("benchmark", {})
        if not isinstance(benchmark, dict):
            benchmark = {}
        metrics = raw_config.get(
            "metrics",
            ["semantic", "potency", "cost", "resilience", "llvm_ir"],
        )
        if not isinstance(metrics, list) or not metrics:
            metrics = ["semantic", "potency", "cost", "resilience", "llvm_ir"]
        datasets = benchmark.get("datasets", ["train"])
        if not isinstance(datasets, list) or not datasets:
            datasets = ["train"]
        return {
            "enabled": bool(raw_config.get("enabled", False)),
            "every_n_episodes": max(1, int(raw_config.get("every_n_episodes", 5))),
            "benchmark": {
                "name": str(benchmark.get("name", "training_periodic_eval")),
                "datasets": [str(item) for item in datasets],
            },
            "metrics": [str(item) for item in metrics],
        }
