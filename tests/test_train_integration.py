from __future__ import annotations

import shutil
import unittest
import uuid

from scripts import prepare_dataset, train
from rl.trainers import detect_maskable_ppo_dependencies
from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file


class TrainIntegrationTests(unittest.TestCase):
    def test_train_uses_split_index_and_snapshots_dataset_binding(self) -> None:
        prepare_dataset.main(["--config", "configs/benchmark/smoke_dataset.example.yaml"])

        run_root = ensure_dir(PROJECT_ROOT / "artifacts" / "train_integration")
        run_dir = run_root / f"run_{uuid.uuid4().hex}"
        config_path = run_root / f"train_config_{uuid.uuid4().hex}.json"
        dump_json(
            config_path,
            {
                "job_name": "mppo_c_ir_test",
                "environment": {
                    "id": "obfuscation-env-v0",
                    "max_steps": 4,
                    "parallel_envs": 1,
                    "semantic_threshold": 0.97,
                },
                "policy": {
                    "algorithm": "maskable_ppo",
                    "hierarchical": True,
                    "state_encoder": "ast_cfg_ir_mlp",
                    "action_masking": True,
                },
                "reward": {
                    "semantic_gate": 0.97,
                    "security_weight": 1.0,
                    "potency_weight": 0.4,
                    "cost_weight": 0.3,
                    "diversity_weight": 0.1,
                },
                "training": {
                    "execution_mode": "auto",
                    "total_timesteps": 32,
                    "n_steps": 8,
                    "batch_size": 8,
                    "seed": 7,
                    "verbose": 0,
                    "checkpoint_selection": {
                        "metric_name": "semantic_first_custom",
                        "semantic_weight": 0.5,
                        "potency_weight": 0.2,
                        "llvm_block_weight": 0.1,
                        "branch_rewrite_weight": 0.05,
                        "ir_delta_weight": 0.05,
                        "verifier_weight": 0.15,
                        "cost_weight": 0.05,
                    },
                    "periodic_evaluation": {
                        "enabled": True,
                        "every_n_episodes": 1,
                        "benchmark": {
                            "name": "train_integration_periodic_eval",
                            "datasets": ["train"],
                        },
                        "metrics": ["semantic", "potency", "cost", "resilience", "llvm_ir"],
                    },
                    "run_root": "runs",
                },
            },
        )
        try:
            train.main(
                [
                    "--config",
                    str(config_path),
                    "--split-index",
                    "data/processed/smoke_dataset/splits/train.index.json",
                    "--run-dir",
                    str(run_dir),
                ]
            )

            metadata = load_structured_file(run_dir / "metadata.json")
            split_snapshot = load_structured_file(run_dir / "data" / "split_index.snapshot.json")
            env_input_cache = load_structured_file(run_dir / "data" / "env_input_cache.json")
            training_runtime = load_structured_file(run_dir / "training_runtime.json")
            sample_ids_text = (run_dir / "data" / "sample_ids.txt").read_text(encoding="utf-8")
            task_lines = (run_dir / "data" / "train_tasks.jsonl").read_text(encoding="utf-8").splitlines()

            self.assertEqual(metadata["dataset"]["split"], "train")
            self.assertEqual(metadata["dataset"]["sample_count"], 1)
            self.assertEqual(metadata["dataset"]["manifest_job_name"], "smoke_dataset")
            self.assertEqual(metadata["dataset"]["sample_preview"], ["sum_stdin"])
            self.assertEqual(
                metadata["dataset"]["split_index_path"],
                "data/processed/smoke_dataset/splits/train.index.json",
            )
            self.assertEqual(
                metadata["dataset"]["task_queue_path"],
                f"{run_dir.relative_to(PROJECT_ROOT).as_posix()}/data/train_tasks.jsonl",
            )
            self.assertEqual(
                metadata["dataset"]["env_input_cache_path"],
                f"{run_dir.relative_to(PROJECT_ROOT).as_posix()}/data/env_input_cache.json",
            )
            self.assertEqual(split_snapshot["split"], "train")
            self.assertEqual(split_snapshot["sample_count"], 1)
            self.assertEqual(env_input_cache["split"], "train")
            self.assertEqual(env_input_cache["sample_count"], 1)
            self.assertEqual(env_input_cache["samples"][0]["sample_id"], "sum_stdin")
            self.assertEqual(env_input_cache["samples"][0]["verification_input"]["case_count"], 2)
            self.assertEqual(len(env_input_cache["samples"][0]["verification_input"]["test_cases"]), 2)
            dependency_status = detect_maskable_ppo_dependencies()
            if all(dependency_status.values()):
                self.assertEqual(metadata["training_runtime"]["status"], "completed")
                self.assertEqual(training_runtime["status"], "completed")
                self.assertEqual(training_runtime["observation_dim"], 12)
                self.assertTrue((run_dir / "checkpoints" / "maskable_ppo_model.zip").exists())
                self.assertTrue((run_dir / "checkpoints" / "best_maskable_ppo_model.zip").exists())
                self.assertTrue((run_dir / "artifacts" / "env_workers").exists())
                self.assertTrue((run_dir / "reports" / "training_callback_summary.json").exists())
                self.assertTrue((run_dir / "reports" / "periodic_evaluations" / "index.json").exists())
                self.assertTrue((run_dir / "reports" / "periodic_evaluations" / "latest.evaluation_summary.json").exists())
                self.assertTrue((run_dir / "logs" / "episode_metrics.jsonl").exists())
                self.assertEqual(training_runtime["callback_summary"]["status"], "ok")
                self.assertEqual(training_runtime["callback_summary"]["selection_metric"], "semantic_first_custom")
                self.assertEqual(training_runtime["callback_summary"]["selection_config"]["semantic_weight"], 0.5)
                self.assertTrue(training_runtime["callback_summary"]["periodic_evaluation"]["enabled"])
                self.assertGreaterEqual(training_runtime["callback_summary"]["periodic_evaluation"]["snapshot_count"], 1)
                self.assertEqual(
                    training_runtime["best_model_path"],
                    f"{run_dir.relative_to(PROJECT_ROOT).as_posix()}/checkpoints/best_maskable_ppo_model.zip",
                )
            else:
                self.assertEqual(metadata["training_runtime"]["status"], "skipped_missing_dependencies")
                self.assertEqual(training_runtime["status"], "skipped_missing_dependencies")
            self.assertEqual(len(task_lines), 1)
            self.assertIn("sum_stdin", task_lines[0])
            self.assertIn("sum_stdin", sample_ids_text)
        finally:
            if config_path.exists():
                config_path.unlink()
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
