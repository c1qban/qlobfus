from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from rl.trainers.callbacks import LLVMAwareTrainingCallback
from rl.trainers.maskable_ppo import _build_env_cache_from_split_index
from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file


class _FakeModel:
    def save(self, path: str) -> None:
        Path(path).write_text("checkpoint\n", encoding="utf-8")


class ValidationCheckpointSelectionTests(unittest.TestCase):
    def test_validation_evaluation_selects_best_checkpoint_from_aggregate_metrics(self) -> None:
        run_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"validation_selection_{uuid.uuid4().hex}")

        def evaluator(_model, episode_count: int):
            return {
                "sample_count": 3,
                "semantic_score": 1.0,
                "verifier_coverage": 0.9,
                "potency_score": episode_count / 1000.0,
                "cost_penalty": 0.1,
                "llvm_block_coverage": 0.2,
                "branch_rewrite_ratio": 0.1,
                "ir_structural_delta": 0.3,
            }

        try:
            callback = LLVMAwareTrainingCallback(
                run_dir=run_dir,
                selection_config={"validation_every_n_episodes": 100},
                validation_evaluator=evaluator,
                validation_metadata={"split": "validation", "sample_count": 3},
            )
            callback.episode_count = 100
            callback._run_validation_selection(_FakeModel(), reason="test")
            callback._write_summary()

            summary = load_structured_file(callback.summary_path)
            validation_index = load_structured_file(callback.validation_eval_index_path)
            self.assertEqual(summary["checkpoint_selection_source"], "validation_aggregate")
            self.assertTrue(summary["feasible_checkpoint_found"])
            self.assertEqual(summary["best_validation_metrics"]["sample_count"], 3)
            self.assertEqual(summary["best_selection_score"], summary["best_validation_metrics"]["selection_score"])
            self.assertTrue(callback.best_model_path.exists())
            self.assertEqual(validation_index["evaluation_count"], 1)
            self.assertTrue(validation_index["evaluations"][0]["improved_best"])
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_checkpoint_selection_rejects_non_validation_index(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"validation_index_{uuid.uuid4().hex}")
        index_path = root / "train.index.json"
        dump_json(index_path, {"split": "train", "samples": []})
        try:
            with self.assertRaisesRegex(ValueError, "must be a validation split"):
                _build_env_cache_from_split_index(index_path)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_semantic_and_transformation_constraints_override_raw_utility(self) -> None:
        run_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"validation_constraints_{uuid.uuid4().hex}")

        def evaluator(_model, episode_count: int):
            if episode_count == 100:
                return {
                    "sample_count": 3,
                    "semantic_score": 0.90,
                    "semantic_pass_rate": 0.90,
                    "transformation_rate": 1.0,
                    "potency_score": 1.0,
                }
            return {
                "sample_count": 3,
                "semantic_score": 0.99,
                "semantic_pass_rate": 1.0,
                "transformation_rate": 0.8,
                "potency_score": 0.2,
            }

        try:
            callback = LLVMAwareTrainingCallback(
                run_dir=run_dir,
                selection_config={
                    "validation_every_n_episodes": 100,
                    "semantic_weight": 0.0,
                    "potency_weight": 1.0,
                    "min_semantic_score": 0.97,
                    "min_semantic_pass_rate": 0.97,
                    "min_transformation_rate": 0.5,
                },
                validation_evaluator=evaluator,
                validation_metadata={"split": "validation", "sample_count": 3},
            )
            callback.episode_count = 100
            callback._run_validation_selection(_FakeModel(), reason="test")
            self.assertEqual(callback.best_validation_metrics, {})
            self.assertFalse(callback.best_model_path.exists())
            self.assertTrue(callback.best_infeasible_model_path.exists())
            self.assertEqual(callback.best_infeasible_validation_metrics["episode_count"], 100)
            self.assertGreater(
                callback.best_infeasible_validation_metrics["constraint_violations"]["total"],
                0.0,
            )

            callback.episode_count = 200
            callback._run_validation_selection(_FakeModel(), reason="test")
            self.assertTrue(callback.best_validation_metrics["selection_eligible"])
            self.assertEqual(callback.best_validation_metrics["episode_count"], 200)
            self.assertEqual(callback.best_selection_score, 0.2)
            self.assertEqual(callback.best_infeasible_validation_metrics["episode_count"], 100)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_best_infeasible_checkpoint_prefers_lower_constraint_violation(self) -> None:
        run_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"validation_fallback_{uuid.uuid4().hex}")

        def evaluator(_model, episode_count: int):
            if episode_count == 100:
                return {
                    "semantic_score": 0.80,
                    "semantic_pass_rate": 0.80,
                    "transformation_rate": 1.0,
                    "potency_score": 1.0,
                }
            return {
                "semantic_score": 0.96,
                "semantic_pass_rate": 0.96,
                "transformation_rate": 1.0,
                "potency_score": 0.1,
            }

        try:
            callback = LLVMAwareTrainingCallback(
                run_dir=run_dir,
                selection_config={
                    "semantic_weight": 0.0,
                    "potency_weight": 1.0,
                    "min_semantic_score": 0.97,
                    "min_semantic_pass_rate": 0.97,
                },
                validation_evaluator=evaluator,
            )
            callback.episode_count = 100
            callback._run_validation_selection(_FakeModel(), reason="test")
            callback.episode_count = 200
            callback._run_validation_selection(_FakeModel(), reason="test")
            callback._write_summary()

            summary = load_structured_file(callback.summary_path)
            self.assertFalse(summary["feasible_checkpoint_found"])
            self.assertEqual(summary["best_infeasible_validation_metrics"]["episode_count"], 200)
            self.assertTrue(summary["best_infeasible_model_path"].endswith("best_infeasible_maskable_ppo_model.zip"))
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
