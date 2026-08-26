from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from scripts.common import dump_json
from scripts.summarize_experiments import main as summarize_main


class SummarizeExperimentsTests(unittest.TestCase):
    def test_summarize_experiments_combines_baseline_ppo_and_tigress_diagnosis(self) -> None:
        root = Path("artifacts") / "test_tmp" / f"summarize_experiments_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            baseline_summary = root / "baseline_summary.json"
            ppo_run_dir = root / "ppo"
            output_dir = root / "summary"
            dump_json(
                baseline_summary,
                {
                    "results": {
                        "random": {
                            "baseline_name": "random",
                            "status": "completed",
                            "episode_count": 1,
                            "completed_episode_count": 1,
                            "episodes": [
                                {
                                    "status": "completed",
                                    "sample_id": "p00001/s1",
                                    "semantic_score": 1.0,
                                    "verifier_coverage": 0.9,
                                    "potency_score": 0.2,
                                    "llvm_metric_available": True,
                                    "llvm_block_coverage": 0.1,
                                }
                            ],
                        },
                        "tigress": {
                            "baseline_name": "tigress",
                            "status": "completed",
                            "episode_count": 1,
                            "completed_episode_count": 1,
                            "episodes": [
                                {
                                    "status": "completed",
                                    "sample_id": "p00002/s2",
                                    "semantic_score": 0.0,
                                    "compile_succeeded": False,
                                    "transform_stderr": "Parsing error",
                                }
                            ],
                        },
                    }
                },
            )
            dump_json(
                root / "tigress_diagnosis.json",
                {
                    "diagnosis": {
                        "p00002/s2": {
                            "decision": "tool_incompatible_for_current_tigress_flatten",
                        }
                    }
                },
            )
            dump_json(
                ppo_run_dir / "training_runtime.json",
                {
                    "algorithm": "maskable_ppo",
                    "status": "completed",
                    "sample_count": 1,
                    "callback_summary": {
                        "episode_count": 1,
                        "mean_semantic_score": 1.0,
                        "mean_verifier_coverage": 0.9,
                        "mean_potency_score": 0.3,
                        "llvm_metric_episode_count": 1,
                        "best_selection_score": 0.5,
                    },
                },
            )

            summarize_main(
                [
                    "--baseline-summary",
                    str(baseline_summary),
                    "--ppo-run-dir",
                    str(ppo_run_dir),
                    "--tigress-diagnosis",
                    str(root / "tigress_diagnosis.json"),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            summary = (output_dir / "experiment_summary.md").read_text(encoding="utf-8")
            payload = (output_dir / "experiment_summary.json").read_text(encoding="utf-8")
            self.assertIn("random", summary)
            self.assertIn("maskable_ppo", summary)
            self.assertIn("tool_incompatible_for_current_tigress_flatten", payload)
            self.assertTrue((output_dir / "experiment_summary.csv").exists())
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            if root.exists():
                root.rmdir()

    def test_summarize_experiments_groups_multi_seed_ppo_runs(self) -> None:
        root = Path("artifacts") / "test_tmp" / f"summarize_grouped_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            output_dir = root / "summary"
            for seed, potency in [(1, 0.2), (7, 0.4)]:
                run_dir = root / f"full_seed{seed}"
                dump_json(
                    run_dir / "metadata.json",
                    {
                        "job_name": f"mppo_formal_eval_long_1536_seed{seed}",
                        "dataset": {
                            "split_index_path": "data/processed/project_codenet_c_100x4_strict/splits/formal_eval_30_v2.index.json"
                        },
                        "training": {
                            "seed": seed,
                            "total_timesteps": 1536,
                        },
                    },
                )
                dump_json(
                    run_dir / "training_runtime.json",
                    {
                        "algorithm": "maskable_ppo",
                        "status": "completed",
                        "sample_count": 1,
                        "seed": seed,
                        "total_timesteps": 1536,
                        "callback_summary": {
                            "episode_count": 1,
                            "mean_episode_reward": 1.0,
                            "mean_semantic_score": 1.0,
                            "mean_verifier_coverage": 0.8,
                            "mean_potency_score": potency,
                            "mean_llvm_block_coverage": 0.1,
                            "mean_branch_rewrite_ratio": 0.2,
                            "mean_ir_structural_delta": 0.3,
                            "llvm_metric_episode_count": 1,
                            "best_selection_score": 0.5,
                        },
                    },
                )

            summarize_main(
                [
                    "--ppo-run-dir",
                    str(root / "full_seed1"),
                    "--ppo-run-dir",
                    str(root / "full_seed7"),
                    "--group-by",
                    "method,variant",
                    "--aggregate",
                    "seed",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            grouped = json.loads((output_dir / "grouped_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(grouped["group_by"], ["method", "variant"])
            self.assertEqual(len(grouped["groups"]), 1)
            group = grouped["groups"][0]
            self.assertEqual(group["group"]["method"], "maskable_ppo")
            self.assertEqual(group["group"]["variant"], "full_long_1536")
            self.assertEqual(group["replicates"]["seed"], ["1", "7"])
            self.assertAlmostEqual(group["mean_potency_score"]["mean"], 0.3)
            self.assertTrue((output_dir / "grouped_summary.csv").exists())
            self.assertIn("Grouped Experiment Summary", (output_dir / "grouped_summary.md").read_text(encoding="utf-8"))
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            if root.exists():
                root.rmdir()


if __name__ == "__main__":
    unittest.main()
