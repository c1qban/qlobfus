from __future__ import annotations

import json
import shutil
import unittest
import uuid

from scripts.analyze_runtime_costs import build_report, main
from scripts.common import PROJECT_ROOT, dump_json, ensure_dir


class AnalyzeRuntimeCostsTests(unittest.TestCase):
    def test_runtime_cost_report_counts_ppo_and_baseline_verification(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"runtime_costs_{uuid.uuid4().hex}")
        try:
            ppo_run = ensure_dir(root / "ppo")
            logs = ensure_dir(ppo_run / "logs")
            trace_dir = ensure_dir(ppo_run / "artifacts" / "env_workers" / "worker_00" / "sample")
            cache_dir = ensure_dir(ppo_run / "artifacts" / "env_workers" / "worker_00" / "_verifier_cache")
            dump_json(cache_dir / "cached.json", {"ok": True})
            trace_path = trace_dir / "transform_trace.jsonl"
            trace_event = {
                "verification": {
                    "notes": ["Loaded verification result from cache."],
                    "compile": {"succeeded": True},
                    "tests": {"cases": [{"name": "case_1"}]},
                    "diff": {"executed": True, "cases": [{"name": "case_1"}]},
                    "fuzz": {"executed": True, "cases": [{"name": "case_1"}]},
                }
            }
            trace_path.write_text(json.dumps(trace_event) + "\n", encoding="utf-8")
            episode = {
                "sample_id": "sample/a",
                "semantic_score": 1.0,
                "status": "completed",
                "trace_path": trace_path.relative_to(PROJECT_ROOT).as_posix(),
            }
            (logs / "episode_metrics.corrected.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
            dump_json(
                ppo_run / "training_runtime.json",
                {
                    "algorithm": "maskable_ppo",
                    "seed": 1,
                    "sample_count": 1,
                    "total_timesteps": 1,
                    "callback_summary": {
                        "episode_count": 1,
                        "observed_step_count": 1,
                        "episode_metrics_path": (logs / "episode_metrics.jsonl").relative_to(PROJECT_ROOT).as_posix(),
                    },
                },
            )

            baseline_summary = root / "baseline" / "reports" / "baseline_summary.json"
            ensure_dir(baseline_summary.parent)
            dump_json(
                baseline_summary,
                {
                    "created_at": "2026-01-01T00:00:00Z",
                    "results": {
                        "random": {
                            "episode_count": 1,
                            "completed_episode_count": 1,
                            "episodes": [
                                {
                                    "sample_id": "sample/a",
                                    "semantic_score": 1.0,
                                    "status": "completed",
                                    "verification_summary": {
                                        "compile": {"succeeded": True},
                                        "tests": {"cases": [{"name": "case_1"}]},
                                        "diff": {"executed": False, "cases": []},
                                        "fuzz": {"executed": False, "cases": []},
                                    },
                                }
                            ],
                        }
                    },
                },
            )

            report = build_report(
                baseline_summaries=[baseline_summary.relative_to(PROJECT_ROOT).as_posix()],
                ppo_run_dirs=[ppo_run.relative_to(PROJECT_ROOT).as_posix()],
                semantic_threshold=0.97,
                job_name="runtime_costs_test",
            )

            self.assertEqual(report["summary"]["row_count"], 2)
            self.assertEqual(report["summary"]["total_verification_calls"], 2)
            self.assertEqual(report["summary"]["total_cache_hits"], 1)
            ppo_row = next(row for row in report["rows"] if row["method"] == "maskable_ppo")
            self.assertEqual(ppo_row["cache_file_count"], 1)
            self.assertEqual(ppo_row["cache_hit_rate"], 1.0)

            out_dir = root / "out"
            exit_code = main(
                [
                    "--baseline-summary",
                    baseline_summary.relative_to(PROJECT_ROOT).as_posix(),
                    "--ppo-run-dir",
                    ppo_run.relative_to(PROJECT_ROOT).as_posix(),
                    "--output-dir",
                    out_dir.relative_to(PROJECT_ROOT).as_posix(),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "runtime_cost_report.md").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

