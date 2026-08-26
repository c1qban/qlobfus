from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from scripts.common import dump_json
from scripts.plot_results import main as plot_main


class PlotResultsTests(unittest.TestCase):
    def test_plot_results_generates_expected_figures(self) -> None:
        root = Path("artifacts") / "test_tmp" / f"plot_results_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            baseline = root / "experiment_summary.json"
            grouped = root / "grouped_summary.json"
            corrected = root / "corrected_summary.json"
            tigress = root / "tigress_breakdown.json"
            v3_summary = root / "v3_experiment_summary.json"
            v3_grouped = root / "v3_grouped_summary.json"
            v3_tigress = root / "v3_tigress_breakdown.json"
            out = root / "figures"

            dump_json(
                baseline,
                {
                    "methods": [
                        {"method": "random", "semantic_pass_rate": 1.0, "mean_potency_score": 0.2, "mean_verifier_coverage": 0.8},
                        {"method": "rule_based", "semantic_pass_rate": 1.0, "mean_potency_score": 0.1, "mean_verifier_coverage": 0.8},
                        {"method": "tigress", "semantic_pass_rate": 0.7, "mean_potency_score": 0.9, "mean_verifier_coverage": 0.7},
                        {"method": "maskable_ppo", "semantic_pass_rate": 1.0, "mean_potency_score": 0.3, "mean_verifier_coverage": 0.85},
                        {"method": "maskable_ppo#2", "semantic_pass_rate": 1.0, "mean_potency_score": 0.4, "mean_verifier_coverage": 0.9},
                    ]
                },
            )
            groups = []
            for variant, semantic, potency, verifier, failures in [
                ("full_short", 1.0, 0.2, 0.8, 0.0),
                ("full_long_1536", 1.0, 0.4, 0.9, 0.0),
                ("no_action_safety", 0.9, 0.2, 0.8, 3.0),
                ("no_fuzz_verifier", 1.0, 0.2, 0.6, 0.0),
                ("no_llvm_reward", 1.0, 0.2, 0.8, 0.0),
            ]:
                groups.append(
                    {
                        "group": {"method": "maskable_ppo", "variant": variant},
                        "row_count": 3,
                        "replicates": {"seed": ["1", "7", "13"]},
                        "semantic_pass_rate": {"mean": semantic, "std": 0.01, "min": semantic, "max": semantic},
                        "mean_semantic_score": {"mean": semantic, "std": 0.01, "min": semantic, "max": semantic},
                        "mean_potency_score": {"mean": potency, "std": 0.01, "min": potency, "max": potency},
                        "mean_verifier_coverage": {"mean": verifier, "std": 0.01, "min": verifier, "max": verifier},
                        "mean_llvm_block_coverage": {"mean": 0.04, "std": 0.01, "min": 0.03, "max": 0.05},
                        "mean_branch_rewrite_ratio": {"mean": 0.005, "std": 0.001, "min": 0.004, "max": 0.006},
                        "mean_ir_structural_delta": {"mean": 0.04, "std": 0.01, "min": 0.03, "max": 0.05},
                        "failure_count": {"mean": failures, "std": 0.5, "min": failures, "max": failures},
                    }
                )
            dump_json(grouped, {"groups": groups})
            dump_json(
                corrected,
                {
                    "aggregates": [
                        {"variant": "full_short", "semantic_pass_rate_mean": 1.0, "semantic_pass_rate_std": 0.0},
                        {"variant": "full_long_1536", "semantic_pass_rate_mean": 1.0, "semantic_pass_rate_std": 0.0},
                        {"variant": "no_action_safety", "semantic_pass_rate_mean": 0.9, "semantic_pass_rate_std": 0.01},
                        {"variant": "no_fuzz_verifier", "semantic_pass_rate_mean": 1.0, "semantic_pass_rate_std": 0.0},
                        {"variant": "no_llvm_reward", "semantic_pass_rate_mean": 1.0, "semantic_pass_rate_std": 0.0},
                    ]
                },
            )
            dump_json(tigress, {"summary": {"by_category": {"tool_parse_failure": 4, "tool_compile_failure": 1, "semantic_failure": 2}}})
            dump_json(
                v3_summary,
                {
                    "methods": [
                        {"method": "random", "semantic_pass_rate": 1.0, "mean_potency_score": 0.2, "mean_verifier_coverage": 0.8, "mean_ir_structural_delta": 0.05},
                        {"method": "rule_based", "semantic_pass_rate": 1.0, "mean_potency_score": 0.1, "mean_verifier_coverage": 0.8, "mean_ir_structural_delta": 0.05},
                        {"method": "tigress", "semantic_pass_rate": 0.78, "mean_potency_score": 0.95, "mean_verifier_coverage": 0.83, "mean_ir_structural_delta": 1.0},
                    ]
                },
            )
            dump_json(
                v3_grouped,
                {
                    "groups": [
                        {
                            "group": {"method": "maskable_ppo", "variant": "full_short"},
                            "semantic_pass_rate": {"mean": 1.0, "std": 0.0},
                            "mean_potency_score": {"mean": 0.17, "std": 0.01},
                            "mean_verifier_coverage": {"mean": 0.84, "std": 0.01},
                            "mean_ir_structural_delta": {"mean": 0.035, "std": 0.001},
                        }
                    ]
                },
            )
            dump_json(v3_tigress, {"summary": {"by_category": {"tool_parse_failure": 3, "tool_compile_failure": 1, "semantic_failure": 1}}})

            plot_main(
                [
                    "--baseline-summary",
                    str(baseline),
                    "--grouped-summary",
                    str(grouped),
                    "--corrected-summary",
                    str(corrected),
                    "--tigress-breakdown",
                    str(tigress),
                    "--v3-summary",
                    str(v3_summary),
                    "--v3-grouped-summary",
                    str(v3_grouped),
                    "--v3-tigress-breakdown",
                    str(v3_tigress),
                    "--output-dir",
                    str(out),
                ]
            )

            self.assertTrue((out / "README.md").exists())
            self.assertTrue((out / "main_baseline_comparison.png").exists())
            self.assertTrue((out / "formal_eval_v3_clean_main.png").exists())
            self.assertTrue((out / "formal_eval_v3_tigress_failure_breakdown.png").exists())
            self.assertTrue((out / "ppo_short_vs_long.png").exists())
            self.assertTrue((out / "ablation_components.png").exists())
            self.assertTrue((out / "tigress_failure_breakdown.png").exists())
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
