from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from scripts.common import dump_json
from scripts.measure_static_obfuscation import main as metrics_main, measure_pair


class StaticObfuscationMetricTests(unittest.TestCase):
    def test_measure_pair_detects_identifier_and_token_changes(self) -> None:
        original = "int main(){int a=1; if(a){return a+1;} return 0;}\n"
        candidate = "int main(){int __ql_local_a=((4)-3); if(__ql_local_a){return __ql_local_a-(-1);} return ((7)-7);}\n"

        metrics = measure_pair(original, candidate)

        self.assertGreater(metrics["token_edit_distance_ratio"], 0.0)
        self.assertGreater(metrics["identifier_jaccard_distance"], 0.0)
        self.assertGreater(metrics["static_obfuscation_score"], 0.0)

    def test_cli_summarizes_baseline_trace_pairs(self) -> None:
        root = Path("artifacts") / "test_tmp" / f"static_metrics_{uuid.uuid4().hex}"
        source_dir = root / "sources"
        trace_dir = root / "run" / "artifacts" / "baselines" / "random" / "sample"
        report_dir = root / "run" / "reports"
        data_dir = root / "run" / "data"
        out_dir = root / "out"
        source_dir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            original = source_dir / "sample.c"
            candidate = trace_dir / "step_01.c"
            trace = trace_dir / "transform_trace.jsonl"
            original.write_text("int main(){int a=1; return a+1;}\n", encoding="utf-8")
            candidate.write_text("int main(){int __ql_local_a=((4)-3); return __ql_local_a-(-1);}\n", encoding="utf-8")
            trace.write_text(json.dumps({"event": "step", "source_artifact": str(candidate)}) + "\n", encoding="utf-8")
            dump_json(
                data_dir / "baseline_env_input_cache.json",
                {"samples": [{"sample_id": "p00000/s000", "source_path": str(original)}]},
            )
            dump_json(
                report_dir / "baseline_summary.json",
                {
                    "run_dir": str(root / "run"),
                    "results": {
                        "random": {
                            "episodes": [
                                {
                                    "sample_id": "p00000/s000",
                                    "trace_path": str(trace),
                                    "semantic_score": 1.0,
                                }
                            ]
                        }
                    },
                },
            )

            metrics_main(
                [
                    "--baseline-summary",
                    str(report_dir / "baseline_summary.json"),
                    "--output-dir",
                    str(out_dir),
                ]
            )

            payload = json.loads((out_dir / "static_obfuscation_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["row_count"], 1)
            self.assertEqual(payload["aggregates"][0]["method"], "random")
            self.assertTrue((out_dir / "static_obfuscation_metrics.md").exists())
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
