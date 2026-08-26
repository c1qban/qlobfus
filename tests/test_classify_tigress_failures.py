from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from scripts.classify_tigress_failures import main as classify_main
from scripts.common import dump_json


class ClassifyTigressFailuresTests(unittest.TestCase):
    def test_classifies_tigress_failures_into_breakdown_buckets(self) -> None:
        root = Path("artifacts") / "test_tmp" / f"classify_tigress_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            baseline_summary = root / "baseline_summary.json"
            output_dir = root / "out"
            dump_json(
                baseline_summary,
                {
                    "results": {
                        "tigress": {
                            "baseline_name": "tigress",
                            "episodes": [
                                {
                                    "sample_id": "p1/s1",
                                    "status": "completed",
                                    "semantic_score": 0.0,
                                    "compile_succeeded": False,
                                    "tests_passed": False,
                                    "transform_stderr": "syntax error\nParsing error",
                                    "verification_summary": {"compile": {"returncode": 1}},
                                },
                                {
                                    "sample_id": "p2/s2",
                                    "status": "completed",
                                    "semantic_score": 0.0,
                                    "compile_succeeded": False,
                                    "tests_passed": False,
                                    "transform_stderr": "",
                                    "verification_summary": {
                                        "compile": {"returncode": 1, "stderr": "candidate.c: error: bad output"}
                                    },
                                },
                                {
                                    "sample_id": "p3/s3",
                                    "status": "completed",
                                    "semantic_score": 0.5,
                                    "compile_succeeded": True,
                                    "tests_passed": False,
                                    "verification_summary": {
                                        "compile": {"returncode": 0},
                                        "tests": {
                                            "passed": False,
                                            "failed_cases": 1,
                                            "cases": [
                                                {
                                                    "name": "case_1",
                                                    "passed": False,
                                                    "notes": ["stdout mismatch"],
                                                }
                                            ],
                                        },
                                    },
                                },
                            ],
                        }
                    }
                },
            )

            classify_main(
                [
                    "--baseline-summary",
                    str(baseline_summary),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            payload = json.loads((output_dir / "tigress_failure_breakdown.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["by_category"]["tool_parse_failure"], 1)
            self.assertEqual(payload["summary"]["by_category"]["tool_compile_failure"], 1)
            self.assertEqual(payload["summary"]["by_category"]["semantic_failure"], 1)
            markdown = (output_dir / "tigress_failure_breakdown.md").read_text(encoding="utf-8")
            self.assertIn("Tigress Failure Breakdown", markdown)
            self.assertTrue((output_dir / "tigress_failure_breakdown.csv").exists())
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
