from __future__ import annotations

import unittest
import uuid

from scripts import create_harness_clean_protocol
from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file, write_text


class HarnessCleanProtocolTests(unittest.TestCase):
    def test_protocol_preserves_splits_filters_harness_and_reports_leakage(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"protocol_{uuid.uuid4().hex}")
        processed = ensure_dir(root / "processed")
        splits_root = ensure_dir(processed / "splits")
        source_root = ensure_dir(root / "sources")

        samples_by_split = {}
        for split, problem_id in (("train", "p1"), ("validation", "p2"), ("test", "p3")):
            source_path = source_root / f"{split}.c"
            write_text(source_path, f"int main(void) {{ return {len(problem_id)}; }}\n")
            samples_by_split[split] = [
                {
                    "sample_id": f"{problem_id}/s1",
                    "split": split,
                    "split_group": problem_id,
                    "source_path": str(source_path),
                    "harness": {"exists": True, "case_count": 2},
                },
                {
                    "sample_id": f"{problem_id}/empty",
                    "split": split,
                    "split_group": problem_id,
                    "source_path": str(source_path),
                    "harness": {"exists": False, "case_count": 0},
                },
            ]
            dump_json(
                splits_root / f"{split}.index.json",
                {"job_name": "fixture", "split": split, "samples": samples_by_split[split]},
            )

        output_root = root / "protocol"
        exit_code = create_harness_clean_protocol.main(
            [
                "--processed-root",
                str(processed),
                "--output-root",
                str(output_root),
                "--min-train-samples",
                "1",
                "--min-validation-samples",
                "1",
                "--min-test-samples",
                "1",
            ]
        )

        self.assertEqual(exit_code, 0)
        report = load_structured_file(output_root / "protocol.report.json")
        self.assertEqual(report["status"], "not_ready")
        self.assertEqual(report["leakage"]["status"], "fail")
        self.assertGreater(report["leakage"]["source_hash_leak_count"], 0)
        for split in ("train", "validation", "test"):
            index = load_structured_file(output_root / f"{split}_harness_clean.index.json")
            self.assertEqual(index["sample_count"], 1)
            self.assertEqual(index["samples"][0]["split"], split)
            self.assertEqual(report["splits"][split]["rejection_breakdown"]["missing_or_empty_harness"], 1)

    def test_protocol_can_require_audit_ok(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"protocol_audit_{uuid.uuid4().hex}")
        processed = ensure_dir(root / "processed")
        splits_root = ensure_dir(processed / "splits")
        source = root / "sample.c"
        write_text(source, "int main(void) { return 0; }\n")
        for split in ("train", "validation", "test"):
            dump_json(
                splits_root / f"{split}.index.json",
                {
                    "split": split,
                    "samples": [
                        {
                            "sample_id": f"{split}/s1",
                            "split": split,
                            "split_group": split,
                            "source_path": str(source),
                            "harness": {"exists": True, "case_count": 1},
                        }
                    ],
                },
            )
        audit = root / "train.audit.json"
        dump_json(audit, {"samples": [{"sample_id": "train/s1", "status": "needs_review"}]})

        create_harness_clean_protocol.main(
            [
                "--processed-root",
                str(processed),
                "--output-root",
                str(root / "protocol"),
                "--harness-audit",
                f"train={audit}",
                "--min-train-samples",
                "0",
                "--min-validation-samples",
                "0",
                "--min-test-samples",
                "0",
            ]
        )

        train_index = load_structured_file(root / "protocol" / "train_harness_clean.index.json")
        report = load_structured_file(root / "protocol" / "protocol.report.json")
        self.assertEqual(train_index["sample_count"], 0)
        self.assertEqual(report["splits"]["train"]["rejection_breakdown"]["harness_audit_needs_review"], 1)

    def test_ast_shape_proxy_match_is_reported_but_does_not_block_protocol(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"protocol_proxy_{uuid.uuid4().hex}")
        processed = ensure_dir(root / "processed")
        splits_root = ensure_dir(processed / "splits")
        source_root = ensure_dir(root / "sources")

        for split, problem_id, expression in (
            ("train", "p1", "x + 1"),
            ("validation", "p2", "x * 2"),
            ("test", "p3", "x - 3"),
        ):
            source_path = source_root / f"{split}.c"
            write_text(source_path, f"int main(void) {{ int x = 1; return {expression}; }}\n")
            dump_json(
                splits_root / f"{split}.index.json",
                {
                    "split": split,
                    "samples": [
                        {
                            "sample_id": f"{problem_id}/s1",
                            "split": split,
                            "split_group": problem_id,
                            "source_path": str(source_path),
                            "harness": {"exists": True, "case_count": 1},
                        }
                    ],
                },
            )

        create_harness_clean_protocol.main(
            [
                "--processed-root",
                str(processed),
                "--output-root",
                str(root / "protocol"),
                "--min-train-samples",
                "1",
                "--min-validation-samples",
                "1",
                "--min-test-samples",
                "1",
            ]
        )

        report = load_structured_file(root / "protocol" / "protocol.report.json")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["leakage"]["status"], "pass")
        self.assertEqual(report["leakage"]["ast_shape_proxy_status"], "warning")
        self.assertGreater(report["leakage"]["ast_shape_proxy_leak_count"], 0)


if __name__ == "__main__":
    unittest.main()
