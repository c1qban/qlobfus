from __future__ import annotations

import uuid
import unittest
from pathlib import Path

from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file
from scripts import create_expanded_harness_clean_eval, run_deobfuscation_attacks


class ExpandedEvalAndDeobfuscationTests(unittest.TestCase):
    def test_create_expanded_harness_clean_eval_filters_audit_and_empty_harness(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"expanded_eval_{uuid.uuid4().hex}")
        split_index = root / "validation.index.json"
        output_index = root / "formal_eval.index.json"
        audit_path = root / "harness_quality_audit.json"
        samples = [
            {
                "sample_id": "p1/s1",
                "split_group": "p1",
                "metadata": {"problem_id": "p1"},
                "harness": {"exists": True, "case_count": 2},
            },
            {
                "sample_id": "p1/s2",
                "split_group": "p1",
                "metadata": {"problem_id": "p1"},
                "harness": {"exists": True, "case_count": 0},
            },
            {
                "sample_id": "p2/s1",
                "split_group": "p2",
                "metadata": {"problem_id": "p2"},
                "harness": {"exists": True, "case_count": 2},
            },
        ]
        dump_json(split_index, {"job_name": "fixture", "split": "validation", "sample_count": 3, "samples": samples})
        dump_json(
            audit_path,
            {
                "samples": [
                    {"sample_id": "p1/s1", "status": "ok"},
                    {"sample_id": "p2/s1", "status": "needs_review"},
                ]
            },
        )

        exit_code = create_expanded_harness_clean_eval.main(
            [
                "--split-index",
                str(split_index),
                "--harness-audit",
                str(audit_path),
                "--output-index",
                str(output_index),
                "--target-samples",
                "10",
                "--min-samples",
                "1",
            ]
        )

        self.assertEqual(exit_code, 0)
        payload = load_structured_file(output_index)
        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["samples"][0]["sample_id"], "p1/s1")
        self.assertEqual(payload["problem_count"], 1)

    def test_deobfuscation_normalize_reduces_rename_only_strength(self) -> None:
        original = "int main(){ int value = 1; return value; }\n"
        candidate = "int main(){ int obf_name = 1; return obf_name; }\n"

        pre = run_deobfuscation_attacks.measure_pair(original, candidate)
        normalized_candidate = run_deobfuscation_attacks.apply_attack(candidate, "normalize")
        normalized_original = run_deobfuscation_attacks.apply_attack(original, "normalize")
        post = run_deobfuscation_attacks.measure_pair(normalized_original, normalized_candidate)

        self.assertGreater(pre["static_obfuscation_score"], post["static_obfuscation_score"])

    def test_deobfuscation_cfg_cleanup_removes_simple_opaque_dead_blocks(self) -> None:
        candidate = "int main(){ int x=1; if (0) { x = 99; } if (1) { x += 1; } return x; }\n"

        cleaned = run_deobfuscation_attacks.apply_attack(candidate, "cfg_cleanup")
        normalized = run_deobfuscation_attacks.apply_attack(candidate, "normalize_cfg_cleanup")

        self.assertNotIn("99", cleaned)
        self.assertNotIn("if", normalized)

    def test_deobfuscation_llvm_o2_falls_back_when_compiler_missing(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"llvm_o2_{uuid.uuid4().hex}")
        result = run_deobfuscation_attacks._compile_to_llvm_ir(
            source_path="sample.c",
            source_text="int main(){return 0;}\n",
            compiler="definitely_missing_clang_for_test",
            timeout_sec=0.1,
            cache_dir=root,
        )

        self.assertIn("LLVM_O2_IR_UNAVAILABLE", result)


if __name__ == "__main__":
    unittest.main()
