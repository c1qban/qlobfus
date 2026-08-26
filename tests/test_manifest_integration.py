from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from scripts import prepare_dataset
from scripts.dataset_report import build_dataset_report
from scripts.prepare_dataset import assign_split_labels
from scripts.common import PROJECT_ROOT, ensure_dir, load_structured_file
from verifier.compile import choose_compiler
from verifier.pipeline import VerificationPipeline


class ManifestIntegrationTests(unittest.TestCase):
    def test_assign_split_labels_groups_project_holdout_samples(self) -> None:
        samples = [
            {"sample_id": "proj_a/file1", "relative_source_path": "proj_a/file1.c"},
            {"sample_id": "proj_a/file2", "relative_source_path": "proj_a/file2.c"},
            {"sample_id": "proj_b/file1", "relative_source_path": "proj_b/file1.c"},
            {"sample_id": "proj_b/file2", "relative_source_path": "proj_b/file2.c"},
        ]

        summary = assign_split_labels(
            samples,
            {
                "strategy": "project_holdout",
                "seed": "unit-test",
                "train_ratio": 0.5,
                "validation_ratio": 0.25,
                "test_ratio": 0.25,
            },
        )

        self.assertEqual(summary["strategy"], "project_holdout")
        self.assertEqual(sum(summary["counts"].values()), 4)
        self.assertEqual(samples[0]["split_group"], "proj_a")
        self.assertEqual(samples[1]["split_group"], "proj_a")
        self.assertEqual(samples[2]["split_group"], "proj_b")
        self.assertEqual(samples[3]["split_group"], "proj_b")
        self.assertEqual(samples[0]["split"], samples[1]["split"])
        self.assertEqual(samples[2]["split"], samples[3]["split"])
        self.assertIn(samples[0]["split"], {"train", "validation", "test"})
        self.assertIn(samples[2]["split"], {"train", "validation", "test"})

    @unittest.skipIf(choose_compiler() is None, "No supported compiler is available for dataset compile filtering.")
    def test_prepare_dataset_writes_compile_and_harness_metadata(self) -> None:
        prepare_dataset.main(["--config", "configs/benchmark/smoke_dataset.example.yaml"])

        manifest_path = PROJECT_ROOT / "data" / "manifests" / "smoke_dataset.manifest.json"
        manifest = load_structured_file(manifest_path)

        self.assertEqual(manifest["dataset"]["name"], "smoke-c")
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(manifest["verification_defaults"]["compile"]["compiler_flags"], ["-O0"])
        self.assertEqual(manifest["verification_defaults"]["harness"]["format"], "executable_stdio_v1")
        self.assertEqual(manifest["discovery"]["sample_count"], 1)
        self.assertEqual(manifest["discovery"]["discovered_sample_count"], 2)
        self.assertEqual(manifest["discovery"]["filtered_out_count"], 1)
        self.assertEqual(manifest["discovery"]["samples_with_harness"], 1)
        self.assertEqual(manifest["split_summary"]["strategy"], "single_bucket")
        self.assertEqual(manifest["split_summary"]["counts"]["train"], 1)
        self.assertEqual(manifest["split_index_files"]["train"]["sample_count"], 1)
        self.assertEqual(manifest["split_index_files"]["validation"]["sample_count"], 0)
        self.assertEqual(manifest["split_index_files"]["test"]["sample_count"], 0)

        sample = manifest["samples"][0]
        self.assertEqual(sample["sample_id"], "sum_stdin")
        self.assertEqual(sample["compile"]["compiler_flags"], ["-O0"])
        self.assertTrue(sample["harness"]["exists"])
        self.assertEqual(sample["harness"]["case_count"], 2)
        self.assertEqual(sample["harness"]["format"], "executable_stdio_v1")
        self.assertEqual(len(sample["harness"]["test_cases"]), 2)
        self.assertEqual(sample["split"], "train")
        self.assertTrue(sample["prepare_checks"]["compile_check"]["succeeded"])

        filtered_sample = manifest["filtered_out_samples"][0]
        self.assertEqual(filtered_sample["sample_id"], "broken_syntax")
        self.assertIn("not_compilable", filtered_sample["filter_reasons"])
        self.assertFalse(filtered_sample["prepare_checks"]["compile_check"]["succeeded"])

        train_index_path = PROJECT_ROOT / "data" / "processed" / "smoke_dataset" / "splits" / "train.index.json"
        validation_index_path = PROJECT_ROOT / "data" / "processed" / "smoke_dataset" / "splits" / "validation.index.json"
        test_index_path = PROJECT_ROOT / "data" / "processed" / "smoke_dataset" / "splits" / "test.index.json"
        train_index = load_structured_file(train_index_path)
        validation_index = load_structured_file(validation_index_path)
        test_index = load_structured_file(test_index_path)

        self.assertEqual(train_index["sample_count"], 1)
        self.assertEqual(train_index["samples"][0]["sample_id"], "sum_stdin")
        self.assertEqual(train_index["samples"][0]["split"], "train")
        self.assertEqual(train_index["samples"][0]["compile"]["compiler_flags"], ["-O0"])
        self.assertEqual(train_index["samples"][0]["harness"]["case_count"], 2)
        self.assertEqual(validation_index["sample_count"], 0)
        self.assertEqual(test_index["sample_count"], 0)

    @unittest.skipIf(choose_compiler() is None, "No supported compiler is available for dataset report tests.")
    def test_dataset_report_summarizes_manifest(self) -> None:
        prepare_dataset.main(["--config", "configs/benchmark/smoke_dataset.example.yaml"])
        manifest_path = PROJECT_ROOT / "data" / "manifests" / "smoke_dataset.manifest.json"
        manifest = load_structured_file(manifest_path)

        report = build_dataset_report(manifest, manifest_path)

        self.assertEqual(report["totals"]["discovered_sample_count"], 2)
        self.assertEqual(report["totals"]["kept_sample_count"], 1)
        self.assertEqual(report["totals"]["filtered_out_count"], 1)
        self.assertEqual(report["compile"]["checked_count"], 2)
        self.assertEqual(report["compile"]["succeeded_count"], 1)
        self.assertEqual(report["compile"]["failed_count"], 1)
        self.assertEqual(report["splits"]["sample_counts"]["train"], 1)
        self.assertEqual(report["filter_reason_counts"]["not_compilable"], 1)

    @unittest.skipIf(choose_compiler() is None, "No supported compiler is available for real verification tests.")
    def test_verification_pipeline_reads_manifest_sample(self) -> None:
        prepare_dataset.main(["--config", "configs/benchmark/smoke_dataset.example.yaml"])

        build_root = ensure_dir(PROJECT_ROOT / "artifacts" / "manifest_verify")
        run_root = build_root / f"run_{uuid.uuid4().hex}"
        ensure_dir(run_root)
        try:
            summary = VerificationPipeline(build_root=run_root).verify_manifest_sample(
                manifest_path=PROJECT_ROOT / "data" / "manifests" / "smoke_dataset.manifest.json",
                sample_id="sum_stdin",
            )
        finally:
            shutil.rmtree(run_root, ignore_errors=True)

        self.assertTrue(summary.compile.succeeded)
        self.assertTrue(summary.tests.passed)
        self.assertEqual(summary.tests.passed_cases, 2)
        self.assertIn("compile", summary.available_signals)
        self.assertIn("tests", summary.available_signals)
        self.assertGreaterEqual(summary.semantic_score, summary.semantic_threshold)


if __name__ == "__main__":
    unittest.main()
