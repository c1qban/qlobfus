from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from scripts import prepare_dataset, run_baselines
from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file, write_text


class CodeNetAndBaselineTests(unittest.TestCase):
    def test_prepare_dataset_supports_project_codenet_layout(self) -> None:
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        fixture_root = temp_root / f"codenet_fixture_{uuid.uuid4().hex}"
        source_root = fixture_root / "raw" / "project-codenet-c"
        metadata_path = source_root / "metadata.csv"
        manifest_path = fixture_root / "manifests" / "codenet_fixture.manifest.json"
        processed_root = fixture_root / "processed" / "codenet_fixture"
        config_path = fixture_root / "codenet_fixture_config.json"

        ensure_dir(source_root / "p00001" / "C")
        ensure_dir(source_root / "p00002" / "C")
        write_text(source_root / "p00001" / "C" / "s1001.c", "int main(void) {\n    return 0;\n}\n")
        write_text(source_root / "p00002" / "C" / "s2002.c", "int main(void) {\n    return 1;\n}\n")
        write_text(
            metadata_path,
            "submission_id,status,user_id\n"
            "s1001,Accepted,u001\n"
            "s2002,Wrong Answer,u002\n",
        )
        dump_json(
            config_path,
            {
                "job_name": "codenet_fixture",
                "dataset": {
                    "name": "project-codenet-c",
                    "language": "c",
                    "layout": "project_codenet",
                    "source_root": str(source_root),
                    "metadata_csv": str(metadata_path),
                    "metadata_key_field": "submission_id",
                    "metadata_lookup": "submission_id",
                    "language_dirs": ["C"],
                    "sample_id_template": "{problem_id}/{submission_id}",
                    "processed_root": str(processed_root),
                    "manifest_path": str(manifest_path),
                    "file_patterns": ["*.c"],
                },
                "split": {
                    "strategy": "project_holdout",
                    "seed": "fixture",
                    "train_ratio": 1.0,
                    "validation_ratio": 0.0,
                    "test_ratio": 0.0,
                },
                "filters": {
                    "accepted_only": True,
                    "metadata_status_field": "status",
                    "accepted_values": ["Accepted"],
                    "require_compilable": False,
                },
            },
        )

        try:
            prepare_dataset.main(["--config", str(config_path)])
            manifest = load_structured_file(manifest_path)
            train_index = load_structured_file(processed_root / "splits" / "train.index.json")
        finally:
            shutil.rmtree(fixture_root, ignore_errors=True)

        self.assertEqual(manifest["dataset"]["layout"], "project_codenet")
        self.assertEqual(manifest["discovery"]["metadata_row_count"], 2)
        self.assertEqual(manifest["discovery"]["sample_count"], 1)
        self.assertEqual(manifest["samples"][0]["sample_id"], "p00001/s1001")
        self.assertEqual(manifest["samples"][0]["split_group_candidate"], "p00001")
        self.assertTrue(manifest["samples"][0]["metadata"]["metadata_available"])
        self.assertEqual(manifest["filtered_out_samples"][0]["filter_reasons"], ["not_accepted"])
        self.assertEqual(train_index["samples"][0]["sample_id"], "p00001/s1001")

    def test_prepare_dataset_supports_project_codenet_metadata_directory(self) -> None:
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        fixture_root = temp_root / f"codenet_metadata_dir_fixture_{uuid.uuid4().hex}"
        dataset_root = fixture_root / "Project_CodeNet"
        source_root = dataset_root / "data"
        metadata_root = dataset_root / "metadata"
        manifest_path = fixture_root / "manifests" / "codenet_metadata_dir.manifest.json"
        processed_root = fixture_root / "processed" / "codenet_metadata_dir"
        config_path = fixture_root / "codenet_metadata_dir_config.json"

        ensure_dir(source_root / "p00001" / "C")
        ensure_dir(source_root / "p00002" / "C")
        ensure_dir(metadata_root)
        write_text(source_root / "p00001" / "C" / "s1001.c", "int main(void) {\n    return 0;\n}\n")
        write_text(source_root / "p00002" / "C" / "s2002.c", "int main(void) {\n    return 1;\n}\n")
        write_text(
            metadata_root / "p00001.csv",
            "submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy\n"
            "s1001,p00001,u001,0,C,C,c,Accepted,1,1,1,1/1\n",
        )
        write_text(
            metadata_root / "p00002.csv",
            "submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy\n"
            "s2002,p00002,u002,0,C,C,c,Wrong Answer,1,1,1,0/1\n",
        )
        write_text(metadata_root / "problem_list.csv", "id,name,dataset,time_limit,memory_limit\n")
        dump_json(
            config_path,
            {
                "job_name": "codenet_metadata_dir_fixture",
                "dataset": {
                    "name": "project-codenet-c",
                    "language": "c",
                    "layout": "project_codenet",
                    "source_root": str(source_root),
                    "metadata_csv": str(metadata_root),
                    "metadata_key_field": "submission_id",
                    "metadata_lookup": "submission_id",
                    "language_dirs": ["C"],
                    "sample_id_template": "{problem_id}/{submission_id}",
                    "processed_root": str(processed_root),
                    "manifest_path": str(manifest_path),
                    "file_patterns": ["*.c"],
                },
                "split": {
                    "strategy": "project_holdout",
                    "seed": "fixture",
                    "train_ratio": 1.0,
                    "validation_ratio": 0.0,
                    "test_ratio": 0.0,
                },
                "filters": {
                    "accepted_only": True,
                    "metadata_status_field": "status",
                    "accepted_values": ["Accepted"],
                    "require_compilable": False,
                },
            },
        )

        try:
            prepare_dataset.main(["--config", str(config_path)])
            manifest = load_structured_file(manifest_path)
        finally:
            shutil.rmtree(fixture_root, ignore_errors=True)

        self.assertEqual(manifest["discovery"]["metadata_row_count"], 2)
        self.assertEqual(manifest["discovery"]["sample_count"], 1)
        self.assertEqual(manifest["samples"][0]["sample_id"], "p00001/s1001")
        self.assertEqual(manifest["filtered_out_samples"][0]["filter_reasons"], ["not_accepted"])

    def test_run_baselines_generates_random_and_rule_summaries(self) -> None:
        prepare_dataset.main(["--config", "configs/benchmark/smoke_dataset.example.yaml"])

        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        run_dir = temp_root / f"baseline_run_{uuid.uuid4().hex}"
        config_path = temp_root / f"baseline_config_{uuid.uuid4().hex}.json"
        dump_json(
            config_path,
            {
                "job_name": "smoke_baselines_test",
                "run_root": "artifacts/baselines",
                "split_index_path": "data/processed/smoke_dataset/splits/train.index.json",
                "environment": {
                    "max_steps": 3,
                    "semantic_threshold": 0.97,
                },
                "reward": {
                    "semantic_gate": 0.97,
                    "security_weight": 1.0,
                    "potency_weight": 0.4,
                    "cost_weight": 0.3,
                    "diversity_weight": 0.1,
                    "llvm_block_weight": 0.15,
                    "branch_rewrite_weight": 0.1,
                    "ir_delta_weight": 0.15,
                },
                "baselines": {
                    "selected": ["random", "rule_based", "tigress"],
                    "random": {"seed": 7},
                    "rule_based": {
                        "operator_priority": ["flatten_cfg", "substitute_instructions", "split_blocks"]
                    },
                    "tigress": {
                        "tool_path": "tigress",
                        "command_template": [],
                    },
                },
            },
        )

        try:
            run_baselines.main(["--config", str(config_path), "--run-dir", str(run_dir)])
            summary = load_structured_file(run_dir / "reports" / "baseline_summary.json")
        finally:
            if config_path.exists():
                config_path.unlink()
            shutil.rmtree(run_dir, ignore_errors=True)

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["results"]["random"]["status"], "completed")
        self.assertEqual(summary["results"]["rule_based"]["status"], "completed")
        self.assertEqual(summary["results"]["random"]["completed_episode_count"], 1)
        self.assertEqual(summary["results"]["rule_based"]["completed_episode_count"], 1)
        self.assertEqual(summary["results"]["tigress"]["status"], "skipped_tool_missing")


if __name__ == "__main__":
    unittest.main()
