from __future__ import annotations

import shutil
import unittest
import uuid

from api.cli import main
from scripts import prepare_dataset
from scripts.common import PROJECT_ROOT, ensure_dir


class ApiCliTests(unittest.TestCase):
    def test_prepare_dataset_dry_run_dispatch(self) -> None:
        exit_code = main(
            [
                "prepare-dataset",
                "--config",
                "configs/benchmark/codenet_c.example.yaml",
                "--dry-run",
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_run_baselines_dispatch(self) -> None:
        prepare_dataset.main(["--config", "configs/benchmark/smoke_dataset.example.yaml"])
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        run_dir = temp_root / f"api_cli_baseline_{uuid.uuid4().hex}"
        try:
            exit_code = main(
                [
                    "run-baselines",
                    "--config",
                    "configs/benchmark/baselines_smoke.example.yaml",
                    "--run-dir",
                    str(run_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_sample_codenet_subset_dispatch(self) -> None:
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        fixture_root = temp_root / f"api_cli_subset_{uuid.uuid4().hex}"
        source_root = fixture_root / "Project_CodeNet"
        data_root = source_root / "data"
        metadata_root = source_root / "metadata"
        output_root = fixture_root / "subset_out"
        config_path = fixture_root / "subset_config.json"
        try:
            ensure_dir(data_root / "p00001" / "C")
            ensure_dir(metadata_root)
            (data_root / "p00001" / "C" / "s1001.c").write_text("int main(void) {\n    return 0;\n}\n", encoding="utf-8")
            (data_root / "p00001" / "C" / "s1002.c").write_text("int main(void) {\n    return 0;\n}\n", encoding="utf-8")
            (metadata_root / "p00001.csv").write_text(
                "submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy\n"
                "s1001,p00001,u001,0,C,C,c,Accepted,1,1,1,1/1\n"
                "s1002,p00001,u002,0,C,C,c,Accepted,1,1,1,1/1\n",
                encoding="utf-8",
            )
            (metadata_root / "problem_list.csv").write_text(
                "id,name,dataset,time_limit,memory_limit,rating,tags,complexity\n"
                "p00001,Problem 1,AIZU,1000,131072,,,\n",
                encoding="utf-8",
            )
            fixture_root.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "{\n"
                f'  "source_root": "{source_root.as_posix()}",\n'
                f'  "data_root": "{data_root.as_posix()}",\n'
                f'  "metadata_root": "{metadata_root.as_posix()}",\n'
                f'  "output_root": "{output_root.as_posix()}",\n'
                '  "problem_count": 1,\n'
                '  "samples_per_problem": 2,\n'
                '  "language_dir": "C",\n'
                '  "language_values": ["C"],\n'
                '  "filename_ext": "c",\n'
                '  "accepted_values": ["Accepted"],\n'
                '  "min_lines": 1,\n'
                '  "max_lines": 50\n'
                "}\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "sample-codenet-subset",
                    "--config",
                    str(config_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "subset_summary.json").exists())
        finally:
            shutil.rmtree(fixture_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
