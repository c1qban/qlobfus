from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from scripts import sample_project_codenet_subset
from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, load_structured_file, write_text


class ProjectCodeNetSubsetTests(unittest.TestCase):
    def test_subset_builder_copies_selected_c_samples_and_metadata(self) -> None:
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        fixture_root = temp_root / f"project_codenet_subset_fixture_{uuid.uuid4().hex}"
        source_root = fixture_root / "Project_CodeNet"
        data_root = source_root / "data"
        metadata_root = source_root / "metadata"
        descriptions_root = source_root / "problem_descriptions"
        output_root = fixture_root / "subset_out"
        config_path = fixture_root / "subset_config.json"

        ensure_dir(data_root / "p00001" / "C")
        ensure_dir(data_root / "p00002" / "C")
        ensure_dir(data_root / "p00003" / "C")
        ensure_dir(metadata_root)
        ensure_dir(descriptions_root)

        for name in ["s1001.c", "s1002.c"]:
            write_text(data_root / "p00001" / "C" / name, "int main(void) {\n    return 0;\n}\n")
            write_text(data_root / "p00002" / "C" / name.replace("1", "2", 1), "int main(void) {\n    return 0;\n}\n")
        write_text(data_root / "p00003" / "C" / "s3001.c", "int main(void) {\n    return 0;\n}\n")

        write_text(
            metadata_root / "p00001.csv",
            "submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy\n"
            "s1001,p00001,u001,0,C,C,c,Accepted,1,1,1,1/1\n"
            "s1002,p00001,u002,0,C,C,c,Accepted,1,1,1,1/1\n",
        )
        write_text(
            metadata_root / "p00002.csv",
            "submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy\n"
            "s2001,p00002,u003,0,C,C,c,Accepted,1,1,1,1/1\n"
            "s2002,p00002,u004,0,C,C,c,Accepted,1,1,1,1/1\n",
        )
        write_text(
            metadata_root / "p00003.csv",
            "submission_id,problem_id,user_id,date,language,original_language,filename_ext,status,cpu_time,memory,code_size,accuracy\n"
            "s3001,p00003,u005,0,C,C,c,Wrong Answer,1,1,1,0/1\n",
        )
        write_text(
            metadata_root / "problem_list.csv",
            "id,name,dataset,time_limit,memory_limit,rating,tags,complexity\n"
            "p00001,Problem 1,AIZU,1000,131072,,,\n"
            "p00002,Problem 2,AIZU,1000,131072,,,\n"
            "p00003,Problem 3,AIZU,1000,131072,,,\n",
        )
        write_text(descriptions_root / "p00001.html", "<html>Problem 1</html>\n")
        write_text(descriptions_root / "p00002.html", "<html>Problem 2</html>\n")

        dump_json(
            config_path,
            {
                "source_root": str(source_root),
                "data_root": str(data_root),
                "metadata_root": str(metadata_root),
                "problem_descriptions_root": str(descriptions_root),
                "output_root": str(output_root),
                "problem_count": 2,
                "samples_per_problem": 2,
                "language_dir": "C",
                "language_values": ["C"],
                "filename_ext": "c",
                "accepted_values": ["Accepted"],
                "min_lines": 1,
                "max_lines": 50,
            },
        )

        try:
            sample_project_codenet_subset.main(["--config", str(config_path)])
            summary = load_structured_file(output_root / "subset_summary.json")
            metadata_1 = (output_root / "metadata" / "p00001.csv").read_text(encoding="utf-8")
            metadata_2 = (output_root / "metadata" / "p00002.csv").read_text(encoding="utf-8")
            has_sample_1 = (output_root / "data" / "p00001" / "C" / "s1001.c").exists()
            has_sample_2 = (output_root / "data" / "p00002" / "C" / "s2002.c").exists()
            has_problem_3 = (output_root / "data" / "p00003").exists()
        finally:
            shutil.rmtree(fixture_root, ignore_errors=True)

        self.assertEqual(summary["problem_count_selected"], 2)
        self.assertEqual(summary["sample_count_copied"], 4)
        self.assertTrue(has_sample_1)
        self.assertTrue(has_sample_2)
        self.assertFalse(has_problem_3)
        self.assertIn("s1001", metadata_1)
        self.assertIn("s2002", metadata_2)


if __name__ == "__main__":
    unittest.main()
