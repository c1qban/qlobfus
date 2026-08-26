from __future__ import annotations

import unittest
import uuid

from scripts.common import PROJECT_ROOT, ensure_dir, load_structured_file
from unittest.mock import patch

from scripts.generate_reference_consensus_harness import (
    parse_args,
    usable_seed_inputs_for_source,
    write_harnesses,
    write_per_sample_fallback_harnesses,
)


class GenerateReferenceConsensusHarnessTests(unittest.TestCase):
    def test_audit_failures_only_flag_is_available(self) -> None:
        args = parse_args(
            [
                "--processed-root",
                "processed",
                "--harness-root",
                "harness",
                "--harness-audit",
                "validation=audit.json",
                "--audit-failures-only",
                "--existing-harness-only",
            ]
        )

        self.assertTrue(args.audit_failures_only)
        self.assertTrue(args.existing_harness_only)

    def test_write_harnesses_can_skip_existing_and_stop_at_target(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"harness_write_{uuid.uuid4().hex}")
        existing = ensure_dir(root / "p1" / "C") / "s1.tests.json"
        existing.write_text('{"format":"existing","test_cases":[]}\n', encoding="utf-8")
        samples = [
            {"split_group": "p1", "relative_source_path": "p1/C/s1.c"},
            {"split_group": "p1", "relative_source_path": "p1/C/s2.c"},
            {"split_group": "p1", "relative_source_path": "p1/C/s3.c"},
        ]
        cases_by_problem = {"p1": [{"name": "case_1", "input_data": "", "expected_stdout": "0\n"}]}

        written = write_harnesses(
            root,
            samples,
            cases_by_problem,
            skip_existing=True,
            target_remaining=1,
        )

        self.assertEqual(written, 1)
        self.assertEqual(load_structured_file(existing)["format"], "existing")
        self.assertTrue((root / "p1" / "C" / "s2.tests.json").exists())
        self.assertFalse((root / "p1" / "C" / "s3.tests.json").exists())

    def test_strict_seed_inputs_filter_cases_too_short_for_first_scanf(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"harness_seed_{uuid.uuid4().hex}")
        source = root / "sample.c"
        source.write_text(
            '#include <stdio.h>\nint main(){int a,b,c;if(scanf("%d %d %d",&a,&b,&c)!=3)return 0;printf("%d\\n",a+b+c);}\n',
            encoding="utf-8",
        )

        seeds, audit = usable_seed_inputs_for_source(source, strict=True)

        self.assertEqual(audit["first_scanf_conversion_count"], 3)
        self.assertGreater(audit["filtered_seed_count"], 0)
        self.assertTrue(all(len(seed.replace(",", " ").split()) >= 3 for seed in seeds))

    def test_problem_specific_seed_is_prioritized(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"harness_problem_seed_{uuid.uuid4().hex}")
        source = root / "sample.c"
        source.write_text(
            '#include <stdio.h>\nint main(){int a;while(scanf("%d",&a)==1){}return 0;}\n',
            encoding="utf-8",
        )

        seeds, audit = usable_seed_inputs_for_source(source, strict=True, problem_id="p00511")

        self.assertEqual(seeds[0], "1\n=\n")
        self.assertEqual(audit["problem_seed_count"], 2)

    def test_write_harnesses_overwrites_forced_sample_even_when_skip_existing(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"harness_force_{uuid.uuid4().hex}")
        existing = ensure_dir(root / "p1" / "C") / "s1.tests.json"
        existing.write_text('{"format":"existing","test_cases":[]}\n', encoding="utf-8")
        samples = [
            {"sample_id": "p1/s1", "split_group": "p1", "relative_source_path": "p1/C/s1.c"},
            {"sample_id": "p1/s2", "split_group": "p1", "relative_source_path": "p1/C/s2.c"},
        ]
        cases_by_problem = {"p1": [{"name": "case_1", "input_data": "1 2 3\n", "expected_stdout": "6\n"}]}

        written = write_harnesses(
            root,
            samples,
            cases_by_problem,
            skip_existing=True,
            target_remaining=1,
            force_sample_ids={"p1/s1"},
        )

        self.assertEqual(written, 1)
        self.assertEqual(load_structured_file(existing)["format"], "executable_stdio_v1")
        self.assertFalse((root / "p1" / "C" / "s2.tests.json").exists())

    def test_fallback_writes_each_sample_to_its_own_harness_path(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"harness_fallback_{uuid.uuid4().hex}")
        samples = [
            {"sample_id": "p1/s1", "split_group": "p1", "relative_source_path": "p1/C/s1.c"},
            {"sample_id": "p2/s2", "split_group": "p2", "relative_source_path": "p2/C/s2.c"},
        ]

        def fake_fallback_cases_for_sample(**kwargs):
            sample_id = kwargs["sample"]["sample_id"]
            return [{"name": "case", "input_data": "1\n", "expected_stdout": sample_id}], {"sample_id": sample_id}

        with patch(
            "scripts.generate_reference_consensus_harness.fallback_cases_for_sample",
            side_effect=fake_fallback_cases_for_sample,
        ):
            written, _audits = write_per_sample_fallback_harnesses(
                harness_root=root,
                samples=samples,
                cases_by_problem={},
                build_root=root / "build",
                compiler="unused",
                max_cases=1,
                timeout_sec=1.0,
            )

        self.assertEqual(written, 2)
        first = load_structured_file(root / "p1" / "C" / "s1.tests.json")
        second = load_structured_file(root / "p2" / "C" / "s2.tests.json")
        self.assertEqual(first["test_cases"][0]["expected_stdout"], "p1/s1")
        self.assertEqual(second["test_cases"][0]["expected_stdout"], "p2/s2")


if __name__ == "__main__":
    unittest.main()
