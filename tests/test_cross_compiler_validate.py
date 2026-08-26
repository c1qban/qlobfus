from __future__ import annotations

import unittest

from scripts import cross_compiler_validate


class CrossCompilerValidateTests(unittest.TestCase):
    def test_parse_profile_accepts_backend_compiler_and_flags(self) -> None:
        profile = cross_compiler_validate.parse_profile("gcc_O2=wsl:gcc:-O2 -std=c11")

        self.assertEqual(profile["name"], "gcc_O2")
        self.assertEqual(profile["backend"], "wsl")
        self.assertEqual(profile["compiler"], "gcc")
        self.assertEqual(profile["flags"], ["-O2", "-std=c11"])

    def test_parse_profile_rejects_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            cross_compiler_validate.parse_profile("bad=remote:gcc:-O0")

    def test_aggregate_groups_by_method_seed_and_profile(self) -> None:
        rows = [
            {
                "method": "maskable_ppo",
                "variant": "full_short",
                "seed": "1",
                "profile": "clang_O0",
                "compile_succeeded": True,
                "semantic_passed": True,
                "tests_passed": True,
                "diff_passed": True,
                "fuzz_passed": True,
                "semantic_score": 1.0,
                "failure_category": "",
            },
            {
                "method": "maskable_ppo",
                "variant": "full_short",
                "seed": "1",
                "profile": "clang_O0",
                "compile_succeeded": False,
                "semantic_passed": False,
                "tests_passed": False,
                "diff_passed": False,
                "fuzz_passed": False,
                "semantic_score": 0.0,
                "failure_category": "compile_failure",
            },
        ]

        aggregates = cross_compiler_validate.aggregate(rows)

        self.assertEqual(len(aggregates), 1)
        self.assertEqual(aggregates[0]["rows"], 2)
        self.assertEqual(aggregates[0]["compile_pass_rate"], 0.5)
        self.assertEqual(aggregates[0]["semantic_pass_rate"], 0.5)
        self.assertEqual(aggregates[0]["failure_breakdown"], {"compile_failure": 1})


if __name__ == "__main__":
    unittest.main()
