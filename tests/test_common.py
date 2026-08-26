from __future__ import annotations

import unittest

from scripts.common import PROJECT_ROOT, load_config, slugify


class CommonTests(unittest.TestCase):
    def test_load_config_reads_json_compatible_yaml(self) -> None:
        path, config = load_config(PROJECT_ROOT / "configs" / "policy" / "mppo_c_ir.example.yaml")
        self.assertTrue(path.exists())
        self.assertEqual(config["policy"]["algorithm"], "maskable_ppo")

    def test_slugify_normalizes_text(self) -> None:
        self.assertEqual(slugify("MPPO C/IR Smoke"), "mppo_c_ir_smoke")


if __name__ == "__main__":
    unittest.main()

