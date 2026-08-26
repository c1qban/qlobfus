from __future__ import annotations

import unittest
import uuid

from rl.envs import GymObfuscationEnv
from scripts import evaluate_policy
from scripts.common import PROJECT_ROOT, ensure_dir, write_text


class _StopModel:
    def __init__(self) -> None:
        self.masks = []

    def predict(self, observation, *, deterministic, action_masks):
        self.masks.append(action_masks.tolist())
        stop_index = len(action_masks) - 1
        return stop_index, None


class EvaluatePolicyTests(unittest.TestCase):
    def test_evaluate_model_uses_masks_and_covers_each_sample_once(self) -> None:
        root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"evaluate_policy_{uuid.uuid4().hex}")
        source = root / "sample.c"
        write_text(source, "int main(void) { return 0; }\n")
        env = GymObfuscationEnv(
            env_cache={
                "job_name": "fixture",
                "split": "test",
                "sample_count": 2,
                "samples": [
                    {
                        "sample_id": sample_id,
                        "split": "test",
                        "split_group": sample_id,
                        "source_path": str(source),
                        "relative_source_path": source.name,
                        "source_stats": {"line_count": 1},
                        "compile": {},
                        "verification_input": {"case_count": 0, "test_cases": []},
                    }
                    for sample_id in ("p1/s1", "p2/s1")
                ],
            },
            max_steps=2,
            artifact_root=root / "episodes",
        )
        model = _StopModel()

        result = evaluate_policy.evaluate_model(model=model, env=env, deterministic=True, seed=7)

        self.assertEqual(result["episode_count"], 2)
        self.assertEqual(result["completed_episode_count"], 2)
        self.assertEqual([item["sample_id"] for item in result["episodes"]], ["p1/s1", "p2/s1"])
        self.assertEqual(len(model.masks), 2)
        self.assertTrue(all(mask[-1] == 1 for mask in model.masks))

    def test_main_refuses_train_split_before_loading_checkpoint(self) -> None:
        args = evaluate_policy.parse_args(
            [
                "--config",
                "config.json",
                "--checkpoint",
                "model.zip",
                "--split-index",
                "train.index.json",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.split_index, "train.index.json")
        self.assertTrue(args.deterministic)


if __name__ == "__main__":
    unittest.main()
