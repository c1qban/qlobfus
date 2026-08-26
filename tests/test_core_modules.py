from __future__ import annotations

import shutil
import unittest
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

from obfuscator.operators import ClangAstNode, ClangAstSummary, LlvmIrBlock
from obfuscator.operators.c_source import build_operator_registry, collect_addition_candidates, collect_declaration_candidates, collect_literal_candidates
import obfuscator.operators.clang_ast as clang_ast_module
from obfuscator.operators.clang_ast import choose_clang_ast_binary, load_clang_ast_summary, parse_clang_cfg_dump, parse_clang_ast_json, parse_llvm_ir_text
from eval.reporting import build_evaluation_summary
from rl.envs import GymObfuscationEnv, ObfuscationEnv
from scripts.common import PROJECT_ROOT, dump_json, ensure_dir, write_text
from verifier.compile import choose_compiler
from verifier.fuzz.runner import _build_mutations, run_fuzz_campaign
from verifier.pipeline import VerificationPipeline


class CoreModuleTests(unittest.TestCase):
    def test_clang_command_output_uses_safe_utf8_decoding(self) -> None:
        completed = clang_ast_module._run_command(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'valid\\xfftext')"]
        )

        self.assertIsNotNone(completed)
        self.assertIn("valid", completed.stdout)
        self.assertIn("\ufffd", completed.stdout)

    def test_clang_cfg_and_llvm_ir_parsers_extract_block_ids(self) -> None:
        cfg_dump = """
int main()
 [B3]
   1: if [B3.0]
   Succs (2): B2 B1

 [B2]
   1: return;
   Succs (1): B1
"""
        llvm_ir = """
define dso_local i32 @main() !dbg !10 {
entry:
  br i1 %cmp, label %if.then, label %if.end, !dbg !21
if.then:
  br label %if.end, !dbg !22
if.end:
  ret i32 0
}
!21 = !DILocation(line: 7, column: 5, scope: !10)
!22 = !DILocation(line: 8, column: 3, scope: !10)
"""
        cfg_blocks = parse_clang_cfg_dump(cfg_dump)
        llvm_blocks = parse_llvm_ir_text(llvm_ir)

        self.assertEqual(cfg_blocks[0].block_id, "B3")
        self.assertEqual(cfg_blocks[0].successors, ["B2", "B1"])
        self.assertEqual(llvm_blocks[0].block_id, "entry")
        self.assertEqual(llvm_blocks[0].successors, ["if.then", "if.end"])
        self.assertEqual(llvm_blocks[0].line_no, 7)

    @unittest.skipIf(choose_clang_ast_binary() is None, "No clang binary is available for real AST/LLVM tests.")
    def test_real_clang_ast_summary_emits_llvm_blocks(self) -> None:
        source_path = PROJECT_ROOT / "data" / "raw" / "smoke" / "sum_stdin.c"
        summary = load_clang_ast_summary(source_path.read_text(encoding="utf-8"))

        self.assertTrue(summary.available)
        self.assertTrue(summary.llvm_ir_available)
        self.assertGreater(len(summary.llvm_ir_blocks), 0)
        self.assertGreater(len(summary.branches), 0)
        self.assertEqual(summary.branches[0].metadata["llvm_block_id"], "entry")

    def test_parse_clang_ast_summary_enriches_branch_with_cfg_and_llvm_ids(self) -> None:
        ast_payload = {
            "kind": "TranslationUnitDecl",
            "inner": [
                {
                    "kind": "FunctionDecl",
                    "name": "main",
                    "loc": {"line": 1, "col": 1},
                    "inner": [
                        {
                            "kind": "IfStmt",
                            "loc": {"line": 7, "col": 5},
                            "inner": [],
                        }
                    ],
                }
            ],
        }
        summary = parse_clang_ast_json(ast_payload, compiler_path="clang")
        summary.cfg_blocks = parse_clang_cfg_dump(
            """
int main()
 [B3]
   1: if [B3.0]
   Succs (2): B2 B1
"""
        )
        summary.cfg_available = True
        summary.llvm_ir_blocks = parse_llvm_ir_text(
            """
define dso_local i32 @main() !dbg !10 {
entry:
  br i1 %cmp, label %if.then, label %if.end, !dbg !21
if.then:
  ret i32 1
if.end:
  ret i32 0
}
!21 = !DILocation(line: 7, column: 5, scope: !10)
"""
        )
        summary.llvm_ir_available = True

        clang_ast_module._enrich_nodes(summary)

        branch = summary.branches[0]
        self.assertEqual(branch.metadata["cfg_block_id"], "B3")
        self.assertEqual(branch.metadata["llvm_block_id"], "entry")
        self.assertEqual(branch.metadata["analysis_backend"], "clang_ast_json+clang_cfg+llvm_ir")

    @patch("obfuscator.operators.c_source.load_clang_ast_summary")
    def test_collectors_prefer_clang_ast_backend_when_available(self, mock_load_clang_ast_summary) -> None:
        source_text = """int main(void) {
    int value = 7;
    int other = value + value;
    return other;
}
"""
        mock_load_clang_ast_summary.return_value = ClangAstSummary(
            available=True,
            backend="clang_ast_json",
            compiler_path="clang",
            declarations=[
                ClangAstNode(kind="VarDecl", function_name="main", line_no=2, column_no=13, detail="value", metadata={"analysis_backend": "clang_ast_json+llvm_ir", "llvm_block_id": "entry"}),
                ClangAstNode(kind="VarDecl", function_name="main", line_no=3, column_no=13, detail="other", metadata={"analysis_backend": "clang_ast_json+llvm_ir", "llvm_block_id": "entry"}),
            ],
            additions=[
                ClangAstNode(kind="BinaryOperator", function_name="main", line_no=3, column_no=17, metadata={"analysis_backend": "clang_ast_json+llvm_ir", "llvm_block_id": "entry"}),
            ],
            literals=[
                ClangAstNode(kind="IntegerLiteral", function_name="main", line_no=2, column_no=17, detail="7", metadata={"analysis_backend": "clang_ast_json+llvm_ir", "llvm_block_id": "entry"}),
            ],
        )

        declarations = collect_declaration_candidates(source_text)
        additions = collect_addition_candidates(source_text)
        literals = collect_literal_candidates(source_text)

        self.assertEqual(declarations[0].column_no, 13)
        self.assertEqual(declarations[0].metadata["analysis_backend"], "clang_ast_json+llvm_ir")
        self.assertEqual(declarations[0].metadata["clang_kind"], "VarDecl")
        self.assertEqual(declarations[0].metadata["llvm_block_id"], "entry")
        self.assertEqual(additions[0].column_no, 17)
        self.assertEqual(additions[0].metadata["clang_kind"], "BinaryOperator")
        self.assertEqual(literals[0].column_no, 17)
        self.assertEqual(literals[0].metadata["clang_kind"], "IntegerLiteral")

    def test_source_artifact_persistence_recreates_missing_episode_directory(self) -> None:
        run_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"episode_artifacts_{uuid.uuid4().hex}")
        try:
            env = ObfuscationEnv(artifact_root=run_dir)
            env._episode_root = run_dir / "missing_episode"

            artifact_path = env._persist_source_artifact("int main(void){return 0;}\n", 1, "rename/locals")

            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact_path.parent, env._episode_root)
            self.assertIn("rename_locals", artifact_path.name)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_source_artifact_names_are_short_for_windows_paths(self) -> None:
        run_dir = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp" / f"episode_artifacts_{uuid.uuid4().hex}")
        try:
            env = ObfuscationEnv(artifact_root=run_dir)
            env._episode_root = run_dir / "p00284_s649875354_20260612_185555_244326"
            long_label = "rename_locals_ast_local_slot_04_compact_prefix"

            artifact_path = env._persist_source_artifact("int main(void){return 0;}\n", 1, long_label)

            self.assertLessEqual(len(artifact_path.name), 40)
            self.assertTrue(artifact_path.name.startswith("step_01_rename_locals_ast"))
            self.assertTrue(artifact_path.exists())
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    @patch("obfuscator.operators.c_source.load_clang_ast_summary")
    def test_collectors_bind_fallback_candidates_to_nearest_llvm_block(self, mock_load_clang_ast_summary) -> None:
        source_text = """int main(void) {
    int value = 7;
    return value;
}
"""
        mock_load_clang_ast_summary.return_value = ClangAstSummary(
            available=True,
            backend="clang_ast_json",
            compiler_path="clang",
            llvm_ir_available=True,
            llvm_ir_blocks=[
                LlvmIrBlock(function_name="main", block_id="entry", branch_ordinal=1, line_no=2, column_no=5),
                LlvmIrBlock(function_name="main", block_id="return", branch_ordinal=2, line_no=3, column_no=5),
            ],
        )

        declarations = collect_declaration_candidates(source_text)
        literals = collect_literal_candidates(source_text)

        self.assertEqual(declarations[0].metadata["llvm_block_id"], "entry")
        self.assertEqual(declarations[0].metadata["llvm_binding_strategy"], "nearest_source_line")
        self.assertEqual(literals[0].metadata["llvm_block_id"], "entry")
        self.assertIn("llvm_ir_nearest", str(literals[0].metadata["analysis_backend"]))

    @patch("obfuscator.operators.c_source.load_clang_ast_summary")
    def test_collectors_cover_uninitialized_declarations_and_float_literals(self, mock_load_clang_ast_summary) -> None:
        source_text = """int main(void) {
    double a, b, c;
    float threshold = 2.5f;
    return threshold > 1.0 ? 0 : 1;
}
"""
        mock_load_clang_ast_summary.return_value = ClangAstSummary(
            available=True,
            backend="clang_ast_json",
            compiler_path="clang",
            llvm_ir_available=True,
            llvm_ir_blocks=[
                LlvmIrBlock(function_name="main", block_id="entry", branch_ordinal=1, line_no=2, column_no=5),
                LlvmIrBlock(function_name="main", block_id="return", branch_ordinal=2, line_no=4, column_no=5),
            ],
        )

        declarations = collect_declaration_candidates(source_text)
        literals = collect_literal_candidates(source_text)
        registry = build_operator_registry()
        rename_actions = registry["rename_locals"].enumerate_actions(source_text)
        split_actions = registry["split_blocks"].enumerate_actions(source_text)
        literal_actions = registry["encode_literals"].enumerate_actions(source_text)

        self.assertEqual([candidate.metadata["name"] for candidate in declarations[:3]], ["a", "b", "c"])
        self.assertFalse(bool(declarations[0].metadata["is_initialized"]))
        self.assertTrue(any(candidate.metadata["literal_kind"] == "float" for candidate in literals))
        self.assertTrue(rename_actions)
        self.assertTrue(any(action.action_id == "rename_locals:ast_local_bulk:compact_prefix" and action.parameters.get("llvm_block_ids") for action in rename_actions))
        self.assertTrue(any(action.action_id == "split_blocks:ast_decl_slot_04:single" for action in split_actions))
        float_action = next(action for action in literal_actions if str(action.parameters.get("literal_kind")) == "float")
        application = registry["encode_literals"].apply(source_text, float_action)
        self.assertTrue(application.applied)
        self.assertIn("+ 3.0f", application.source_text)

    @patch("obfuscator.operators.c_source.load_clang_ast_summary")
    def test_flatten_cfg_propagates_cfg_and_llvm_block_ids(self, mock_load_clang_ast_summary) -> None:
        mock_load_clang_ast_summary.return_value = ClangAstSummary(
            available=True,
            backend="clang_ast_json",
            compiler_path="clang",
            branches=[
                ClangAstNode(
                    kind="IfStmt",
                    function_name="main",
                    line_no=7,
                    column_no=5,
                    metadata={
                        "analysis_backend": "clang_ast_json+clang_cfg+llvm_ir",
                        "cfg_block_id": "B3",
                        "cfg_successor_ids": ["B2", "B1"],
                        "llvm_block_id": "entry",
                        "llvm_successor_ids": ["if.then", "if.end"],
                    },
                )
            ],
        )
        env = ObfuscationEnv(max_steps=3)
        state = env.reset(sample_id="sample_a")
        flatten_action = next(action for action in state.legal_actions if action.startswith("flatten_cfg:"))

        next_state, reward, done, info = env.step(flatten_action)
        self.assertGreater(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["operator_metadata"]["cfg_block_id"], "B3")
        self.assertEqual(info["operator_metadata"]["llvm_block_id"], "entry")
        self.assertEqual(info["operator_metadata"]["parameters"]["analysis_backend"], "clang_ast_json+clang_cfg+llvm_ir")
        self.assertGreater(info["reward_breakdown"]["llvm_block_component"], 0.0)
        self.assertGreater(info["reward_breakdown"]["branch_rewrite_component"], 0.0)
        self.assertGreater(info["llvm_block_coverage"], 0.0)
        self.assertGreaterEqual(info["branch_rewrite_ratio"], 1.0)

    def test_obfuscation_env_step(self) -> None:
        env = ObfuscationEnv(max_steps=3)
        state = env.reset(sample_id="sample_a")
        self.assertEqual(state.remaining_budget, 3)
        self.assertGreaterEqual(state.semantic_score, 0.97)
        rename_action = next(action for action in state.legal_actions if action.startswith("rename_locals:"))

        next_state, reward, done, info = env.step(rename_action)
        self.assertEqual(next_state.step_index, 1)
        self.assertGreater(reward, 0.0)
        self.assertFalse(done)
        self.assertTrue(info["action_valid"])
        self.assertTrue(any(entry.startswith(f"{rename_action}@") for entry in next_state.action_history))
        self.assertEqual(info["action_mask"][-1], 1)
        self.assertTrue(next_state.current_source_path.endswith(".c"))
        self.assertIn("verification_summary", info)
        self.assertIn("reward_breakdown", info)
        self.assertIn("trace_path", info)
        self.assertEqual(info["resolved_action"], rename_action)
        self.assertIn("node_id", info["operator_metadata"])
        trace_path = PROJECT_ROOT / info["trace_path"]
        trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
        final_event = trace_lines[-1]
        self.assertIn('"potency_score"', final_event)
        self.assertIn('"cost_penalty"', final_event)

    def test_early_stop_requires_a_successful_transformation_when_actions_exist(self) -> None:
        env = ObfuscationEnv(
            max_steps=3,
            early_stop_config={
                "enabled": True,
                "min_successful_actions": 1,
                "allow_if_no_transform_actions": True,
            },
        )
        state = env.reset(sample_id="sample_a")
        self.assertNotIn("stop", state.legal_actions)
        self.assertEqual(env.current_action_mask()[-1], 0)

        invalid_state, reward, done, info = env.step("stop")
        self.assertTrue(done)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["termination_reason"], "invalid_action")
        self.assertEqual(invalid_state.action_history, [])

        state = env.reset(sample_id="sample_b")
        action = next(item for item in state.legal_actions if item != "stop")
        next_state, _reward, done, _info = env.step(action)
        self.assertFalse(done)
        self.assertIn("stop", next_state.legal_actions)
        self.assertEqual(env.current_action_mask()[-1], 1)

    def test_rename_locals_uses_stable_unique_targets_across_steps(self) -> None:
        source_text = """int main(void) {
    double a, b;
    return 0;
}
"""
        registry = build_operator_registry()
        first_action = next(
            action
            for action in registry["rename_locals"].enumerate_actions(source_text)
            if action.action_id.startswith("rename_locals:ast_local_slot_01")
        )
        first_application = registry["rename_locals"].apply(source_text, first_action)
        second_action = next(
            action
            for action in registry["rename_locals"].enumerate_actions(first_application.source_text)
            if action.action_id.startswith("rename_locals:ast_local_slot_01")
        )
        second_application = registry["rename_locals"].apply(first_application.source_text, second_action)

        self.assertTrue(first_application.applied)
        self.assertTrue(second_application.applied)
        self.assertIn("__ql_local_a", second_application.source_text)
        self.assertIn("__ql_local_b", second_application.source_text)
        self.assertNotIn("double __ql_local_a, __ql_local_a", second_application.source_text)

    def test_rename_locals_does_not_rewrite_preprocessor_includes(self) -> None:
        source_text = """#include<stdio.h>

int main(void) {
    int h = 1;
    return h;
}
"""
        registry = build_operator_registry()
        action = next(
            action
            for action in registry["rename_locals"].enumerate_actions(source_text)
            if action.action_id.startswith("rename_locals:ast_local_slot_01")
        )
        application = registry["rename_locals"].apply(source_text, action)

        self.assertTrue(application.applied)
        self.assertIn("#include<stdio.h>", application.source_text)
        self.assertIn("__ql_local_h", application.source_text)

    def test_flatten_cfg_skips_nested_return_without_top_level_final_return(self) -> None:
        source_text = """long int LCM(long int a, long int b){
  int r;
  if (b==0){
    return a;
  }
  else {
    r = a % b;
    return LCM(b,r);
  }
}
"""
        registry = build_operator_registry()

        self.assertEqual(registry["flatten_cfg"].enumerate_actions(source_text), [])

    def test_action_safety_can_filter_high_risk_actions_from_mask(self) -> None:
        env = ObfuscationEnv(
            max_steps=2,
            action_safety_config={
                "enabled": True,
                "filter_risky_actions": True,
                "max_allowed_risk": 0.1,
                "penalize_bulk_without_harness": True,
            },
        )
        state = env.reset(sample_id="sample_a")

        self.assertFalse(any(action.endswith("_bulk:offset_7") for action in state.legal_actions))
        self.assertTrue(env._filtered_action_risks)
        self.assertTrue(all("risk_score" in item for item in env._filtered_action_risks.values()))

    def test_action_safety_blocks_threshold_equal_high_risk_actions(self) -> None:
        env = ObfuscationEnv(
            max_steps=2,
            action_safety_config={
                "enabled": True,
                "filter_risky_actions": True,
                "max_allowed_risk": 0.7,
                "blocked_risk_levels": ["high"],
                "blocked_parameter_tags": ["neg_add"],
            },
        )
        state = env.reset(sample_id="sample_a")

        self.assertFalse(any(":neg_add" in action for action in state.legal_actions))
        self.assertTrue(any(":neg_add" in action for action in env._filtered_action_risks))

    def test_obfuscation_env_loads_env_cache(self) -> None:
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        cache_dir = temp_root / f"env_cache_{uuid.uuid4().hex}"
        ensure_dir(cache_dir)
        env_cache_path = cache_dir / "env_input_cache.json"
        dump_json(
            env_cache_path,
            {
                "job_name": "unit_test_dataset",
                "split": "train",
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sum_stdin",
                        "split": "train",
                        "split_group": "sum_stdin",
                        "source_path": "data/raw/smoke/sum_stdin.c",
                        "relative_source_path": "sum_stdin.c",
                        "source_stats": {"line_count": 14},
                        "compile": {"compiler": None, "compiler_flags": ["-O0"], "timeout_sec": 10.0},
                        "verification_input": {
                            "harness_format": "executable_stdio_v1",
                            "harness_path": "data/harness/smoke/sum_stdin.tests.json",
                            "case_count": 2,
                            "test_cases": [
                                {
                                    "name": "adds_small_numbers",
                                    "input_data": "1 2\n",
                                    "expected_stdout": "3\n",
                                    "expected_returncode": 0,
                                }
                            ],
                        },
                    }
                ],
            },
        )

        try:
            env = ObfuscationEnv.from_env_cache(env_cache_path, max_steps=4)

            self.assertEqual(env.available_sample_ids(), ["sum_stdin"])
            state = env.reset()
            self.assertEqual(state.sample_id, "sum_stdin")
            self.assertEqual(state.test_case_count, 2)
            self.assertIn("split_blocks:ast_decl_slot_01:single", state.legal_actions)
            self.assertIn("split_blocks:ast_decl_slot_02:single", state.legal_actions)
            self.assertIn("encode_literals:ast_literal_slot_01:offset_3", state.legal_actions)
            flatten_action = next(action for action in state.legal_actions if action.startswith("flatten_cfg:"))
            self.assertIn("stop", state.legal_actions)

            next_state, reward, done, info = env.step(flatten_action)
            self.assertGreater(reward, 0.1)
            self.assertFalse(done)
            self.assertEqual(next_state.sample_id, "sum_stdin")
            self.assertTrue(any(entry.startswith(f"{flatten_action}@") for entry in next_state.action_history))
            self.assertEqual(info["bound_job_name"], "unit_test_dataset")
            self.assertEqual(info["verification_case_count"], 2)
            self.assertTrue(info["verification_summary"]["compile"]["succeeded"])
            self.assertGreaterEqual(info["verification_summary"]["semantic_score"], env.semantic_threshold)
            self.assertTrue(info["verification_summary"]["diff"]["executed"])
            self.assertTrue(info["verification_summary"]["fuzz"]["executed"])
            self.assertTrue(str(info["operator_metadata"]["node_id"]).startswith("cfg.if."))
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    def test_gym_obfuscation_env_exposes_gymnasium_step_signature(self) -> None:
        env_cache = {
            "job_name": "unit_test_dataset",
            "split": "train",
            "sample_count": 1,
            "samples": [
                {
                    "sample_id": "sum_stdin",
                    "split": "train",
                    "split_group": "sum_stdin",
                    "source_path": "data/raw/smoke/sum_stdin.c",
                    "relative_source_path": "sum_stdin.c",
                    "source_stats": {"line_count": 14},
                    "compile": {"compiler": None, "compiler_flags": ["-O0"], "timeout_sec": 10.0},
                    "verification_input": {
                        "harness_format": "executable_stdio_v1",
                        "harness_path": "data/harness/smoke/sum_stdin.tests.json",
                        "case_count": 2,
                        "test_cases": [],
                    },
                }
            ],
        }
        env = GymObfuscationEnv(env_cache=env_cache, max_steps=4)

        observation, info = env.reset()
        self.assertEqual(observation.shape, (12,))
        self.assertTrue(np.issubdtype(observation.dtype, np.floating))
        self.assertEqual(info["sample_id"], "sum_stdin")
        self.assertEqual(len(info["action_mask"]), env.action_space.n)
        self.assertIn("verification_summary", info)

        next_observation, reward, terminated, truncated, step_info = env.step(0)
        self.assertEqual(next_observation.shape, (12,))
        self.assertGreaterEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(step_info["action_name"].startswith("flatten_cfg:"))
        self.assertIn("node_id", step_info["operator_metadata"])
        self.assertEqual(len(env.action_masks()), env.action_space.n)
        self.assertTrue(step_info["compile_succeeded"])

    def test_verification_stub_passes_semantic_gate(self) -> None:
        summary = VerificationPipeline(semantic_threshold=0.97).run_stub(sample_id="sample_b")
        self.assertTrue(summary.compile.succeeded)
        self.assertTrue(summary.tests.passed)
        self.assertGreaterEqual(summary.semantic_score, summary.semantic_threshold)

    def test_evaluation_summary_embeds_verification(self) -> None:
        verification = VerificationPipeline(semantic_threshold=0.95).run_stub(sample_id="sample_c")
        summary = build_evaluation_summary(
            job_name="eval_job",
            benchmark={"datasets": ["juliet"]},
            metrics=["semantic"],
            training_context={"algorithm": "maskable_ppo"},
            verification=verification,
        )
        self.assertIn("semantic_verification", summary)
        self.assertTrue(summary["derived_flags"]["passes_semantic_gate"])

    def test_evaluation_summary_aggregates_llvm_trace_metrics(self) -> None:
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        run_dir = temp_root / f"llvm_eval_{uuid.uuid4().hex}"
        trace_dir = ensure_dir(run_dir / "artifacts" / "env_workers" / "worker_00" / "sample_episode")
        write_text(
            trace_dir / "transform_trace.jsonl",
            "\n".join(
                [
                    '{"event":"reset","sample_id":"sum_stdin","ir_metrics":{"llvm_metric_available":true,"baseline_llvm_block_count":2,"current_llvm_block_count":2,"touched_llvm_block_count":0,"baseline_branch_block_count":1,"rewritten_branch_block_count":0,"llvm_block_coverage":0.0,"branch_rewrite_ratio":0.0,"ir_structural_delta":0.0}}',
                    '{"event":"step","reward_breakdown":{"ir_metrics":{"llvm_metric_available":true,"baseline_llvm_block_count":2,"current_llvm_block_count":3,"touched_llvm_block_count":1,"baseline_branch_block_count":1,"rewritten_branch_block_count":1,"llvm_block_coverage":0.5,"branch_rewrite_ratio":1.0,"ir_structural_delta":0.5}}}',
                ]
            )
            + "\n",
        )
        try:
            summary = build_evaluation_summary(
                job_name="eval_job",
                benchmark={"datasets": ["juliet"]},
                metrics=["semantic", "ir"],
                verification=VerificationPipeline(semantic_threshold=0.95).run_stub(sample_id="sample_c"),
                run_dir=run_dir,
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        self.assertEqual(summary["llvm_ir_metrics"]["llvm_metric_episode_count"], 1)
        self.assertEqual(summary["llvm_ir_metrics"]["mean_llvm_block_coverage"], 0.5)
        self.assertEqual(summary["llvm_ir_metrics"]["mean_branch_rewrite_ratio"], 1.0)
        self.assertEqual(summary["llvm_ir_metrics"]["mean_ir_structural_delta"], 0.5)
        self.assertTrue(summary["derived_flags"]["llvm_metrics_available"])

    def test_evaluation_summary_aggregates_obfuscation_trace_metrics(self) -> None:
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        run_dir = temp_root / f"obf_eval_{uuid.uuid4().hex}"
        trace_dir = ensure_dir(run_dir / "artifacts" / "env_workers" / "worker_00" / "sample_episode")
        write_text(
            trace_dir / "transform_trace.jsonl",
            "\n".join(
                [
                    '{"event":"reset","sample_id":"sum_stdin"}',
                    '{"event":"step","action":"rename_locals:slot:param","action_valid":true,"operator_metadata":{"operator_name":"rename_locals"},"potency_score":0.25,"cost_penalty":0.1,"reward_breakdown":{"semantic_gate_passed":true},"termination_reason":"continue","verification":{"semantic_score":1.0,"verifier_coverage":0.9,"compile":{"succeeded":true}}}',
                    '{"event":"step","action":"stop","action_valid":true,"operator_metadata":{},"potency_score":0.4,"cost_penalty":0.2,"reward_breakdown":{"semantic_gate_passed":true},"termination_reason":"stop","verification":{"semantic_score":1.0,"verifier_coverage":0.9,"compile":{"succeeded":true}}}',
                ]
            )
            + "\n",
        )
        try:
            summary = build_evaluation_summary(
                job_name="eval_job",
                benchmark={"datasets": ["codenet"]},
                metrics=["semantic", "potency", "cost"],
                verification=VerificationPipeline(semantic_threshold=0.95).run_stub(sample_id="sample_c"),
                run_dir=run_dir,
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        self.assertEqual(summary["obfuscation_metrics"]["episode_count"], 1)
        self.assertEqual(summary["obfuscation_metrics"]["mean_potency_score"], 0.4)
        self.assertEqual(summary["obfuscation_metrics"]["mean_cost_penalty"], 0.2)
        self.assertEqual(summary["obfuscation_metrics"]["mean_applied_action_count"], 1.0)
        self.assertEqual(summary["obfuscation_metrics"]["semantic_gate_pass_rate"], 1.0)
        self.assertTrue(summary["derived_flags"]["obfuscation_metrics_available"])

    def test_evaluation_summary_aggregates_resilience_proxy_metrics(self) -> None:
        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        run_dir = temp_root / f"res_eval_{uuid.uuid4().hex}"
        trace_dir = ensure_dir(run_dir / "artifacts" / "env_workers" / "worker_00" / "sample_episode")
        write_text(
            trace_dir / "transform_trace.jsonl",
            "\n".join(
                [
                    '{"event":"reset","sample_id":"sum_stdin"}',
                    '{"event":"step","action":"flatten_cfg:cfg_branch_slot_01:for_switch","action_valid":true,"operator_metadata":{"operator_name":"flatten_cfg"},"reward_breakdown":{"semantic_gate_passed":true,"ir_metrics":{"branch_rewrite_ratio":1.0,"ir_structural_delta":0.5,"llvm_block_coverage":0.5}},"verification":{"semantic_score":1.0,"compile":{"succeeded":true}}}',
                    '{"event":"step","action":"encode_literals:slot:param","action_valid":true,"operator_metadata":{"operator_name":"encode_literals"},"reward_breakdown":{"semantic_gate_passed":true,"ir_metrics":{"branch_rewrite_ratio":1.0,"ir_structural_delta":0.5,"llvm_block_coverage":0.5}},"verification":{"semantic_score":1.0,"compile":{"succeeded":true}}}',
                    '{"event":"step","action":"stop","action_valid":true,"operator_metadata":{},"reward_breakdown":{"semantic_gate_passed":true,"ir_metrics":{"branch_rewrite_ratio":1.0,"ir_structural_delta":0.5,"llvm_block_coverage":0.5}},"verification":{"semantic_score":1.0,"compile":{"succeeded":true}}}',
                ]
            )
            + "\n",
        )
        try:
            summary = build_evaluation_summary(
                job_name="eval_job",
                benchmark={"datasets": ["codenet"]},
                metrics=["semantic", "resilience"],
                verification=VerificationPipeline(semantic_threshold=0.95).run_stub(sample_id="sample_c"),
                run_dir=run_dir,
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        self.assertEqual(summary["resilience_metrics"]["episode_count"], 1)
        self.assertTrue(summary["resilience_metrics"]["proxy_based"])
        self.assertEqual(summary["resilience_metrics"]["flatten_cfg_episode_rate"], 1.0)
        self.assertEqual(summary["resilience_metrics"]["mean_operator_diversity_ratio"], 1.0)
        self.assertGreater(summary["resilience_metrics"]["mean_proxy_resilience_score"], 0.0)
        self.assertTrue(summary["derived_flags"]["resilience_metrics_available"])

    @unittest.skipIf(choose_compiler() is None, "No supported compiler is available for real verification tests.")
    def test_verification_pipeline_compiles_and_runs_real_tests(self) -> None:
        source_path = PROJECT_ROOT / "data" / "raw" / "smoke" / "sum_stdin.c"
        test_cases = [
            {
                "name": "adds_pair",
                "input_data": "2 5\n",
                "expected_stdout": "7\n",
                "expected_returncode": 0,
            },
            {
                "name": "adds_zero",
                "input_data": "0 0\n",
                "expected_stdout": "0\n",
                "expected_returncode": 0,
            },
        ]

        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        build_root = temp_root / f"run_{uuid.uuid4().hex}"
        ensure_dir(build_root)
        try:
            pipeline = VerificationPipeline(
                semantic_threshold=0.97,
                build_root=build_root,
            )
            summary = pipeline.verify_source(
                source_path=source_path,
                test_cases=test_cases,
                sample_id="sum_stdin_test",
            )
        finally:
            shutil.rmtree(build_root, ignore_errors=True)

        self.assertTrue(summary.compile.succeeded)
        self.assertTrue(summary.tests.passed)
        self.assertEqual(summary.tests.passed_cases, 2)
        self.assertGreaterEqual(summary.semantic_score, summary.semantic_threshold)
        self.assertIn("compile", summary.available_signals)
        self.assertIn("tests", summary.available_signals)

    @unittest.skipIf(choose_compiler() is None, "No supported compiler is available for differential verification tests.")
    def test_verification_pipeline_runs_diff_and_fuzz(self) -> None:
        source_path = PROJECT_ROOT / "data" / "raw" / "smoke" / "sum_stdin.c"
        test_cases = [
            {
                "name": "adds_pair",
                "input_data": "2 5\n",
                "expected_stdout": "7\n",
                "expected_returncode": 0,
            },
            {
                "name": "adds_negative",
                "input_data": "-3 10\n",
                "expected_stdout": "7\n",
                "expected_returncode": 0,
            },
        ]

        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        build_root = temp_root / f"diff_run_{uuid.uuid4().hex}"
        ensure_dir(build_root)
        try:
            pipeline = VerificationPipeline(
                semantic_threshold=0.97,
                build_root=build_root,
            )
            summary = pipeline.verify_source(
                source_path=source_path,
                reference_source_path=source_path,
                test_cases=test_cases,
                sample_id="sum_stdin_diff_test",
            )
        finally:
            shutil.rmtree(build_root, ignore_errors=True)

        self.assertTrue(summary.compile.succeeded)
        self.assertTrue(summary.tests.passed)
        self.assertTrue(summary.diff.executed)
        self.assertTrue(summary.diff.passed)
        self.assertTrue(summary.fuzz.executed)
        self.assertTrue(summary.fuzz.passed)
        self.assertEqual(summary.fuzz_diff_error_rate, 0.0)
        self.assertIn("functional_similarity", summary.available_signals)
        self.assertIn("fuzz", summary.available_signals)

    @unittest.skipIf(choose_compiler() is None, "No supported compiler is available for differential verification tests.")
    def test_verification_pipeline_credits_weak_expected_harness_when_diff_matches(self) -> None:
        source_path = PROJECT_ROOT / "data" / "raw" / "smoke" / "sum_stdin.c"
        test_cases = [
            {
                "name": "bad_expected_but_reference_consistent",
                "input_data": "2 5\n",
                "expected_stdout": "999\n",
                "expected_returncode": 0,
            }
        ]

        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        build_root = temp_root / f"weak_harness_run_{uuid.uuid4().hex}"
        ensure_dir(build_root)
        try:
            pipeline = VerificationPipeline(
                semantic_threshold=0.97,
                build_root=build_root,
            )
            summary = pipeline.verify_source(
                source_path=source_path,
                reference_source_path=source_path,
                test_cases=test_cases,
                sample_id="sum_stdin_weak_harness_test",
            )
        finally:
            shutil.rmtree(build_root, ignore_errors=True)

        self.assertTrue(summary.diff.executed)
        self.assertTrue(summary.diff.passed)
        self.assertTrue(summary.tests.passed)
        self.assertIn("credited by differential execution", "; ".join(summary.tests.cases[0].notes))
        self.assertGreaterEqual(summary.semantic_score, summary.semantic_threshold)

    def test_fuzz_mutations_keep_positive_integer_seeds_in_domain(self) -> None:
        from scripts.run_baselines import build_external_fuzz_mutations

        mutations = dict(_build_mutations("1 1\n7 13\n100 10\n"))

        self.assertNotIn("negate_first_int", mutations)
        self.assertNotIn("append_zero_token", mutations)
        self.assertNotIn("decrement_first_int", mutations)
        self.assertNotIn("swap_first_two_ints", dict(_build_mutations("2 -1 -2 -1 -1 -5\n")))
        self.assertEqual(mutations["increment_first_int"], "2 1\n7 13\n100 10\n")

        binary_grid = (
            "100000000000\n"
            "000000000000\n"
            "011000000000\n"
            "000000000000\n"
        )
        self.assertEqual(_build_mutations(binary_grid), [])
        self.assertEqual(build_external_fuzz_mutations(binary_grid), [])

    def test_fuzz_reference_failures_are_not_semantic_mismatches(self) -> None:
        import subprocess
        import verifier.fuzz.runner as fuzz_runner

        def fake_execute(executable_path, _case, _input_data, _workdir):
            name = Path(executable_path).name
            if "reference" in name:
                return subprocess.CompletedProcess([name], 124, "", "\nExecution timed out.")
            return subprocess.CompletedProcess([name], 0, "candidate output\n", "")

        with patch.object(fuzz_runner, "_execute_case", side_effect=fake_execute):
            result = run_fuzz_campaign(
                Path("reference.exe"),
                Path("candidate.exe"),
                [fuzz_runner.TestCase(name="seed", input_data="1\n")],
            )

        self.assertTrue(result.executed)
        self.assertTrue(result.passed)
        self.assertEqual(result.failed_cases, 0)
        self.assertEqual(result.diff_error_rate, 0.0)
        self.assertIn("reference failed under fuzz input", "; ".join(result.cases[0].notes))

    def test_fuzz_candidate_only_failures_still_count(self) -> None:
        import subprocess
        import verifier.fuzz.runner as fuzz_runner

        def fake_execute(executable_path, _case, _input_data, _workdir):
            name = Path(executable_path).name
            if "reference" in name:
                return subprocess.CompletedProcess([name], 0, "ok\n", "")
            return subprocess.CompletedProcess([name], 124, "", "\nExecution timed out.")

        with patch.object(fuzz_runner, "_execute_case", side_effect=fake_execute):
            result = run_fuzz_campaign(
                Path("reference.exe"),
                Path("candidate.exe"),
                [fuzz_runner.TestCase(name="seed", input_data="1\n")],
            )

        self.assertTrue(result.executed)
        self.assertFalse(result.passed)
        self.assertEqual(result.failed_cases, 1)
        self.assertEqual(result.diff_error_rate, 1.0)
        self.assertIn("return code diverged under fuzz input", "; ".join(result.cases[0].notes))

    @unittest.skipIf(choose_compiler() is None, "No supported compiler is available for fuzz-disable tests.")
    def test_verification_pipeline_can_disable_fuzz(self) -> None:
        source_path = PROJECT_ROOT / "data" / "raw" / "smoke" / "sum_stdin.c"
        test_cases = [
            {
                "name": "adds_pair",
                "input_data": "2 5\n",
                "expected_stdout": "7\n",
                "expected_returncode": 0,
            }
        ]

        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        build_root = temp_root / f"no_fuzz_run_{uuid.uuid4().hex}"
        ensure_dir(build_root)
        try:
            pipeline = VerificationPipeline(
                semantic_threshold=0.97,
                build_root=build_root,
            )
            summary = pipeline.verify_source(
                source_path=source_path,
                reference_source_path=source_path,
                test_cases=test_cases,
                sample_id="sum_stdin_no_fuzz_test",
                fuzz_enabled=False,
                fuzz_max_cases=0,
            )
        finally:
            shutil.rmtree(build_root, ignore_errors=True)

        self.assertTrue(summary.diff.executed)
        self.assertFalse(summary.fuzz.executed)
        self.assertTrue(summary.fuzz.passed)
        self.assertNotIn("fuzz", summary.available_signals)

    @unittest.skipIf(choose_compiler() is None, "No supported compiler is available for cache verification tests.")
    def test_verification_pipeline_reuses_cache(self) -> None:
        source_path = PROJECT_ROOT / "data" / "raw" / "smoke" / "sum_stdin.c"
        test_cases = [
            {
                "name": "adds_pair",
                "input_data": "2 5\n",
                "expected_stdout": "7\n",
                "expected_returncode": 0,
            }
        ]

        temp_root = ensure_dir(PROJECT_ROOT / "artifacts" / "test_tmp")
        build_root = temp_root / f"cache_run_{uuid.uuid4().hex}"
        cache_root = build_root / "shared_cache"
        ensure_dir(build_root)
        try:
            pipeline = VerificationPipeline(
                semantic_threshold=0.97,
                build_root=build_root,
                cache_root=cache_root,
            )
            first_summary = pipeline.verify_source(
                source_path=source_path,
                reference_source_path=source_path,
                test_cases=test_cases,
                sample_id="sum_stdin_cache_test",
            )
            second_summary = pipeline.verify_source(
                source_path=source_path,
                reference_source_path=source_path,
                test_cases=test_cases,
                sample_id="sum_stdin_cache_test",
            )
        finally:
            shutil.rmtree(build_root, ignore_errors=True)

        self.assertTrue(first_summary.compile.succeeded)
        self.assertTrue(second_summary.compile.succeeded)
        self.assertIn("Loaded verification result from cache.", second_summary.notes)


if __name__ == "__main__":
    unittest.main()
