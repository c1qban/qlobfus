from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np

from eval.llvm_metrics import build_ir_metrics_snapshot, normalize_block_id
from obfuscator.operators import ActionSpec, build_default_action_library, build_operator_registry
from obfuscator.operators.c_source import analyze_source
from obfuscator.trace import append_trace_event
from scripts.common import ensure_dir, load_structured_file, project_relative, resolve_path, write_text
from verifier.compile import choose_compiler
from verifier.pipeline import VerificationPipeline

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

    class _FallbackEnv:
        metadata: dict[str, object] = {}

        def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> None:
            return None

    class _FallbackDiscrete:
        def __init__(self, n: int) -> None:
            self.n = n

    class _FallbackBox:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _FallbackSpaces:
        Discrete = _FallbackDiscrete
        Box = _FallbackBox

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()


DEFAULT_ACTION_LIBRARY = build_default_action_library()
DEFAULT_ACTION_VOCAB = [action.action_id for action in DEFAULT_ACTION_LIBRARY] + ["stop"]
DEFAULT_STUB_SOURCE = """#include <stdio.h>

int main(void) {
    int a = 0;
    int b = 0;

    if (scanf("%d %d", &a, &b) != 2) {
        return 2;
    }

    printf("%d\\n", a + b);
    return 0;
}
"""
OBSERVATION_DIM = 12


@dataclass
class EnvSample:
    sample_id: str
    split: str
    split_group: str
    source_path: str
    relative_source_path: str
    line_count: int
    compile: dict[str, Any] = field(default_factory=dict)
    verification_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvState:
    sample_id: str
    step_index: int
    remaining_budget: int
    legal_actions: list[str]
    split: str = ""
    source_path: str = ""
    relative_source_path: str = ""
    current_source_path: str = ""
    line_count: int = 0
    test_case_count: int = 0
    action_history: list[str] = field(default_factory=list)
    last_reward: float = 0.0
    semantic_score: float = 1.0
    verifier_coverage: float = 0.0
    potency_score: float = 0.0
    cost_penalty: float = 0.0
    compile_succeeded: bool = True
    latest_operator: str = "reset"
    latest_operator_family: str = "reset"
    llvm_block_coverage: float = 0.0
    branch_rewrite_ratio: float = 0.0
    ir_structural_delta: float = 0.0
    llvm_metric_available: bool = False


def load_env_cache(path_value: str | Path) -> dict[str, Any]:
    payload = load_structured_file(path_value)
    if not isinstance(payload, dict):
        raise ValueError(f"Environment cache must decode to an object: {path_value}")
    if not isinstance(payload.get("samples", []), list):
        raise ValueError(f"Environment cache samples must be a list: {path_value}")
    return payload


def build_env_sample(raw_sample: dict[str, Any]) -> EnvSample:
    source_stats = dict(raw_sample.get("source_stats", {}))
    verification_input = dict(raw_sample.get("verification_input", {}))
    return EnvSample(
        sample_id=str(raw_sample.get("sample_id", "")),
        split=str(raw_sample.get("split", "")),
        split_group=str(raw_sample.get("split_group", "")),
        source_path=str(raw_sample.get("source_path", "")),
        relative_source_path=str(raw_sample.get("relative_source_path", "")),
        line_count=int(source_stats.get("line_count", 0)),
        compile=dict(raw_sample.get("compile", {})),
        verification_input=verification_input,
    )


def split_to_scalar(split: str) -> float:
    return {
        "train": 0.0,
        "validation": 0.5,
        "test": 1.0,
    }.get(split, 0.25)


class ObfuscationEnv:
    """Source-transform environment with real operators, verification, and reward shaping."""

    def __init__(
        self,
        max_steps: int = 8,
        semantic_threshold: float = 0.97,
        legal_actions: list[str] | None = None,
        env_cache_path: str | Path | None = None,
        env_cache: dict[str, Any] | None = None,
        initial_cursor: int = 0,
        reward_config: dict[str, Any] | None = None,
        action_safety_config: dict[str, Any] | None = None,
        verification_config: dict[str, Any] | None = None,
        early_stop_config: dict[str, Any] | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.semantic_threshold = semantic_threshold
        self.reward_config = {
            "semantic_gate": float((reward_config or {}).get("semantic_gate", semantic_threshold)),
            "security_weight": float((reward_config or {}).get("security_weight", 1.0)),
            "potency_weight": float((reward_config or {}).get("potency_weight", 0.4)),
            "cost_weight": float((reward_config or {}).get("cost_weight", 0.3)),
            "diversity_weight": float((reward_config or {}).get("diversity_weight", 0.1)),
            "llvm_block_weight": float((reward_config or {}).get("llvm_block_weight", 0.15)),
            "branch_rewrite_weight": float((reward_config or {}).get("branch_rewrite_weight", 0.1)),
            "ir_delta_weight": float((reward_config or {}).get("ir_delta_weight", 0.15)),
        }
        raw_verification = verification_config or {}
        self.verification_config = {
            "fuzz_enabled": bool(raw_verification.get("fuzz_enabled", True)),
            "fuzz_max_cases": int(raw_verification.get("fuzz_max_cases", 16)),
        }
        raw_early_stop = early_stop_config or {}
        self.early_stop_config = {
            "enabled": bool(raw_early_stop.get("enabled", False)),
            "min_successful_actions": max(0, int(raw_early_stop.get("min_successful_actions", 1))),
            "allow_if_no_transform_actions": bool(raw_early_stop.get("allow_if_no_transform_actions", True)),
        }
        raw_safety = action_safety_config or {}
        self.action_safety_config = {
            "enabled": bool(raw_safety.get("enabled", True)),
            "filter_risky_actions": bool(raw_safety.get("filter_risky_actions", True)),
            "max_allowed_risk": float(raw_safety.get("max_allowed_risk", 0.7)),
            "penalize_bulk_without_harness": bool(raw_safety.get("penalize_bulk_without_harness", True)),
            "penalize_unmapped_llvm": bool(raw_safety.get("penalize_unmapped_llvm", False)),
            "blocked_risk_levels": [str(item) for item in raw_safety.get("blocked_risk_levels", ["high"])],
            "blocked_action_patterns": [str(item) for item in raw_safety.get("blocked_action_patterns", [])],
            "blocked_parameter_tags": [str(item) for item in raw_safety.get("blocked_parameter_tags", [])],
        }
        self.samples: list[EnvSample] = []
        self.job_name = "unbound"
        self.split = "unspecified"
        self._state: EnvState | None = None
        self._current_sample: EnvSample | None = None
        self._cursor = initial_cursor
        self._initial_cursor = initial_cursor
        self._artifact_root = resolve_path(artifact_root or "artifacts/rl_env_episodes")
        self._verifier_cache_root = self._artifact_root.parent / "_verifier_cache"
        self._episode_root: Path | None = None
        self._trace_path: Path | None = None
        self._current_source_text = ""
        self._current_source_artifact = ""
        self._original_source_path: Path | None = None
        self._original_source_text = ""
        self._base_line_count = 0
        self._latest_verification: dict[str, Any] = {}
        self._latest_reward_breakdown: dict[str, Any] = {}
        self._operator_registry = build_operator_registry()
        self._selected_actions = self._resolve_action_library(legal_actions)
        self.action_vocab = [action.action_id for action in self._selected_actions] + ["stop"]
        self._base_action_specs = {action.action_id: action for action in self._selected_actions}
        self._operator_action_ids: dict[str, list[str]] = {}
        for action in self._selected_actions:
            self._operator_action_ids.setdefault(action.operator_name, []).append(action.action_id)
        self._current_action_bindings: dict[str, ActionSpec] = {}
        self._filtered_action_risks: dict[str, dict[str, Any]] = {}
        self._baseline_llvm_block_ids: set[str] = set()
        self._baseline_branch_block_ids: set[str] = set()
        self._touched_llvm_block_ids: set[str] = set()
        self._rewritten_branch_block_ids: set[str] = set()
        self._latest_ir_metrics: dict[str, Any] = {}

        if env_cache is not None:
            self.bind_env_cache(env_cache)
        elif env_cache_path is not None:
            self.bind_env_cache(load_env_cache(env_cache_path))

    @classmethod
    def from_env_cache(
        cls,
        env_cache_path: str | Path,
        *,
        max_steps: int = 8,
        semantic_threshold: float = 0.97,
        legal_actions: list[str] | None = None,
        initial_cursor: int = 0,
        reward_config: dict[str, Any] | None = None,
        action_safety_config: dict[str, Any] | None = None,
        verification_config: dict[str, Any] | None = None,
        early_stop_config: dict[str, Any] | None = None,
        artifact_root: str | Path | None = None,
    ) -> "ObfuscationEnv":
        return cls(
            max_steps=max_steps,
            semantic_threshold=semantic_threshold,
            legal_actions=legal_actions,
            env_cache_path=env_cache_path,
            initial_cursor=initial_cursor,
            reward_config=reward_config,
            action_safety_config=action_safety_config,
            verification_config=verification_config,
            early_stop_config=early_stop_config,
            artifact_root=artifact_root,
        )

    def bind_env_cache(self, payload: dict[str, Any]) -> None:
        self.job_name = str(payload.get("job_name", "unknown"))
        self.split = str(payload.get("split", "unspecified"))
        self.samples = [build_env_sample(dict(sample)) for sample in payload.get("samples", [])]
        self._cursor = self._initial_cursor
        self._state = None
        self._current_sample = None

    def available_sample_ids(self) -> list[str]:
        return [sample.sample_id for sample in self.samples]

    def current_state(self) -> EnvState | None:
        return self._state

    def current_action_mask(self) -> list[int]:
        if self._state is None:
            return [1 if action == "stop" else 0 for action in self.action_vocab]
        legal = set(self._state.legal_actions)
        return [1 if action in legal else 0 for action in self.action_vocab]

    def _execution_key_for_action(self, action_spec: ActionSpec) -> str:
        node_id = str(action_spec.parameters.get("node_id", ""))
        if not node_id:
            return action_spec.action_id
        return f"{action_spec.action_id}@{node_id}"

    def _annotate_action_safety(self, action_spec: ActionSpec) -> ActionSpec:
        if not self.action_safety_config["enabled"]:
            return action_spec

        parameters = dict(action_spec.parameters)
        risk_score = 0.0
        reasons: list[str] = []
        operator_name = action_spec.operator_name
        candidate_count = int(parameters.get("candidate_count", 1) or 1)
        test_case_count = int(self._current_sample.verification_input.get("case_count", 0)) if self._current_sample is not None else 0

        if str(parameters.get("function_name", "")) == "global":
            risk_score += 0.7
            reasons.append("global_scope_rewrite")
        if self.action_safety_config["penalize_unmapped_llvm"] and not parameters.get("llvm_block_id") and not parameters.get("llvm_block_ids"):
            risk_score += 0.15
            reasons.append("missing_llvm_block_binding")
        if action_spec.location_tag.endswith("_bulk"):
            risk_score += 0.15
            reasons.append("bulk_rewrite")
            if candidate_count > 8:
                risk_score += 0.2
                reasons.append("large_bulk_rewrite")
            if self.action_safety_config["penalize_bulk_without_harness"] and test_case_count <= 0:
                risk_score += 0.2
                reasons.append("bulk_without_harness_cases")
        if operator_name == "rename_locals":
            if not bool(parameters.get("is_initialized", True)) and not action_spec.location_tag.endswith("_bulk"):
                risk_score += 0.05
                reasons.append("uninitialized_declaration_rename")
            if int(parameters.get("declarator_count", 1) or 1) > 1 and not action_spec.location_tag.endswith("_bulk"):
                risk_score += 0.05
                reasons.append("multi_declarator_rename")
        elif operator_name == "encode_literals":
            if str(parameters.get("literal_kind", "")) == "float":
                risk_score += 0.2
                reasons.append("floating_literal_rewrite")
            if action_spec.location_tag.endswith("_bulk") and candidate_count > 6:
                risk_score += 0.15
                reasons.append("many_literal_rewrites")
        elif operator_name == "flatten_cfg":
            risk_score += 0.1
            reasons.append("control_flow_rewrite")
        elif operator_name == "substitute_instructions":
            if action_spec.parameter_tag == "neg_add":
                risk_score += 0.7
                reasons.append("arithmetic_identity_rewrite")

        risk_score = min(risk_score, 1.0)
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.35:
            risk_level = "medium"
        else:
            risk_level = "low"

        parameters.update(
            {
                "risk_score": round(risk_score, 6),
                "risk_level": risk_level,
                "risk_reasons": reasons,
            }
        )
        return ActionSpec(
            action_id=action_spec.action_id,
            operator_name=action_spec.operator_name,
            location_tag=action_spec.location_tag,
            parameter_tag=action_spec.parameter_tag,
            parameters=parameters,
        )

    def _is_action_allowed_by_safety(self, action_spec: ActionSpec) -> bool:
        if not self.action_safety_config["enabled"] or not self.action_safety_config["filter_risky_actions"]:
            return True
        if str(action_spec.parameters.get("risk_level", "")) in set(self.action_safety_config["blocked_risk_levels"]):
            return False
        if action_spec.parameter_tag in set(self.action_safety_config["blocked_parameter_tags"]):
            return False
        for pattern in self.action_safety_config["blocked_action_patterns"]:
            if fnmatch(action_spec.action_id, pattern):
                return False
        return float(action_spec.parameters.get("risk_score", 0.0)) < float(self.action_safety_config["max_allowed_risk"])

    def _resolve_action_library(self, configured_actions: list[str] | None) -> list[ActionSpec]:
        if not configured_actions:
            return list(DEFAULT_ACTION_LIBRARY)

        selected_ids: list[str] = []
        known_action_ids = {action.action_id for action in DEFAULT_ACTION_LIBRARY}
        known_operator_names = {action.operator_name for action in DEFAULT_ACTION_LIBRARY}

        for configured_action in configured_actions:
            if configured_action == "stop":
                continue
            if configured_action in known_action_ids:
                selected_ids.append(configured_action)
                continue
            if configured_action in known_operator_names:
                selected_ids.extend(
                    action.action_id
                    for action in DEFAULT_ACTION_LIBRARY
                    if action.operator_name == configured_action
                )
                continue
            raise ValueError(f"Unknown action or operator in policy action_vocab: {configured_action}")

        seen: set[str] = set()
        selected_actions: list[ActionSpec] = []
        for action in DEFAULT_ACTION_LIBRARY:
            if action.action_id in selected_ids and action.action_id not in seen:
                selected_actions.append(action)
                seen.add(action.action_id)
        return selected_actions

    def _build_stub_sample(self, sample_id: str) -> EnvSample:
        return EnvSample(
            sample_id=sample_id,
            split=self.split,
            split_group=sample_id,
            source_path="",
            relative_source_path="",
            line_count=len(DEFAULT_STUB_SOURCE.splitlines()),
            compile={"compiler_flags": ["-O0"]},
            verification_input={"case_count": 0, "test_cases": []},
        )

    def _select_sample(self, sample_id: str | None = None) -> EnvSample:
        if sample_id is not None:
            if not self.samples:
                return self._build_stub_sample(sample_id)
            for sample in self.samples:
                if sample.sample_id == sample_id:
                    return sample
            raise KeyError(f"Sample not found in bound environment cache: {sample_id}")

        if self.samples:
            sample = self.samples[self._cursor % len(self.samples)]
            self._cursor += 1
            return sample

        return self._build_stub_sample("stub_sample")

    def _create_episode_workspace(self, sample: EnvSample) -> None:
        stamp = datetime.now().strftime("%H%M%S%f")
        sample_digest = hashlib.sha1(sample.sample_id.encode("utf-8")).hexdigest()[:10]
        self._episode_root = ensure_dir(self._artifact_root / f"e_{sample_digest}_{stamp}")
        self._trace_path = self._episode_root / "transform_trace.jsonl"

    def _load_source_text(self, sample: EnvSample) -> str:
        if sample.source_path:
            source_path = resolve_path(sample.source_path)
            if source_path.exists():
                return source_path.read_text(encoding="utf-8")
        return DEFAULT_STUB_SOURCE

    def _persist_source_artifact(self, source_text: str, step_index: int, label: str) -> Path:
        if self._episode_root is None:
            raise RuntimeError("Episode workspace was not initialized.")
        label_digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
        episode_root = ensure_dir(self._episode_root)
        artifact_path = episode_root / f"s{step_index:02d}_{label_digest}.c"
        ensure_dir(artifact_path.parent)
        write_text(artifact_path, source_text)
        return artifact_path

    def _verify_source_artifact(self, sample: EnvSample, source_path: Path, step_index: int) -> dict[str, Any]:
        if choose_compiler(sample.compile.get("compiler")) is None:
            return VerificationPipeline(semantic_threshold=self.semantic_threshold).run_stub(sample_id=f"{sample.sample_id}_step_{step_index}").to_dict()

        pipeline = VerificationPipeline(
            semantic_threshold=self.semantic_threshold,
            compiler=sample.compile.get("compiler"),
            compiler_flags=list(sample.compile.get("compiler_flags", [])),
            compile_timeout_sec=float(sample.compile.get("timeout_sec", 10.0)),
            build_root=(self._episode_root / "verifier") if self._episode_root is not None else None,
            cache_root=self._verifier_cache_root,
        )
        summary = pipeline.verify_source(
            source_path=source_path,
            test_cases=list(sample.verification_input.get("test_cases", [])),
            sample_id=f"{sample.sample_id}_step_{step_index}",
            compiler=sample.compile.get("compiler"),
            compiler_flags=list(sample.compile.get("compiler_flags", [])),
            compile_timeout_sec=float(sample.compile.get("timeout_sec", 10.0)),
            reference_source_path=self._original_source_path if step_index > 0 else None,
            build_dir=(self._episode_root / "verifier" / f"step_{step_index:02d}") if self._episode_root is not None else None,
            workdir=self._episode_root,
            fuzz_max_cases=int(sample.verification_input.get("fuzz", {}).get("max_cases", self.verification_config["fuzz_max_cases"])),
            fuzz_enabled=bool(self.verification_config["fuzz_enabled"]),
        )
        return summary.to_dict()

    def _count_lines(self, source_text: str) -> int:
        return len(source_text.splitlines())

    def _collect_llvm_block_ids(self, bindings: dict[str, ActionSpec]) -> set[str]:
        block_ids: set[str] = set()
        for action_spec in bindings.values():
            llvm_block_id = normalize_block_id(action_spec.parameters.get("llvm_block_id", ""))
            if llvm_block_id:
                block_ids.add(llvm_block_id)
            for raw_block_id in action_spec.parameters.get("llvm_block_ids", []) or []:
                block_id = normalize_block_id(raw_block_id)
                if block_id:
                    block_ids.add(block_id)
        return block_ids

    def _collect_branch_block_ids(self, bindings: dict[str, ActionSpec]) -> set[str]:
        branch_blocks: set[str] = set()
        for action_spec in bindings.values():
            if action_spec.operator_name != "flatten_cfg":
                continue
            llvm_block_id = normalize_block_id(action_spec.parameters.get("llvm_block_id", ""))
            if llvm_block_id:
                branch_blocks.add(llvm_block_id)
            for raw_block_id in action_spec.parameters.get("llvm_block_ids", []) or []:
                block_id = normalize_block_id(raw_block_id)
                if block_id:
                    branch_blocks.add(block_id)
        return branch_blocks

    def _collect_source_llvm_block_ids(self, source_text: str) -> set[str]:
        try:
            analysis = analyze_source(source_text)
        except Exception:
            return set()
        return {
            normalize_block_id(block.block_id)
            for block in analysis.clang_summary.llvm_ir_blocks
            if normalize_block_id(block.block_id)
        }

    def _collect_source_branch_block_ids(self, source_text: str) -> set[str]:
        try:
            analysis = analyze_source(source_text)
        except Exception:
            return set()
        branch_blocks: set[str] = set()
        for branch in analysis.clang_summary.branches:
            llvm_block_id = normalize_block_id(branch.metadata.get("llvm_block_id", ""))
            if llvm_block_id:
                branch_blocks.add(llvm_block_id)
        return branch_blocks

    def _compute_ir_metrics(self, bindings: dict[str, ActionSpec], source_text: str = "") -> dict[str, Any]:
        current_llvm_blocks = self._collect_llvm_block_ids(bindings)
        if source_text:
            current_llvm_blocks.update(self._collect_source_llvm_block_ids(source_text))
        return build_ir_metrics_snapshot(
            baseline_llvm_blocks=self._baseline_llvm_block_ids,
            current_llvm_blocks=current_llvm_blocks,
            touched_llvm_blocks=self._touched_llvm_block_ids,
            baseline_branch_blocks=self._baseline_branch_block_ids,
            rewritten_branch_blocks=self._rewritten_branch_block_ids,
        )

    def _bind_actions_for_source(self, source_text: str, action_history: list[str]) -> tuple[dict[str, ActionSpec], list[str]]:
        used = set(action_history)
        bindings: dict[str, ActionSpec] = {}
        filtered_risks: dict[str, dict[str, Any]] = {}
        for operator_name, operator in self._operator_registry.items():
            if operator_name not in self._operator_action_ids:
                continue
            for action_spec in operator.enumerate_actions(source_text):
                if action_spec.action_id not in self._base_action_specs:
                    continue
                action_spec = self._annotate_action_safety(action_spec)
                execution_key = self._execution_key_for_action(action_spec)
                if execution_key in used:
                    continue
                if not self._is_action_allowed_by_safety(action_spec):
                    filtered_risks[action_spec.action_id] = {
                        "risk_score": float(action_spec.parameters.get("risk_score", 0.0)),
                        "risk_level": str(action_spec.parameters.get("risk_level", "")),
                        "risk_reasons": list(action_spec.parameters.get("risk_reasons", [])),
                    }
                    continue
                bindings[action_spec.action_id] = action_spec

        legal_actions = [action_id for action_id in self.action_vocab if action_id != "stop" and action_id in bindings]
        stop_allowed = (
            not self.early_stop_config["enabled"]
            or len(action_history) >= int(self.early_stop_config["min_successful_actions"])
            or (not legal_actions and self.early_stop_config["allow_if_no_transform_actions"])
        )
        if "stop" in self.action_vocab and stop_allowed:
            legal_actions.append("stop")
        self._filtered_action_risks = filtered_risks
        return bindings, legal_actions

    def _resolve_requested_action(self, requested_action: str, legal_actions: list[str]) -> str | None:
        if requested_action in legal_actions:
            return requested_action
        for action_id in legal_actions:
            action_spec = self._current_action_bindings.get(action_id)
            if action_spec is not None and action_spec.operator_name == requested_action:
                return action_id
        return None

    def _compute_potency_score(self, source_text: str, action_history: list[str]) -> float:
        if not self._original_source_text:
            return 0.0
        edit_delta = 1.0 - SequenceMatcher(None, self._original_source_text, source_text).ratio()
        current_lines = self._count_lines(source_text)
        line_growth = max(current_lines - self._base_line_count, 0) / max(float(self._base_line_count), 1.0)
        diversity_ratio = len(set(action_history)) / max(float(max(len(self.action_vocab) - 1, 1)), 1.0)
        potency = (0.55 * edit_delta) + (0.30 * min(line_growth, 1.0)) + (0.15 * diversity_ratio)
        return float(min(max(potency, 0.0), 1.0))

    def _compute_cost_penalty(self, source_text: str) -> float:
        current_lines = self._count_lines(source_text)
        line_growth = max(current_lines - self._base_line_count, 0) / max(float(self._base_line_count), 1.0)
        return float(min(max(line_growth, 0.0), 1.0))

    def _compute_reward_breakdown(
        self,
        *,
        verification: dict[str, Any],
        potency_score: float,
        cost_penalty: float,
        action_history: list[str],
        ir_metrics: dict[str, Any],
    ) -> dict[str, float | bool]:
        semantic_score = float(verification.get("semantic_score", 0.0))
        semantic_gate = float(self.reward_config["semantic_gate"])
        compile_succeeded = bool(verification.get("compile", {}).get("succeeded", False))
        diversity_ratio = len(set(action_history)) / max(float(max(len(self.action_vocab) - 1, 1)), 1.0)
        security_weight = float(self.reward_config["security_weight"])
        potency_weight = float(self.reward_config["potency_weight"])
        diversity_weight = float(self.reward_config["diversity_weight"])
        cost_weight = float(self.reward_config["cost_weight"])
        llvm_block_weight = float(self.reward_config["llvm_block_weight"])
        branch_rewrite_weight = float(self.reward_config["branch_rewrite_weight"])
        ir_delta_weight = float(self.reward_config["ir_delta_weight"])
        llvm_block_coverage = float(ir_metrics.get("llvm_block_coverage", 0.0))
        branch_rewrite_ratio = float(ir_metrics.get("branch_rewrite_ratio", 0.0))
        ir_structural_delta = float(ir_metrics.get("ir_structural_delta", 0.0))

        if not compile_succeeded or semantic_score < semantic_gate:
            semantic_component = -1.0 - security_weight * max(semantic_gate - semantic_score, 0.0)
            total_reward = semantic_component - cost_weight * cost_penalty
            return {
                "semantic_component": semantic_component,
                "potency_component": 0.0,
                "diversity_component": 0.0,
                "llvm_block_component": 0.0,
                "branch_rewrite_component": 0.0,
                "ir_delta_component": 0.0,
                "cost_component": cost_weight * cost_penalty,
                "total_reward": total_reward,
                "semantic_gate_passed": False,
                "ir_metrics": dict(ir_metrics),
            }

        semantic_component = security_weight * ((semantic_score - semantic_gate) / max(1.0 - semantic_gate, 1e-6))
        potency_component = potency_weight * potency_score
        diversity_component = diversity_weight * diversity_ratio
        llvm_block_component = llvm_block_weight * llvm_block_coverage
        branch_rewrite_component = branch_rewrite_weight * branch_rewrite_ratio
        ir_delta_component = ir_delta_weight * ir_structural_delta
        cost_component = cost_weight * cost_penalty
        total_reward = (
            semantic_component
            + potency_component
            + diversity_component
            + llvm_block_component
            + branch_rewrite_component
            + ir_delta_component
            - cost_component
        )
        return {
            "semantic_component": semantic_component,
            "potency_component": potency_component,
            "diversity_component": diversity_component,
            "llvm_block_component": llvm_block_component,
            "branch_rewrite_component": branch_rewrite_component,
            "ir_delta_component": ir_delta_component,
            "cost_component": cost_component,
            "total_reward": total_reward,
            "semantic_gate_passed": True,
            "ir_metrics": dict(ir_metrics),
        }

    def _build_state(
        self,
        sample: EnvSample,
        *,
        source_artifact: str,
        source_text: str,
        step_index: int,
        action_history: list[str],
        last_reward: float,
        latest_operator: str,
        verification: dict[str, Any],
        potency_score: float,
        cost_penalty: float,
        ir_metrics: dict[str, Any],
    ) -> EnvState:
        bindings, legal_actions = self._bind_actions_for_source(source_text, action_history)
        self._current_action_bindings = bindings
        if latest_operator == "stop":
            latest_operator_family = "stop"
        elif latest_operator == "reset":
            latest_operator_family = "reset"
        else:
            latest_operator_family = self._current_action_bindings.get(
                latest_operator,
                ActionSpec("reset", "reset", "reset", "reset"),
            ).operator_name
        return EnvState(
            sample_id=sample.sample_id,
            step_index=step_index,
            remaining_budget=max(self.max_steps - step_index, 0),
            legal_actions=legal_actions,
            split=sample.split,
            source_path=sample.source_path,
            relative_source_path=sample.relative_source_path,
            current_source_path=source_artifact,
            line_count=self._count_lines(source_text),
            test_case_count=int(sample.verification_input.get("case_count", 0)),
            action_history=list(action_history),
            last_reward=last_reward,
            semantic_score=float(verification.get("semantic_score", 0.0)),
            verifier_coverage=float(verification.get("verifier_coverage", 0.0)),
            potency_score=potency_score,
            cost_penalty=cost_penalty,
            compile_succeeded=bool(verification.get("compile", {}).get("succeeded", False)),
            latest_operator=latest_operator,
            latest_operator_family=latest_operator_family,
            llvm_block_coverage=float(ir_metrics.get("llvm_block_coverage", 0.0)),
            branch_rewrite_ratio=float(ir_metrics.get("branch_rewrite_ratio", 0.0)),
            ir_structural_delta=float(ir_metrics.get("ir_structural_delta", 0.0)),
            llvm_metric_available=bool(ir_metrics.get("llvm_metric_available", False)),
        )

    def observation_from_state(self, state: EnvState) -> np.ndarray:
        feature_vector = np.asarray(
            [
                state.step_index / max(float(self.max_steps), 1.0),
                state.remaining_budget / max(float(self.max_steps), 1.0),
                min(float(state.line_count), 5000.0) / 5000.0,
                min(float(state.test_case_count), 64.0) / 64.0,
                len(state.action_history) / max(float(max(len(self.action_vocab) - 1, 1)), 1.0),
                len(state.legal_actions) / max(float(len(self.action_vocab)), 1.0),
                split_to_scalar(state.split),
                float(min(max(state.semantic_score, 0.0), 1.0)),
                float(min(max(state.potency_score, 0.0), 1.0)),
                float(min(max(state.cost_penalty, 0.0), 1.0)),
                1.0 if state.compile_succeeded else 0.0,
                float(min(max(state.verifier_coverage, 0.0), 1.0)),
            ],
            dtype=np.float32,
        )
        return feature_vector

    def _append_trace(self, payload: dict[str, Any]) -> None:
        if self._trace_path is not None:
            append_trace_event(self._trace_path, payload)

    def reset(self, sample_id: str | None = None) -> EnvState:
        self._current_sample = self._select_sample(sample_id=sample_id)
        self._create_episode_workspace(self._current_sample)
        self._original_source_text = self._load_source_text(self._current_sample)
        self._current_source_text = self._original_source_text
        self._base_line_count = self._count_lines(self._original_source_text)

        original_path = self._persist_source_artifact(self._original_source_text, 0, "original")
        self._original_source_path = original_path
        verification = self._verify_source_artifact(self._current_sample, original_path, 0)
        self._current_source_artifact = project_relative(original_path)
        self._latest_verification = verification
        self._latest_reward_breakdown = {
            "semantic_component": 0.0,
            "potency_component": 0.0,
            "diversity_component": 0.0,
            "llvm_block_component": 0.0,
            "branch_rewrite_component": 0.0,
            "ir_delta_component": 0.0,
            "cost_component": 0.0,
            "total_reward": 0.0,
            "semantic_gate_passed": bool(float(verification.get("semantic_score", 0.0)) >= float(self.reward_config["semantic_gate"])),
            "ir_metrics": {},
        }
        initial_bindings, _initial_legal_actions = self._bind_actions_for_source(self._current_source_text, [])
        self._baseline_llvm_block_ids = self._collect_llvm_block_ids(initial_bindings)
        self._baseline_llvm_block_ids.update(self._collect_source_llvm_block_ids(self._current_source_text))
        self._baseline_branch_block_ids = self._collect_branch_block_ids(initial_bindings)
        self._baseline_branch_block_ids.update(self._collect_source_branch_block_ids(self._current_source_text))
        self._touched_llvm_block_ids = set()
        self._rewritten_branch_block_ids = set()
        self._latest_ir_metrics = self._compute_ir_metrics(initial_bindings, self._current_source_text)
        self._latest_reward_breakdown["ir_metrics"] = dict(self._latest_ir_metrics)
        self._state = self._build_state(
            self._current_sample,
            source_artifact=project_relative(original_path),
            source_text=self._current_source_text,
            step_index=0,
            action_history=[],
            last_reward=0.0,
            latest_operator="reset",
            verification=verification,
            potency_score=0.0,
            cost_penalty=0.0,
            ir_metrics=self._latest_ir_metrics,
        )
        self._append_trace(
            {
                "event": "reset",
                "sample_id": self._current_sample.sample_id,
                "source_artifact": project_relative(original_path),
                "verification": verification,
                "ir_metrics": self._latest_ir_metrics,
                "action_safety": dict(self.action_safety_config),
                "verification_config": dict(self.verification_config),
                "filtered_action_risks": dict(self._filtered_action_risks),
            }
        )
        return self._state

    def step(self, action: str) -> tuple[EnvState, float, bool, dict[str, object]]:
        if self._state is None or self._current_sample is None:
            raise RuntimeError("Call reset() before step().")

        next_step = self._state.step_index + 1
        next_history = list(self._state.action_history)
        resolved_action = self._resolve_requested_action(action, self._state.legal_actions)
        action_valid = resolved_action is not None
        termination_reason = "continue"
        operator_metadata: dict[str, object] = {}
        verification = dict(self._latest_verification)
        reward_breakdown = dict(self._latest_reward_breakdown)
        source_text = self._current_source_text
        source_artifact = self._current_source_artifact
        action_spec: ActionSpec | None = None
        ir_metrics = dict(self._latest_ir_metrics)

        if action == "stop" and action_valid:
            reward = 0.05 * (
                float(self._state.potency_score)
                + float(self._state.llvm_block_coverage)
                + float(self._state.branch_rewrite_ratio)
                + float(self._state.ir_structural_delta)
            )
            reward_breakdown = {
                "semantic_component": 0.0,
                "potency_component": reward,
                "diversity_component": 0.0,
                "llvm_block_component": 0.0,
                "branch_rewrite_component": 0.0,
                "ir_delta_component": 0.0,
                "cost_component": 0.0,
                "total_reward": reward,
                "semantic_gate_passed": True,
                "ir_metrics": dict(ir_metrics),
            }
            termination_reason = "stop"
        elif not action_valid:
            reward = -1.0
            reward_breakdown = {
                "semantic_component": -1.0,
                "potency_component": 0.0,
                "diversity_component": 0.0,
                "llvm_block_component": 0.0,
                "branch_rewrite_component": 0.0,
                "ir_delta_component": 0.0,
                "cost_component": 0.0,
                "total_reward": reward,
                "semantic_gate_passed": False,
                "ir_metrics": dict(ir_metrics),
            }
            termination_reason = "invalid_action"
        else:
            action_spec = self._current_action_bindings.get(resolved_action or "")
            operator = self._operator_registry.get(action_spec.operator_name if action_spec is not None else "")
            if operator is None:
                reward = -1.0
                reward_breakdown = {
                    "semantic_component": -1.0,
                    "potency_component": 0.0,
                    "diversity_component": 0.0,
                    "llvm_block_component": 0.0,
                    "branch_rewrite_component": 0.0,
                    "ir_delta_component": 0.0,
                    "cost_component": 0.0,
                    "total_reward": reward,
                    "semantic_gate_passed": False,
                    "ir_metrics": dict(ir_metrics),
                }
                termination_reason = "missing_operator"
            else:
                application = operator.apply(self._current_source_text, action_spec)
                operator_metadata = dict(application.metadata)
                operator_metadata["notes"] = list(application.notes)
                operator_metadata["requested_action"] = action
                operator_metadata["resolved_action"] = resolved_action
                operator_metadata["execution_key"] = self._execution_key_for_action(action_spec)
                operator_metadata["operator_name"] = action_spec.operator_name
                operator_metadata["location_tag"] = action_spec.location_tag
                operator_metadata["parameter_tag"] = action_spec.parameter_tag
                operator_metadata["parameters"] = dict(action_spec.parameters)
                if not application.applied:
                    reward = -1.0
                    reward_breakdown = {
                        "semantic_component": -1.0,
                        "potency_component": 0.0,
                        "diversity_component": 0.0,
                        "llvm_block_component": 0.0,
                        "branch_rewrite_component": 0.0,
                        "ir_delta_component": 0.0,
                        "cost_component": 0.0,
                        "total_reward": reward,
                        "semantic_gate_passed": False,
                        "ir_metrics": dict(ir_metrics),
                    }
                    termination_reason = "operator_not_applicable"
                else:
                    source_text = application.source_text
                    source_path = self._persist_source_artifact(source_text, next_step, resolved_action or action)
                    source_artifact = project_relative(source_path)
                    verification = self._verify_source_artifact(self._current_sample, source_path, next_step)
                    next_history.append(self._execution_key_for_action(action_spec))
                    touched_block_ids: set[str] = set()
                    llvm_block_id = normalize_block_id(action_spec.parameters.get("llvm_block_id", ""))
                    if llvm_block_id:
                        touched_block_ids.add(llvm_block_id)
                    for raw_block_id in action_spec.parameters.get("llvm_block_ids", []) or []:
                        block_id = normalize_block_id(raw_block_id)
                        if block_id:
                            touched_block_ids.add(block_id)
                    self._touched_llvm_block_ids.update(touched_block_ids)
                    if action_spec.operator_name == "flatten_cfg":
                        self._rewritten_branch_block_ids.update(touched_block_ids)
                    next_bindings, _next_legal_actions = self._bind_actions_for_source(source_text, next_history)
                    ir_metrics = self._compute_ir_metrics(next_bindings, source_text)
                    self._latest_ir_metrics = ir_metrics
                    potency_score = self._compute_potency_score(source_text, next_history)
                    cost_penalty = self._compute_cost_penalty(source_text)
                    reward_breakdown = self._compute_reward_breakdown(
                        verification=verification,
                        potency_score=potency_score,
                        cost_penalty=cost_penalty,
                        action_history=next_history,
                        ir_metrics=ir_metrics,
                    )
                    reward = float(reward_breakdown["total_reward"])
                    self._current_source_text = source_text
                    self._current_source_artifact = source_artifact
                    self._latest_verification = verification
                    self._latest_reward_breakdown = reward_breakdown
                    if not bool(reward_breakdown["semantic_gate_passed"]):
                        termination_reason = "semantic_gate_failed"

        if action in {"stop"}:
            potency_score = self._state.potency_score
            cost_penalty = self._state.cost_penalty
        elif action_valid and action_spec is not None and self._execution_key_for_action(action_spec) in next_history:
            potency_score = self._compute_potency_score(source_text, next_history)
            cost_penalty = self._compute_cost_penalty(source_text)
        else:
            potency_score = self._state.potency_score
            cost_penalty = self._state.cost_penalty
            ir_metrics = dict(self._latest_ir_metrics)

        if termination_reason == "continue" and next_step >= self.max_steps:
            termination_reason = "max_steps"
        if termination_reason == "continue":
            _bindings, next_legal_actions = self._bind_actions_for_source(source_text, next_history)
            if len(next_legal_actions) == 1 and next_legal_actions[0] == "stop":
                termination_reason = "no_actions_remaining"

        done = termination_reason != "continue"
        self._state = self._build_state(
            self._current_sample,
            source_artifact=source_artifact,
            source_text=source_text,
            step_index=next_step,
            action_history=next_history,
            last_reward=float(reward_breakdown["total_reward"]),
            latest_operator=resolved_action or action,
            verification=verification,
            potency_score=potency_score,
            cost_penalty=cost_penalty,
            ir_metrics=ir_metrics,
        )
        self._append_trace(
            {
                "event": "step",
                "action": resolved_action or action,
                "requested_action": action,
                "action_valid": action_valid,
                "operator_metadata": operator_metadata,
                "potency_score": potency_score,
                "cost_penalty": cost_penalty,
                "semantic_score": float(self._state.semantic_score),
                "verifier_coverage": float(self._state.verifier_coverage),
                "reward_breakdown": reward_breakdown,
                "source_artifact": source_artifact,
                "termination_reason": termination_reason,
                "verification": verification,
            }
        )
        info = {
            "action_valid": action_valid,
            "action_mask": self.current_action_mask(),
            "action_safety": dict(self.action_safety_config),
            "early_stop": dict(self.early_stop_config),
            "verification_config": dict(self.verification_config),
            "available_sample_ids": self.available_sample_ids(),
            "bound_job_name": self.job_name,
            "compile_succeeded": self._state.compile_succeeded,
            "current_source_artifact": source_artifact,
            "episode_done": done,
            "operator_metadata": operator_metadata,
            "reward_breakdown": reward_breakdown,
            "requested_action": action,
            "resolved_action": resolved_action or action,
            "sample_id": self._current_sample.sample_id,
            "semantic_score": self._state.semantic_score,
            "semantic_threshold": self.semantic_threshold,
            "split": self._current_sample.split,
            "termination_reason": termination_reason,
            "trace_path": project_relative(self._trace_path) if self._trace_path is not None else "",
            "llvm_block_coverage": self._state.llvm_block_coverage,
            "branch_rewrite_ratio": self._state.branch_rewrite_ratio,
            "ir_structural_delta": self._state.ir_structural_delta,
            "llvm_metric_available": self._state.llvm_metric_available,
            "verification_case_count": int(self._current_sample.verification_input.get("case_count", 0)),
            "verification_summary": verification,
            "filtered_action_risks": dict(self._filtered_action_risks),
        }
        return self._state, float(reward_breakdown["total_reward"]), done, info


class GymObfuscationEnv(gym.Env):
    """Gymnasium-compatible wrapper around the env-cache-driven ObfuscationEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_cache_path: str | Path | None = None,
        env_cache: dict[str, Any] | None = None,
        max_steps: int = 8,
        semantic_threshold: float = 0.97,
        legal_actions: list[str] | None = None,
        initial_cursor: int = 0,
        reward_config: dict[str, Any] | None = None,
        action_safety_config: dict[str, Any] | None = None,
        verification_config: dict[str, Any] | None = None,
        early_stop_config: dict[str, Any] | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.core_env = ObfuscationEnv(
            max_steps=max_steps,
            semantic_threshold=semantic_threshold,
            legal_actions=legal_actions,
            env_cache_path=env_cache_path,
            env_cache=env_cache,
            initial_cursor=initial_cursor,
            reward_config=reward_config,
            action_safety_config=action_safety_config,
            verification_config=verification_config,
            early_stop_config=early_stop_config,
            artifact_root=artifact_root,
        )
        self.action_space = spaces.Discrete(len(self.core_env.action_vocab))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)

    @classmethod
    def from_env_cache(
        cls,
        env_cache_path: str | Path,
        *,
        max_steps: int = 8,
        semantic_threshold: float = 0.97,
        legal_actions: list[str] | None = None,
        initial_cursor: int = 0,
        reward_config: dict[str, Any] | None = None,
        action_safety_config: dict[str, Any] | None = None,
        verification_config: dict[str, Any] | None = None,
        early_stop_config: dict[str, Any] | None = None,
        artifact_root: str | Path | None = None,
    ) -> "GymObfuscationEnv":
        return cls(
            env_cache_path=env_cache_path,
            max_steps=max_steps,
            semantic_threshold=semantic_threshold,
            legal_actions=legal_actions,
            initial_cursor=initial_cursor,
            reward_config=reward_config,
            action_safety_config=action_safety_config,
            verification_config=verification_config,
            early_stop_config=early_stop_config,
            artifact_root=artifact_root,
        )

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.core_env.current_action_mask(), dtype=np.int8)

    def _build_info(self, state: EnvState, extra: dict[str, object] | None = None) -> dict[str, object]:
        info: dict[str, object] = {
            "sample_id": state.sample_id,
            "split": state.split,
            "relative_source_path": state.relative_source_path,
            "current_source_path": state.current_source_path,
            "line_count": state.line_count,
            "test_case_count": state.test_case_count,
            "legal_actions": list(state.legal_actions),
            "action_history": list(state.action_history),
            "action_mask": self.action_masks().tolist(),
            "filtered_action_risks": dict(self.core_env._filtered_action_risks),
            "action_safety": dict(self.core_env.action_safety_config),
            "early_stop": dict(self.core_env.early_stop_config),
            "verification_config": dict(self.core_env.verification_config),
            "bound_job_name": self.core_env.job_name,
            "semantic_score": state.semantic_score,
            "potency_score": state.potency_score,
            "cost_penalty": state.cost_penalty,
            "compile_succeeded": state.compile_succeeded,
            "latest_operator": state.latest_operator,
            "llvm_block_coverage": state.llvm_block_coverage,
            "branch_rewrite_ratio": state.branch_rewrite_ratio,
            "ir_structural_delta": state.ir_structural_delta,
            "llvm_metric_available": state.llvm_metric_available,
        }
        if extra:
            info.update(extra)
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if GYMNASIUM_AVAILABLE:
            super().reset(seed=seed)

        sample_id = str(options["sample_id"]) if options and "sample_id" in options else None
        state = self.core_env.reset(sample_id=sample_id)
        observation = self.core_env.observation_from_state(state)
        info = self._build_info(
            state,
            {
                "seed": seed,
                "trace_path": project_relative(self.core_env._trace_path) if self.core_env._trace_path is not None else "",
                "verification_summary": self.core_env._latest_verification,
                "reward_breakdown": self.core_env._latest_reward_breakdown,
            },
        )
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_index = int(action)
        if 0 <= action_index < len(self.core_env.action_vocab):
            action_name = self.core_env.action_vocab[action_index]
        else:
            action_name = "__invalid_action__"

        state, reward, done, step_info = self.core_env.step(action_name)
        observation = self.core_env.observation_from_state(state)
        terminated = bool(done and step_info.get("termination_reason") != "max_steps")
        truncated = bool(done and step_info.get("termination_reason") == "max_steps")
        info = self._build_info(
            state,
            {
                **step_info,
                "action_name": action_name,
                "action_index": action_index,
            },
        )
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        return None
