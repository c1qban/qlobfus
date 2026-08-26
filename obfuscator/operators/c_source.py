from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from dataclasses import dataclass, field

from .clang_ast import ClangAstNode, ClangAstSummary, LlvmIrBlock, load_clang_ast_summary


CONTROL_FLOW_PATTERN = re.compile(r"^\s*if\s*\((?P<condition>.+)\)\s*\{\s*$")
SINGLE_LINE_RETURN_IF_PATTERN = re.compile(r"^\s*if\s*\((?P<condition>.+)\)\s*return\s+(?P<expr>[^;]+);\s*$")
TYPE_PATTERN = r"(?:(?:unsigned|signed)\s+)?(?:(?:long\s+long)|(?:long\s+double)|int|long|short|char|float|double)"
DECLARATION_PATTERN = re.compile(
    rf"^(?P<indent>\s*)(?P<ctype>{TYPE_PATTERN})\s+(?P<declarators>[^;()]+);\s*$",
    re.MULTILINE,
)
DECLARATION_NAME_PATTERN = re.compile(
    rf"(?P<ctype>{TYPE_PATTERN})\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?:=\s*[^;]+)?;"
)
NUMERIC_LITERAL_PATTERN = re.compile(
    r"(?<![\w.])(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFlLuU]*(?![\w.])"
)
IDENTIFIER_ADDITION_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\+\s*([A-Za-z_]\w*)\b")
IDENTIFIER_PATTERN_TEMPLATE = r"\b{identifier}\b"
FUNCTION_HEADER_PATTERN = re.compile(
    r"^\s*(?!if\b|for\b|while\b|switch\b|else\b)(?:[A-Za-z_]\w*[\s\*]+)+(?P<name>[A-Za-z_]\w*)\s*\([^;]*\)\s*\{\s*$"
)

MAX_FLATTEN_LOCATIONS = 2
MAX_DECLARATION_LOCATIONS = 4
MAX_LOCAL_LOCATIONS = 4
MAX_ADDITION_LOCATIONS = 4
MAX_LITERAL_LOCATIONS = 6
MAX_SOURCE_ANALYSIS_CACHE = 64


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    operator_name: str
    location_tag: str
    parameter_tag: str
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass
class OperatorApplication:
    action_name: str
    applied: bool
    source_text: str
    metadata: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FunctionSpan:
    name: str
    ordinal: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class NodeCandidate:
    node_id: str
    function_name: str
    line_no: int
    column_no: int
    slot_index: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceAnalysis:
    masked_code: str
    line_offsets: list[int]
    functions: list[FunctionSpan]
    clang_summary: ClangAstSummary


_SOURCE_ANALYSIS_CACHE: dict[str, SourceAnalysis] = {}


class SourceOperator:
    name: str

    def enumerate_actions(self, source_text: str) -> list[ActionSpec]:
        raise NotImplementedError

    def can_apply(self, source_text: str) -> bool:
        return bool(self.enumerate_actions(source_text))

    def apply(self, source_text: str, action_spec: ActionSpec) -> OperatorApplication:
        raise NotImplementedError


def split_segments(source_text: str) -> list[tuple[bool, str]]:
    segments: list[tuple[bool, str]] = []
    buffer: list[str] = []
    is_code = True
    index = 0

    def flush(next_is_code: bool) -> None:
        nonlocal buffer, is_code
        if buffer:
            segments.append((is_code, "".join(buffer)))
            buffer = []
        is_code = next_is_code

    while index < len(source_text):
        current = source_text[index]
        lookahead = source_text[index + 1] if index + 1 < len(source_text) else ""

        if is_code and current == "/" and lookahead == "/":
            flush(False)
            buffer.extend([current, lookahead])
            index += 2
            while index < len(source_text):
                buffer.append(source_text[index])
                if source_text[index] == "\n":
                    index += 1
                    break
                index += 1
            flush(True)
            continue

        if is_code and current == "/" and lookahead == "*":
            flush(False)
            buffer.extend([current, lookahead])
            index += 2
            while index < len(source_text):
                buffer.append(source_text[index])
                if source_text[index] == "*" and index + 1 < len(source_text) and source_text[index + 1] == "/":
                    buffer.append("/")
                    index += 2
                    break
                index += 1
            flush(True)
            continue

        if is_code and current in {'"', "'"}:
            quote = current
            flush(False)
            buffer.append(current)
            index += 1
            escaped = False
            while index < len(source_text):
                char = source_text[index]
                buffer.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    index += 1
                    break
                index += 1
            flush(True)
            continue

        buffer.append(current)
        index += 1

    if buffer:
        segments.append((is_code, "".join(buffer)))
    return segments


def mask_non_code(source_text: str) -> str:
    masked: list[str] = []
    for is_code, segment in split_segments(source_text):
        if is_code:
            masked.append(segment)
            continue
        masked.append("".join("\n" if char == "\n" else " " for char in segment))
    return "".join(masked)


def mask_preprocessor_lines(source_text: str) -> str:
    masked_lines: list[str] = []
    for line in source_text.splitlines(keepends=True):
        line_body = line[:-1] if line.endswith("\n") else line
        newline = "\n" if line.endswith("\n") else ""
        if line_body.lstrip().startswith("#"):
            masked_lines.append(" " * len(line_body) + newline)
        else:
            masked_lines.append(line)
    return "".join(masked_lines)


def indent_block(lines: list[str], indent: str) -> list[str]:
    indented: list[str] = []
    for line in lines:
        if line.strip():
            indented.append(f"{indent}{line.strip()}")
        else:
            indented.append("")
    return indented


def build_line_offsets(source_text: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(source_text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def position_to_line_col(offset: int, line_offsets: list[int]) -> tuple[int, int]:
    line_index = bisect_right(line_offsets, offset) - 1
    return line_index + 1, (offset - line_offsets[line_index]) + 1


def analyze_source(source_text: str) -> SourceAnalysis:
    cache_key = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    cached = _SOURCE_ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    analysis = SourceAnalysis(
        masked_code=mask_non_code(source_text),
        line_offsets=build_line_offsets(source_text),
        functions=extract_function_spans(source_text),
        clang_summary=load_clang_ast_summary(source_text),
    )
    _SOURCE_ANALYSIS_CACHE[cache_key] = analysis
    if len(_SOURCE_ANALYSIS_CACHE) > MAX_SOURCE_ANALYSIS_CACHE:
        oldest_key = next(iter(_SOURCE_ANALYSIS_CACHE))
        _SOURCE_ANALYSIS_CACHE.pop(oldest_key, None)
    return analysis


def extract_function_spans(source_text: str) -> list[FunctionSpan]:
    lines = source_text.splitlines()
    spans: list[FunctionSpan] = []
    start_index = 0
    ordinal = 0
    while start_index < len(lines):
        line = lines[start_index]
        match = FUNCTION_HEADER_PATTERN.match(line)
        if match is None:
            start_index += 1
            continue
        brace_depth = line.count("{") - line.count("}")
        if brace_depth <= 0:
            start_index += 1
            continue
        for end_index in range(start_index + 1, len(lines)):
            brace_depth += lines[end_index].count("{") - lines[end_index].count("}")
            if brace_depth == 0:
                ordinal += 1
                spans.append(
                    FunctionSpan(
                        name=match.group("name"),
                        ordinal=ordinal,
                        start_line=start_index + 1,
                        end_line=end_index + 1,
                    )
                )
                start_index = end_index + 1
                break
        else:
            start_index += 1
    return spans


def locate_function(functions: list[FunctionSpan], line_no: int) -> FunctionSpan:
    for function in functions:
        if function.start_line <= line_no <= function.end_line:
            return function
    return FunctionSpan(name="global", ordinal=0, start_line=1, end_line=line_no)


def build_node_id(
    *,
    family: str,
    function_name: str,
    line_no: int,
    column_no: int,
    detail: str = "",
) -> str:
    parts = [family, function_name or "global", f"L{line_no}", f"C{column_no}"]
    if detail:
        parts.append(detail)
    return ".".join(parts)


def slot_tag(prefix: str, index: int) -> str:
    return f"{prefix}_slot_{index:02d}"


def make_action_spec(
    operator_name: str,
    location_tag: str,
    parameter_tag: str,
    **parameters: object,
) -> ActionSpec:
    return ActionSpec(
        action_id=f"{operator_name}:{location_tag}:{parameter_tag}",
        operator_name=operator_name,
        location_tag=location_tag,
        parameter_tag=parameter_tag,
        parameters=parameters,
    )


def bind_slot_action(
    base_action: ActionSpec,
    candidate: NodeCandidate,
    **extra_parameters: object,
) -> ActionSpec:
    parameters = dict(base_action.parameters)
    parameters.update(
        {
            "node_id": candidate.node_id,
            "function_name": candidate.function_name,
            "line_no": candidate.line_no,
            "column_no": candidate.column_no,
            "slot_index": candidate.slot_index,
            **candidate.metadata,
            **extra_parameters,
        }
    )
    return ActionSpec(
        action_id=base_action.action_id,
        operator_name=base_action.operator_name,
        location_tag=base_action.location_tag,
        parameter_tag=base_action.parameter_tag,
        parameters=parameters,
    )


def bind_bulk_action(
    base_action: ActionSpec,
    candidates: list[NodeCandidate],
    **extra_parameters: object,
) -> ActionSpec:
    parameters = dict(base_action.parameters)
    llvm_block_ids = [
        str(candidate.metadata["llvm_block_id"])
        for candidate in candidates
        if str(candidate.metadata.get("llvm_block_id", ""))
    ]
    parameters.update(
        {
            "candidate_count": len(candidates),
            "llvm_block_ids": sorted(set(llvm_block_ids)),
            "llvm_block_id": llvm_block_ids[0] if llvm_block_ids else "",
            **extra_parameters,
        }
    )
    return ActionSpec(
        action_id=base_action.action_id,
        operator_name=base_action.operator_name,
        location_tag=base_action.location_tag,
        parameter_tag=base_action.parameter_tag,
        parameters=parameters,
    )


def replace_spans(source_text: str, replacements: list[tuple[int, int, str]]) -> str:
    updated = source_text
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    return updated


def _clang_identity(node: ClangAstNode) -> tuple[str, str, int, int, str]:
    return (node.kind, node.function_name, node.line_no, node.column_no, node.detail)


def _match_clang_node(
    nodes: list[ClangAstNode],
    *,
    function_name: str,
    line_no: int,
    column_no: int,
    detail: str = "",
    used: set[tuple[str, str, int, int, str]] | None = None,
) -> ClangAstNode | None:
    candidates = [
        node
        for node in nodes
        if node.function_name == function_name and node.line_no == line_no
    ]
    if detail:
        detail_matches = [node for node in candidates if node.detail == detail]
        if detail_matches:
            candidates = detail_matches
    if not candidates:
        return None

    for node in sorted(candidates, key=lambda item: abs(item.column_no - column_no)):
        identity = _clang_identity(node)
        if used is not None and identity in used:
            continue
        if used is not None:
            used.add(identity)
        return node
    return None


def _clang_metadata(clang_node: ClangAstNode | None, summary: ClangAstSummary) -> dict[str, object]:
    if clang_node is None:
        return {
            "analysis_backend": "lightweight",
            "clang_kind": "",
        }
    metadata = dict(clang_node.metadata)
    metadata["analysis_backend"] = str(metadata.get("analysis_backend", summary.backend))
    metadata["clang_kind"] = clang_node.kind
    return metadata


def _llvm_fallback_metadata(
    *,
    clang_node: ClangAstNode | None,
    summary: ClangAstSummary,
    function_name: str,
    line_no: int,
    column_no: int,
) -> dict[str, object]:
    metadata = _clang_metadata(clang_node, summary)
    if metadata.get("llvm_block_id"):
        return metadata

    llvm_block = _nearest_llvm_block(
        summary.llvm_ir_blocks,
        function_name=function_name,
        line_no=line_no,
        column_no=column_no,
    )
    if llvm_block is None:
        return metadata

    metadata.update(
        {
            "llvm_block_id": llvm_block.block_id,
            "llvm_successor_ids": list(llvm_block.successors),
            "llvm_branch_ordinal": llvm_block.branch_ordinal,
            "llvm_dbg_line": llvm_block.line_no,
            "llvm_dbg_column": llvm_block.column_no,
            "llvm_binding_strategy": "nearest_source_line",
        }
    )
    backend = str(metadata.get("analysis_backend", "lightweight"))
    if "llvm_ir" not in backend:
        metadata["analysis_backend"] = f"{backend}+llvm_ir_nearest"
    return metadata


def _nearest_llvm_block(
    blocks: list[LlvmIrBlock],
    *,
    function_name: str,
    line_no: int,
    column_no: int,
) -> LlvmIrBlock | None:
    function_blocks = [block for block in blocks if block.function_name == function_name]
    if not function_blocks:
        return None

    positioned = [block for block in function_blocks if block.line_no > 0]
    if positioned:
        return sorted(
            positioned,
            key=lambda block: (
                abs(block.line_no - line_no),
                abs(block.column_no - column_no),
                block.branch_ordinal,
            ),
        )[0]

    return sorted(function_blocks, key=lambda block: block.branch_ordinal)[0]


def _split_declarators(declarators: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(declarators):
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(declarators[start:index].strip())
            start = index + 1
    tail = declarators[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_declarator(declarator: str) -> dict[str, str] | None:
    match = re.match(
        r"^\s*(?P<pointer>\*+\s*)?(?P<name>[A-Za-z_]\w*)\s*(?P<suffix>(?:\[[^\]]*\]\s*)*)"
        r"(?:=\s*(?P<expr>.+))?\s*$",
        declarator,
    )
    if match is None:
        return None
    return {
        "name": match.group("name"),
        "pointer": (match.group("pointer") or "").replace(" ", ""),
        "suffix": (match.group("suffix") or "").strip(),
        "expr": (match.group("expr") or "").strip(),
    }


def _safe_identifier_suffix(identifier: str) -> str:
    suffix = re.sub(r"\W+", "_", identifier).strip("_")
    return suffix or "local"


def _is_preprocessor_line(masked_code: str, offset: int, line_offsets: list[int]) -> bool:
    line_no, _ = position_to_line_col(offset, line_offsets)
    line_start = line_offsets[line_no - 1]
    line_end = masked_code.find("\n", line_start)
    if line_end < 0:
        line_end = len(masked_code)
    return masked_code[line_start:line_end].lstrip().startswith("#")


def _is_float_literal(literal: str) -> bool:
    normalized = literal.rstrip("fFlLuU")
    return "." in normalized or "e" in normalized.lower()


def _top_level_return_indices(lines: list[str], function_start: int, function_end: int) -> list[tuple[int, str]]:
    returns: list[tuple[int, str]] = []
    depth = 1
    for index in range(function_start + 1, function_end):
        stripped = lines[index].strip()
        if depth == 1 and stripped.startswith("return "):
            returns.append((index, stripped[len("return ") :].rstrip(";")))
        depth += lines[index].count("{") - lines[index].count("}")
    return returns


def _function_body_depths(lines: list[str], function_start: int, function_end: int) -> dict[int, int]:
    depths: dict[int, int] = {}
    depth = 1
    for index in range(function_start + 1, function_end):
        depths[index] = depth
        depth += lines[index].count("{") - lines[index].count("}")
    return depths


class FlattenCfgOperator(SourceOperator):
    name = "flatten_cfg"

    def enumerate_actions(self, source_text: str) -> list[ActionSpec]:
        analyses = self._collect_candidates(source_text)
        if not analyses:
            return []

        actions: list[ActionSpec] = []
        for candidate in analyses[:MAX_FLATTEN_LOCATIONS]:
            node_candidate = candidate["node_candidate"]
            slot_index = node_candidate.slot_index
            while_action = make_action_spec(
                self.name,
                slot_tag("cfg_branch", slot_index),
                "while_switch",
                loop_style="while",
                state_prefix=f"__ql_{slot_index}",
            )
            for_action = make_action_spec(
                self.name,
                slot_tag("cfg_branch", slot_index),
                "for_switch",
                loop_style="for",
                state_prefix=f"__qlx_{slot_index}",
            )
            actions.append(bind_slot_action(while_action, node_candidate))
            actions.append(bind_slot_action(for_action, node_candidate))
        return actions

    def apply(self, source_text: str, action_spec: ActionSpec) -> OperatorApplication:
        node_id = str(action_spec.parameters.get("node_id", ""))
        analyses = self._collect_candidates(source_text)
        analysis = next((item for item in analyses if item["node_candidate"].node_id == node_id), None)
        if analysis is None:
            return OperatorApplication(
                action_name=action_spec.action_id,
                applied=False,
                source_text=source_text,
                notes=["No matching CFG branch node was found for flattening."],
            )

        state_prefix = str(action_spec.parameters.get("state_prefix", "__ql"))
        loop_style = str(action_spec.parameters.get("loop_style", "while"))
        state_name = f"{state_prefix}_state"
        result_name = f"{state_prefix}_result"

        lines = source_text.splitlines()
        function_start, function_end = analysis["function_span"]
        if_start, if_end = analysis["if_span"]
        final_return_index = analysis["final_return_index"]
        declaration_lines = lines[function_start + 1 : if_start]
        tail_lines = lines[if_end + 1 : final_return_index]
        indent = analysis["indent"]
        inner_indent = indent + "    "
        switch_indent = inner_indent + "    "
        case_indent = switch_indent + "    "
        loop_header = f"{indent}while (1) {{" if loop_style == "while" else f"{indent}for (;;) {{"

        flattened_body = [
            *declaration_lines,
            f"{indent}int {state_name} = 0;",
            f"{indent}int {result_name} = 0;",
            loop_header,
            f"{inner_indent}switch ({state_name}) {{",
            f"{switch_indent}case 0:",
            f"{case_indent}if ({analysis['condition']}) {{",
            f"{case_indent}    {result_name} = {analysis['early_return_expr']};",
            f"{case_indent}    {state_name} = 2;",
            f"{case_indent}}} else {{",
            f"{case_indent}    {state_name} = 1;",
            f"{case_indent}}}",
            f"{case_indent}break;",
            f"{switch_indent}case 1:",
            *indent_block(tail_lines, case_indent),
            f"{case_indent}{result_name} = {analysis['final_return_expr']};",
            f"{case_indent}{state_name} = 2;",
            f"{case_indent}break;",
            f"{switch_indent}default:",
            f"{case_indent}return {result_name};",
            f"{inner_indent}}}",
            f"{indent}}}",
        ]

        updated_lines = [
            *lines[: function_start + 1],
            *flattened_body,
            *lines[function_end:],
        ]
        return OperatorApplication(
            action_name=action_spec.action_id,
            applied=True,
            source_text="\n".join(updated_lines) + ("\n" if source_text.endswith("\n") else ""),
            metadata={
                "node_id": node_id,
                "function_name": str(action_spec.parameters.get("function_name", "")),
                "line_no": int(action_spec.parameters.get("line_no", 0)),
                "column_no": int(action_spec.parameters.get("column_no", 0)),
                "analysis_backend": str(action_spec.parameters.get("analysis_backend", "lightweight")),
                "cfg_block_id": str(action_spec.parameters.get("cfg_block_id", "")),
                "llvm_block_id": str(action_spec.parameters.get("llvm_block_id", "")),
                "flattened_cases": 2,
                "loop_style": loop_style,
                "state_prefix": state_prefix,
            },
        )

    def _collect_candidates(self, source_text: str) -> list[dict[str, object]]:
        analysis = analyze_source(source_text)
        lines = source_text.splitlines()
        functions = analysis.functions
        used_clang_nodes: set[tuple[str, str, int, int, str]] = set()
        candidates: list[dict[str, object]] = []
        slot_index = 0
        for function in functions:
            function_start = function.start_line - 1
            function_end = function.end_line - 1
            body_lines = lines[function_start + 1 : function_end]

            top_level_returns = _top_level_return_indices(lines, function_start, function_end)
            final_return_index = None
            final_return_expr = ""
            if top_level_returns:
                final_return_index, final_return_expr = top_level_returns[-1]
            if final_return_index is None or not final_return_expr:
                continue
            line_depths = _function_body_depths(lines, function_start, function_end)

            for offset, line in enumerate(body_lines):
                absolute_line_index = function_start + 1 + offset
                if line_depths.get(absolute_line_index, 0) != 1:
                    continue
                single_line_match = SINGLE_LINE_RETURN_IF_PATTERN.match(line)
                if single_line_match:
                    if_start = absolute_line_index
                    if if_start < final_return_index:
                        slot_index += 1
                        fallback_column = len(re.match(r"^\s*", lines[if_start]).group(0)) + 1
                        clang_node = _match_clang_node(
                            analysis.clang_summary.branches,
                            function_name=function.name,
                            line_no=if_start + 1,
                            column_no=fallback_column,
                            used=used_clang_nodes,
                        )
                        column_no = clang_node.column_no if clang_node is not None else fallback_column
                        node_candidate = NodeCandidate(
                            node_id=build_node_id(
                                family="cfg.if",
                                function_name=function.name,
                                line_no=if_start + 1,
                                column_no=column_no,
                            ),
                            function_name=function.name,
                            line_no=if_start + 1,
                            column_no=column_no,
                            slot_index=slot_index,
                            metadata={
                                "condition": single_line_match.group("condition").strip(),
                                **_llvm_fallback_metadata(
                                    clang_node=clang_node,
                                    summary=analysis.clang_summary,
                                    function_name=function.name,
                                    line_no=if_start + 1,
                                    column_no=column_no,
                                ),
                            },
                        )
                        candidates.append(
                            {
                                "node_candidate": node_candidate,
                                "function_span": (function_start, function_end),
                                "if_span": (if_start, if_start),
                                "condition": single_line_match.group("condition").strip(),
                                "early_return_expr": single_line_match.group("expr").strip(),
                                "final_return_expr": final_return_expr,
                                "final_return_index": final_return_index,
                                "indent": re.match(r"^\s*", lines[if_start]).group(0),
                            }
                        )
                    continue
                match = CONTROL_FLOW_PATTERN.match(line)
                if not match:
                    continue
                local_depth = line.count("{") - line.count("}")
                return_line_index = None
                early_return_expr = ""
                for nested_offset in range(offset + 1, len(body_lines)):
                    nested_line = body_lines[nested_offset]
                    if return_line_index is None and nested_line.strip().startswith("return "):
                        return_line_index = nested_offset
                        early_return_expr = nested_line.strip()[len("return ") :].rstrip(";")
                    local_depth += nested_line.count("{") - nested_line.count("}")
                    if local_depth == 0:
                        if return_line_index is not None:
                            if_start = function_start + 1 + offset
                            if_end = function_start + 1 + nested_offset
                            if if_end < final_return_index:
                                slot_index += 1
                                fallback_column = len(re.match(r"^\s*", lines[if_start]).group(0)) + 1
                                clang_node = _match_clang_node(
                                    analysis.clang_summary.branches,
                                    function_name=function.name,
                                    line_no=if_start + 1,
                                    column_no=fallback_column,
                                    used=used_clang_nodes,
                                )
                                column_no = clang_node.column_no if clang_node is not None else fallback_column
                                node_candidate = NodeCandidate(
                                    node_id=build_node_id(
                                        family="cfg.if",
                                        function_name=function.name,
                                        line_no=if_start + 1,
                                        column_no=column_no,
                                    ),
                                    function_name=function.name,
                                    line_no=if_start + 1,
                                    column_no=column_no,
                                    slot_index=slot_index,
                                    metadata={
                                        "condition": match.group("condition").strip(),
                                        **_llvm_fallback_metadata(
                                            clang_node=clang_node,
                                            summary=analysis.clang_summary,
                                            function_name=function.name,
                                            line_no=if_start + 1,
                                            column_no=column_no,
                                        ),
                                    },
                                )
                                candidates.append(
                                    {
                                        "node_candidate": node_candidate,
                                        "function_span": (function_start, function_end),
                                        "if_span": (if_start, if_end),
                                        "condition": match.group("condition").strip(),
                                        "early_return_expr": early_return_expr,
                                        "final_return_expr": final_return_expr,
                                        "final_return_index": final_return_index,
                                        "indent": re.match(r"^\s*", lines[if_start]).group(0),
                                    }
                                )
                        break
        return candidates


class SplitBlocksOperator(SourceOperator):
    name = "split_blocks"

    def enumerate_actions(self, source_text: str) -> list[ActionSpec]:
        candidates = [
            candidate
            for candidate in collect_declaration_candidates(source_text)
            if bool(candidate.metadata.get("can_split", False))
        ]
        if not candidates:
            return []

        actions: list[ActionSpec] = []
        for candidate in candidates[:MAX_DECLARATION_LOCATIONS]:
            base_action = make_action_spec(self.name, slot_tag("ast_decl", candidate.slot_index), "single")
            actions.append(bind_slot_action(base_action, candidate))
        actions.append(bind_bulk_action(make_action_spec(self.name, "ast_decl_bulk", "bulk"), candidates))
        return actions

    def apply(self, source_text: str, action_spec: ActionSpec) -> OperatorApplication:
        node_id = str(action_spec.parameters.get("node_id", ""))
        declaration_candidates = [
            candidate
            for candidate in collect_declaration_candidates(source_text)
            if bool(candidate.metadata.get("can_split", False))
        ]
        if node_id:
            candidate = next((item for item in declaration_candidates if item.node_id == node_id), None)
            if candidate is None:
                return OperatorApplication(
                    action_name=action_spec.action_id,
                    applied=False,
                    source_text=source_text,
                    notes=["No matching AST declaration node was found."],
                )

            replacement = f"{candidate.metadata['indent']}{candidate.metadata['ctype']} {candidate.metadata['name']};\n{candidate.metadata['indent']}{candidate.metadata['name']} = {candidate.metadata['expr']};"
            updated_source = replace_spans(source_text, [(int(candidate.metadata["span_start"]), int(candidate.metadata["span_end"]), replacement)])
            return OperatorApplication(
                action_name=action_spec.action_id,
                applied=True,
                source_text=updated_source,
                metadata={
                    "node_id": candidate.node_id,
                    "function_name": candidate.function_name,
                    "line_no": candidate.line_no,
                    "column_no": candidate.column_no,
                    "split_declarations": 1,
                },
            )

        replacements: list[tuple[int, int, str]] = []
        for candidate in declaration_candidates:
            replacements.append(
                (
                    int(candidate.metadata["span_start"]),
                    int(candidate.metadata["span_end"]),
                    f"{candidate.metadata['indent']}{candidate.metadata['ctype']} {candidate.metadata['name']};\n{candidate.metadata['indent']}{candidate.metadata['name']} = {candidate.metadata['expr']};",
                )
            )
        updated_source = replace_spans(source_text, replacements)
        return OperatorApplication(
            action_name=action_spec.action_id,
            applied=bool(replacements),
            source_text=updated_source,
            metadata={
                "split_declarations": len(replacements),
                "llvm_block_ids": action_spec.parameters.get("llvm_block_ids", []),
                "llvm_block_id": str(action_spec.parameters.get("llvm_block_id", "")),
            },
            notes=[] if replacements else ["No initialized local declarations were found."],
        )


class RenameLocalsOperator(SourceOperator):
    name = "rename_locals"

    def enumerate_actions(self, source_text: str) -> list[ActionSpec]:
        candidates = collect_declaration_candidates(source_text)
        if not candidates:
            return []

        actions: list[ActionSpec] = []
        for candidate in candidates[:MAX_LOCAL_LOCATIONS]:
            base_action = make_action_spec(self.name, slot_tag("ast_local", candidate.slot_index), "compact_prefix", prefix="__ql_local")
            actions.append(bind_slot_action(base_action, candidate))
        actions.append(bind_bulk_action(make_action_spec(self.name, "ast_local_bulk", "compact_prefix", prefix="__ql_local"), candidates))
        actions.append(bind_bulk_action(make_action_spec(self.name, "ast_local_bulk", "debug_prefix", prefix="__ql_dbg_local"), candidates))
        return actions

    def apply(self, source_text: str, action_spec: ActionSpec) -> OperatorApplication:
        prefix = str(action_spec.parameters.get("prefix", "__ql_local"))
        declaration_candidates = collect_declaration_candidates(source_text)
        node_id = str(action_spec.parameters.get("node_id", ""))
        selected_candidates = declaration_candidates
        if node_id:
            selected_candidates = [candidate for candidate in declaration_candidates if candidate.node_id == node_id]
            if not selected_candidates:
                return OperatorApplication(
                    action_name=action_spec.action_id,
                    applied=False,
                    source_text=source_text,
                    notes=["No matching AST local declaration node was found."],
                )

        rename_map = {
            str(candidate.metadata["name"]): f"{prefix}_{_safe_identifier_suffix(str(candidate.metadata['name']))}"
            for candidate in selected_candidates
        }
        if not rename_map:
            return OperatorApplication(
                action_name=action_spec.action_id,
                applied=False,
                source_text=source_text,
                notes=["No eligible local identifiers were found."],
            )

        masked_code = mask_preprocessor_lines(mask_non_code(source_text))
        replacements: list[tuple[int, int, str]] = []
        for original_name, new_name in rename_map.items():
            pattern = re.compile(IDENTIFIER_PATTERN_TEMPLATE.format(identifier=re.escape(original_name)))
            for match in pattern.finditer(masked_code):
                replacements.append((match.start(), match.end(), new_name))

        updated_source = replace_spans(source_text, replacements)
        return OperatorApplication(
            action_name=action_spec.action_id,
            applied=bool(replacements),
            source_text=updated_source,
            metadata={
                "node_id": node_id,
                "renamed_identifiers": rename_map,
                "replacement_count": len(replacements),
                "llvm_block_ids": action_spec.parameters.get("llvm_block_ids", []),
                "llvm_block_id": str(action_spec.parameters.get("llvm_block_id", "")),
            },
        )


class SubstituteInstructionsOperator(SourceOperator):
    name = "substitute_instructions"

    def enumerate_actions(self, source_text: str) -> list[ActionSpec]:
        candidates = collect_addition_candidates(source_text)
        if not candidates:
            return []

        actions: list[ActionSpec] = []
        for candidate in candidates[:MAX_ADDITION_LOCATIONS]:
            base_action = make_action_spec(self.name, slot_tag("ast_add_expr", candidate.slot_index), "neg_add", style="neg_add")
            actions.append(bind_slot_action(base_action, candidate))
        actions.append(bind_bulk_action(make_action_spec(self.name, "ast_add_expr_bulk", "neg_add", style="neg_add"), candidates))
        return actions

    def apply(self, source_text: str, action_spec: ActionSpec) -> OperatorApplication:
        node_id = str(action_spec.parameters.get("node_id", ""))
        candidates = collect_addition_candidates(source_text)
        selected_candidates = candidates
        if node_id:
            selected_candidates = [candidate for candidate in candidates if candidate.node_id == node_id]
            if not selected_candidates:
                return OperatorApplication(
                    action_name=action_spec.action_id,
                    applied=False,
                    source_text=source_text,
                    notes=["No matching AST addition-expression node was found."],
                )

        replacements: list[tuple[int, int, str]] = []
        for candidate in selected_candidates:
            replacements.append(
                (
                    int(candidate.metadata["span_start"]),
                    int(candidate.metadata["span_end"]),
                    f"(({candidate.metadata['left']}) - (-({candidate.metadata['right']})))",
                )
            )
        updated_source = replace_spans(source_text, replacements)
        return OperatorApplication(
            action_name=action_spec.action_id,
            applied=bool(replacements),
            source_text=updated_source,
            metadata={
                "node_id": node_id,
                "substituted_additions": len(replacements),
                "llvm_block_ids": action_spec.parameters.get("llvm_block_ids", []),
                "llvm_block_id": str(action_spec.parameters.get("llvm_block_id", "")),
            },
            notes=[] if replacements else ["No identifier addition expression was found."],
        )


class EncodeLiteralsOperator(SourceOperator):
    name = "encode_literals"

    def enumerate_actions(self, source_text: str) -> list[ActionSpec]:
        candidates = collect_literal_candidates(source_text)
        if not candidates:
            return []

        actions: list[ActionSpec] = []
        for candidate in candidates[:MAX_LITERAL_LOCATIONS]:
            actions.append(bind_slot_action(make_action_spec(self.name, slot_tag("ast_literal", candidate.slot_index), "offset_3", offset=3), candidate))
            actions.append(bind_slot_action(make_action_spec(self.name, slot_tag("ast_literal", candidate.slot_index), "offset_7", offset=7), candidate))
        actions.append(bind_bulk_action(make_action_spec(self.name, "ast_literal_bulk", "offset_7", offset=7), candidates))
        return actions

    def apply(self, source_text: str, action_spec: ActionSpec) -> OperatorApplication:
        offset = int(action_spec.parameters.get("offset", 7))
        node_id = str(action_spec.parameters.get("node_id", ""))
        candidates = collect_literal_candidates(source_text)
        selected_candidates = candidates
        if node_id:
            selected_candidates = [candidate for candidate in candidates if candidate.node_id == node_id]
            if not selected_candidates:
                return OperatorApplication(
                    action_name=action_spec.action_id,
                    applied=False,
                    source_text=source_text,
                    notes=["No matching AST integer-literal node was found."],
                )

        replacements: list[tuple[int, int, str]] = []
        for candidate in selected_candidates:
            literal = str(candidate.metadata["literal"])
            literal_core = literal.rstrip("fFlLuU")
            if _is_float_literal(literal):
                suffix = "f" if literal.lower().endswith("f") else ""
                replacement = f"(({literal_core} + {offset}.0{suffix}) - {offset}.0{suffix})"
            else:
                literal_value = int(literal_core)
                if literal_value == 0:
                    replacement = "(1 - 1)"
                else:
                    replacement = f"(({literal_value + offset}) - {offset})"
            if literal_core in {"0", "0.0", ".0"}:
                replacement = "(1 - 1)"
            replacements.append((int(candidate.metadata["span_start"]), int(candidate.metadata["span_end"]), replacement))

        updated_source = replace_spans(source_text, replacements)
        return OperatorApplication(
            action_name=action_spec.action_id,
            applied=bool(replacements),
            source_text=updated_source,
            metadata={
                "node_id": node_id,
                "encoded_literals": len(replacements),
                "offset": offset,
                "llvm_block_ids": action_spec.parameters.get("llvm_block_ids", []),
                "llvm_block_id": str(action_spec.parameters.get("llvm_block_id", "")),
            },
            notes=[] if replacements else ["No numeric literals were found outside strings/comments."],
        )


def collect_declaration_candidates(source_text: str) -> list[NodeCandidate]:
    analysis = analyze_source(source_text)
    masked_code = analysis.masked_code
    line_offsets = analysis.line_offsets
    functions = analysis.functions
    used_clang_nodes: set[tuple[str, str, int, int, str]] = set()
    candidates: list[NodeCandidate] = []
    slot_index = 0
    for match in DECLARATION_PATTERN.finditer(masked_code):
        line_no, _ = position_to_line_col(match.start(), line_offsets)
        function = locate_function(functions, line_no)
        declarators = _split_declarators(match.group("declarators"))
        for declarator in declarators:
            parsed = _parse_declarator(declarator)
            if parsed is None:
                continue
            name = parsed["name"]
            if name.startswith("__ql_"):
                continue
            name_offset = masked_code.find(name, match.start(), match.end())
            if name_offset < 0:
                continue
            fallback_column = position_to_line_col(name_offset, line_offsets)[1]
            clang_node = _match_clang_node(
                analysis.clang_summary.declarations,
                function_name=function.name,
                line_no=line_no,
                column_no=fallback_column,
                detail=name,
                used=used_clang_nodes,
            )
            column_no = clang_node.column_no if clang_node is not None else fallback_column
            expr = parsed["expr"]
            is_single_plain_initialized = (
                len(declarators) == 1
                and bool(expr)
                and not parsed["suffix"]
                and not parsed["pointer"]
            )
            slot_index += 1
            candidates.append(
                NodeCandidate(
                    node_id=build_node_id(
                        family="ast.decl",
                        function_name=function.name,
                        line_no=line_no,
                        column_no=column_no,
                        detail=name,
                    ),
                    function_name=function.name,
                    line_no=line_no,
                    column_no=column_no,
                    slot_index=slot_index,
                    metadata={
                        "name": name,
                        "ctype": f"{match.group('ctype')}{' ' + parsed['pointer'] if parsed['pointer'] else ''}",
                        "expr": expr,
                        "indent": match.group("indent"),
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "name_span_start": name_offset,
                        "name_span_end": name_offset + len(name),
                        "is_initialized": bool(expr),
                        "can_split": is_single_plain_initialized,
                        "declarator_count": len(declarators),
                        "suffix": parsed["suffix"],
                        **_llvm_fallback_metadata(
                            clang_node=clang_node,
                            summary=analysis.clang_summary,
                            function_name=function.name,
                            line_no=line_no,
                            column_no=column_no,
                        ),
                    },
                )
            )
    return candidates


def collect_addition_candidates(source_text: str) -> list[NodeCandidate]:
    analysis = analyze_source(source_text)
    masked_code = analysis.masked_code
    line_offsets = analysis.line_offsets
    functions = analysis.functions
    used_clang_nodes: set[tuple[str, str, int, int, str]] = set()
    candidates: list[NodeCandidate] = []
    slot_index = 0
    for match in IDENTIFIER_ADDITION_PATTERN.finditer(masked_code):
        line_no, fallback_column = position_to_line_col(match.start(), line_offsets)
        function = locate_function(functions, line_no)
        clang_node = _match_clang_node(
            analysis.clang_summary.additions,
            function_name=function.name,
            line_no=line_no,
            column_no=fallback_column,
            used=used_clang_nodes,
        )
        column_no = clang_node.column_no if clang_node is not None else fallback_column
        slot_index += 1
        candidates.append(
            NodeCandidate(
                node_id=build_node_id(
                    family="ast.binop_add",
                    function_name=function.name,
                    line_no=line_no,
                    column_no=column_no,
                ),
                function_name=function.name,
                line_no=line_no,
                column_no=column_no,
                slot_index=slot_index,
                metadata={
                    "left": match.group(1),
                    "right": match.group(2),
                    "span_start": match.start(),
                    "span_end": match.end(),
                    **_llvm_fallback_metadata(
                        clang_node=clang_node,
                        summary=analysis.clang_summary,
                        function_name=function.name,
                        line_no=line_no,
                        column_no=column_no,
                    ),
                },
            )
        )
    return candidates


def collect_literal_candidates(source_text: str) -> list[NodeCandidate]:
    analysis = analyze_source(source_text)
    masked_code = analysis.masked_code
    line_offsets = analysis.line_offsets
    functions = analysis.functions
    used_clang_nodes: set[tuple[str, str, int, int, str]] = set()
    candidates: list[NodeCandidate] = []
    slot_index = 0
    for match in NUMERIC_LITERAL_PATTERN.finditer(masked_code):
        if _is_preprocessor_line(masked_code, match.start(), line_offsets):
            continue
        line_no, fallback_column = position_to_line_col(match.start(), line_offsets)
        function = locate_function(functions, line_no)
        slot_index += 1
        literal = match.group(0)
        clang_node = _match_clang_node(
            analysis.clang_summary.literals,
            function_name=function.name,
            line_no=line_no,
            column_no=fallback_column,
            detail=literal,
            used=used_clang_nodes,
        )
        column_no = clang_node.column_no if clang_node is not None else fallback_column
        candidates.append(
            NodeCandidate(
                node_id=build_node_id(
                    family="ast.float_literal" if _is_float_literal(literal) else "ast.int_literal",
                    function_name=function.name,
                    line_no=line_no,
                    column_no=column_no,
                    detail=f"v{literal}",
                ),
                function_name=function.name,
                line_no=line_no,
                column_no=column_no,
                slot_index=slot_index,
                metadata={
                    "literal": literal,
                    "literal_kind": "float" if _is_float_literal(literal) else "integer",
                    "span_start": match.start(),
                    "span_end": match.end(),
                    **_llvm_fallback_metadata(
                        clang_node=clang_node,
                        summary=analysis.clang_summary,
                        function_name=function.name,
                        line_no=line_no,
                        column_no=column_no,
                    ),
                },
            )
        )
    return candidates


def build_operator_registry() -> dict[str, SourceOperator]:
    operators: list[SourceOperator] = [
        FlattenCfgOperator(),
        SplitBlocksOperator(),
        RenameLocalsOperator(),
        SubstituteInstructionsOperator(),
        EncodeLiteralsOperator(),
    ]
    return {operator.name: operator for operator in operators}


def build_default_action_library() -> list[ActionSpec]:
    actions: list[ActionSpec] = []

    for index in range(1, MAX_FLATTEN_LOCATIONS + 1):
        actions.append(make_action_spec("flatten_cfg", slot_tag("cfg_branch", index), "while_switch", loop_style="while", state_prefix=f"__ql_{index}"))
        actions.append(make_action_spec("flatten_cfg", slot_tag("cfg_branch", index), "for_switch", loop_style="for", state_prefix=f"__qlx_{index}"))

    for index in range(1, MAX_DECLARATION_LOCATIONS + 1):
        actions.append(make_action_spec("split_blocks", slot_tag("ast_decl", index), "single"))
    actions.append(make_action_spec("split_blocks", "ast_decl_bulk", "bulk"))

    for index in range(1, MAX_LOCAL_LOCATIONS + 1):
        actions.append(make_action_spec("rename_locals", slot_tag("ast_local", index), "compact_prefix", prefix="__ql_local"))
    actions.append(make_action_spec("rename_locals", "ast_local_bulk", "compact_prefix", prefix="__ql_local"))
    actions.append(make_action_spec("rename_locals", "ast_local_bulk", "debug_prefix", prefix="__ql_dbg_local"))

    for index in range(1, MAX_ADDITION_LOCATIONS + 1):
        actions.append(make_action_spec("substitute_instructions", slot_tag("ast_add_expr", index), "neg_add", style="neg_add"))
    actions.append(make_action_spec("substitute_instructions", "ast_add_expr_bulk", "neg_add", style="neg_add"))

    for index in range(1, MAX_LITERAL_LOCATIONS + 1):
        actions.append(make_action_spec("encode_literals", slot_tag("ast_literal", index), "offset_3", offset=3))
        actions.append(make_action_spec("encode_literals", slot_tag("ast_literal", index), "offset_7", offset=7))
    actions.append(make_action_spec("encode_literals", "ast_literal_bulk", "offset_7", offset=7))

    return actions
