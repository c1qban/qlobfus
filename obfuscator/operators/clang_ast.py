from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CLANG_AST_CANDIDATES = [
    "clang",
    "clang.exe",
]

WINDOWS_CLANG_FALLBACKS = [
    r"C:\Program Files\LLVM\bin\clang.exe",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\bin\clang.exe",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clang.exe",
]

CFG_BLOCK_PATTERN = re.compile(r"^\s*\[B(?P<block_id>\d+)(?:\s+\((?P<tag>[^)]+)\))?\]\s*$")
CFG_FUNCTION_PATTERN = re.compile(r"^\s*(?:[A-Za-z_]\w*[\s\*]+)+(?P<name>[A-Za-z_]\w*)\s*\([^;]*\)\s*$")
CFG_SUCC_PATTERN = re.compile(r"^\s*Succs\s*\(\d+\):\s*(?P<succs>.+?)\s*$", re.IGNORECASE)
LLVM_DEFINE_PATTERN = re.compile(r"^\s*define\b.*@(?P<name>[A-Za-z_]\w*)\b.*\{\s*$")
LLVM_LABEL_PATTERN = re.compile(r"^(?P<label>[A-Za-z0-9._-]+):(?:\s*;.*)?$")
LLVM_DBG_LOC_PATTERN = re.compile(r"^!(?P<id>\d+)\s*=\s*!DILocation\(line:\s*(?P<line>\d+),\s*column:\s*(?P<col>\d+)")
LLVM_SUCCESSOR_PATTERN = re.compile(r"label\s+%?(?P<label>[A-Za-z0-9._-]+)")
LLVM_DBG_REF_PATTERN = re.compile(r"!dbg\s+!(?P<id>\d+)")


@dataclass(frozen=True)
class ClangAstNode:
    kind: str
    function_name: str
    line_no: int
    column_no: int
    detail: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ClangCfgBlock:
    function_name: str
    block_id: str
    branch_ordinal: int
    line_no: int = 0
    column_no: int = 0
    successors: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LlvmIrBlock:
    function_name: str
    block_id: str
    branch_ordinal: int
    line_no: int = 0
    column_no: int = 0
    successors: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ClangAstSummary:
    available: bool
    backend: str = "lightweight"
    compiler_path: str = ""
    declarations: list[ClangAstNode] = field(default_factory=list)
    additions: list[ClangAstNode] = field(default_factory=list)
    literals: list[ClangAstNode] = field(default_factory=list)
    branches: list[ClangAstNode] = field(default_factory=list)
    cfg_blocks: list[ClangCfgBlock] = field(default_factory=list)
    llvm_ir_blocks: list[LlvmIrBlock] = field(default_factory=list)
    cfg_available: bool = False
    llvm_ir_available: bool = False
    notes: list[str] = field(default_factory=list)


def choose_clang_ast_binary(preferred: str | None = None) -> str | None:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(CLANG_AST_CANDIDATES)

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    for candidate in WINDOWS_CLANG_FALLBACKS:
        if Path(candidate).exists():
            return str(Path(candidate))
    return None


def load_clang_ast_summary(source_text: str, preferred_clang: str | None = None) -> ClangAstSummary:
    compiler_path = choose_clang_ast_binary(preferred_clang)
    if compiler_path is None:
        return ClangAstSummary(
            available=False,
            backend="lightweight",
            notes=["clang was not found on PATH; using lightweight structural analysis."],
        )

    try:
        temp_dir_path = _create_analysis_temp_dir()
        try:
            temp_dir = str(temp_dir_path)
            source_path = Path(temp_dir) / "module.c"
            source_path.write_text(source_text, encoding="utf-8")

            ast_completed = _run_command(
                [
                    compiler_path,
                    "-Xclang",
                    "-ast-dump=json",
                    "-fsyntax-only",
                    str(source_path),
                ]
            )
            if ast_completed is None:
                return ClangAstSummary(
                    available=False,
                    backend="lightweight",
                    compiler_path=compiler_path,
                    notes=["clang AST dump timed out; using lightweight structural analysis."],
                )
            if ast_completed.returncode != 0:
                stderr = ast_completed.stderr.strip()
                note = "clang AST dump failed; using lightweight structural analysis."
                if stderr:
                    note = f"{note} stderr: {stderr}"
                return ClangAstSummary(
                    available=False,
                    backend="lightweight",
                    compiler_path=compiler_path,
                    notes=[note],
                )

            try:
                payload = json.loads(ast_completed.stdout)
            except json.JSONDecodeError as exc:
                return ClangAstSummary(
                    available=False,
                    backend="lightweight",
                    compiler_path=compiler_path,
                    notes=[f"clang AST JSON could not be parsed: {exc}"],
                )

            summary = parse_clang_ast_json(
                payload,
                compiler_path=compiler_path,
                source_line_count=len(source_text.splitlines()),
            )

            cfg_completed = _run_command(
                [
                    compiler_path,
                    "-Xclang",
                    "-analyze",
                    "-Xclang",
                    "-analyzer-checker=debug.DumpCFG",
                    "-fsyntax-only",
                    str(source_path),
                ]
            )
            if cfg_completed is None:
                summary.notes.append("clang CFG dump timed out; keeping AST-only binding.")
            elif cfg_completed.returncode == 0:
                cfg_dump = "\n".join(part for part in [cfg_completed.stdout, cfg_completed.stderr] if part)
                _merge_cfg_dump(summary, cfg_dump)
            else:
                summary.notes.append("clang CFG dump failed; keeping AST-only binding.")

            llvm_ir_path = Path(temp_dir) / "module.ll"
            llvm_completed = _run_command(
                [
                    compiler_path,
                    "-S",
                    "-emit-llvm",
                    "-gline-tables-only",
                    "-O0",
                    str(source_path),
                    "-o",
                    str(llvm_ir_path),
                ]
            )
            if llvm_completed is None:
                summary.notes.append("LLVM IR emission timed out; keeping AST/CFG-only binding.")
            elif llvm_completed.returncode == 0 and llvm_ir_path.exists():
                _merge_llvm_ir(summary, llvm_ir_path.read_text(encoding="utf-8"))
            else:
                summary.notes.append("LLVM IR emission failed; keeping AST/CFG-only binding.")

            _enrich_nodes(summary)
            return summary
        finally:
            shutil.rmtree(temp_dir_path, ignore_errors=True)
    except OSError as exc:
        return ClangAstSummary(
            available=False,
            backend="lightweight",
            compiler_path=compiler_path,
            notes=[f"Failed to execute clang analysis pipeline: {exc}"],
        )


def _create_analysis_temp_dir() -> Path:
    temp_root = Path.cwd() / "artifacts" / "tmp" / "clang_analysis"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"qlobfus_clang_ast_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=False, exist_ok=False)
    return temp_dir


def parse_clang_ast_json(
    payload: dict[str, Any],
    *,
    compiler_path: str = "",
    source_line_count: int | None = None,
) -> ClangAstSummary:
    summary = ClangAstSummary(
        available=True,
        backend="clang_ast_json",
        compiler_path=compiler_path,
    )
    branch_counters: dict[str, int] = defaultdict(int)
    _walk_ast(
        payload,
        summary,
        current_function="global",
        branch_counters=branch_counters,
        source_line_count=source_line_count,
    )
    return summary


def parse_clang_cfg_dump(cfg_dump: str) -> list[ClangCfgBlock]:
    blocks: list[ClangCfgBlock] = []
    current_function = "global"
    current_block_id = ""
    current_tag = ""
    current_successors: list[str] = []
    current_branch_like = False
    branch_counters: dict[str, int] = defaultdict(int)

    def flush() -> None:
        nonlocal current_block_id, current_tag, current_successors, current_branch_like
        if current_block_id and current_branch_like and current_tag.upper() not in {"ENTRY", "EXIT"}:
            branch_counters[current_function] += 1
            blocks.append(
                ClangCfgBlock(
                    function_name=current_function,
                    block_id=f"B{current_block_id}",
                    branch_ordinal=branch_counters[current_function],
                    successors=list(current_successors),
                    metadata={"cfg_tag": current_tag},
                )
            )
        current_block_id = ""
        current_tag = ""
        current_successors = []
        current_branch_like = False

    for raw_line in cfg_dump.splitlines():
        line = raw_line.rstrip()
        function_match = CFG_FUNCTION_PATTERN.match(line)
        if function_match and "[" not in line and ":" not in line:
            flush()
            current_function = function_match.group("name")
            continue

        block_match = CFG_BLOCK_PATTERN.match(line)
        if block_match:
            flush()
            current_block_id = block_match.group("block_id")
            current_tag = block_match.group("tag") or ""
            continue

        if not current_block_id:
            continue

        succ_match = CFG_SUCC_PATTERN.match(line)
        if succ_match:
            current_successors = [token for token in succ_match.group("succs").split() if token.startswith("B")]
            continue

        normalized = line.strip().lower()
        if normalized.startswith("t:") or " if " in f" {normalized} " or normalized.startswith("if "):
            current_branch_like = True
        if normalized.startswith("switch ") or " switch " in f" {normalized} ":
            current_branch_like = True

    flush()
    return blocks


def parse_llvm_ir_text(ir_text: str) -> list[LlvmIrBlock]:
    debug_locations: dict[str, tuple[int, int]] = {}
    for line in ir_text.splitlines():
        dbg_match = LLVM_DBG_LOC_PATTERN.match(line.strip())
        if dbg_match:
            debug_locations[dbg_match.group("id")] = (
                int(dbg_match.group("line")),
                int(dbg_match.group("col")),
            )

    blocks: list[LlvmIrBlock] = []
    current_function = "global"
    current_block = "entry"
    branch_counters: dict[str, int] = defaultdict(int)

    for raw_line in ir_text.splitlines():
        line = raw_line.rstrip()
        define_match = LLVM_DEFINE_PATTERN.match(line)
        if define_match:
            current_function = define_match.group("name")
            current_block = "entry"
            continue
        if line.strip() == "}":
            current_function = "global"
            current_block = "entry"
            continue

        label_match = LLVM_LABEL_PATTERN.match(line)
        if label_match:
            current_block = label_match.group("label")
            continue

        stripped = line.strip()
        if not stripped.startswith("br "):
            continue

        dbg_match = LLVM_DBG_REF_PATTERN.search(stripped)
        dbg_line = 0
        dbg_col = 0
        if dbg_match and dbg_match.group("id") in debug_locations:
            dbg_line, dbg_col = debug_locations[dbg_match.group("id")]

        successors = [match.group("label") for match in LLVM_SUCCESSOR_PATTERN.finditer(stripped)]
        branch_counters[current_function] += 1
        blocks.append(
            LlvmIrBlock(
                function_name=current_function,
                block_id=current_block,
                branch_ordinal=branch_counters[current_function],
                line_no=dbg_line,
                column_no=dbg_col,
                successors=successors,
                metadata={"instruction": stripped},
            )
        )

    return blocks


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10.0,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None


def _merge_cfg_dump(summary: ClangAstSummary, cfg_dump: str) -> None:
    blocks = parse_clang_cfg_dump(cfg_dump)
    if blocks:
        summary.cfg_blocks = blocks
        summary.cfg_available = True


def _merge_llvm_ir(summary: ClangAstSummary, ir_text: str) -> None:
    blocks = parse_llvm_ir_text(ir_text)
    if blocks:
        summary.llvm_ir_blocks = blocks
        summary.llvm_ir_available = True


def _enrich_nodes(summary: ClangAstSummary) -> None:
    summary.branches = [
        _attach_cfg_and_llvm_metadata(node, summary)
        for node in summary.branches
    ]
    summary.declarations = [
        _attach_llvm_metadata(node, summary)
        for node in summary.declarations
    ]
    summary.additions = [
        _attach_llvm_metadata(node, summary)
        for node in summary.additions
    ]
    summary.literals = [
        _attach_llvm_metadata(node, summary)
        for node in summary.literals
    ]


def _attach_cfg_and_llvm_metadata(node: ClangAstNode, summary: ClangAstSummary) -> ClangAstNode:
    metadata = dict(node.metadata)
    cfg_block = _match_cfg_block(node, summary.cfg_blocks)
    llvm_block = _match_llvm_block(node, summary.llvm_ir_blocks)
    if cfg_block is not None:
        metadata["cfg_block_id"] = cfg_block.block_id
        metadata["cfg_successor_ids"] = list(cfg_block.successors)
        metadata["cfg_branch_ordinal"] = cfg_block.branch_ordinal
        metadata["cfg_tag"] = str(cfg_block.metadata.get("cfg_tag", ""))
    if llvm_block is not None:
        metadata["llvm_block_id"] = llvm_block.block_id
        metadata["llvm_successor_ids"] = list(llvm_block.successors)
        metadata["llvm_branch_ordinal"] = llvm_block.branch_ordinal
        metadata["llvm_dbg_line"] = llvm_block.line_no
        metadata["llvm_dbg_column"] = llvm_block.column_no
    metadata["analysis_backend"] = _join_backend_tags(
        "clang_ast_json",
        "clang_cfg" if cfg_block is not None else "",
        "llvm_ir" if llvm_block is not None else "",
    )
    return ClangAstNode(
        kind=node.kind,
        function_name=node.function_name,
        line_no=node.line_no,
        column_no=node.column_no,
        detail=node.detail,
        metadata=metadata,
    )


def _attach_llvm_metadata(node: ClangAstNode, summary: ClangAstSummary) -> ClangAstNode:
    metadata = dict(node.metadata)
    llvm_block = _match_llvm_block(node, summary.llvm_ir_blocks, allow_ordinal_fallback=False)
    if llvm_block is not None:
        metadata["llvm_block_id"] = llvm_block.block_id
        metadata["llvm_successor_ids"] = list(llvm_block.successors)
        metadata["llvm_branch_ordinal"] = llvm_block.branch_ordinal
        metadata["llvm_dbg_line"] = llvm_block.line_no
        metadata["llvm_dbg_column"] = llvm_block.column_no
        metadata["analysis_backend"] = _join_backend_tags("clang_ast_json", "llvm_ir")
    else:
        metadata["analysis_backend"] = "clang_ast_json"
    return ClangAstNode(
        kind=node.kind,
        function_name=node.function_name,
        line_no=node.line_no,
        column_no=node.column_no,
        detail=node.detail,
        metadata=metadata,
    )


def _match_cfg_block(node: ClangAstNode, blocks: list[ClangCfgBlock]) -> ClangCfgBlock | None:
    function_blocks = [block for block in blocks if block.function_name == node.function_name]
    if not function_blocks:
        return None

    if node.line_no > 0:
        same_line = [block for block in function_blocks if block.line_no == node.line_no and block.line_no > 0]
        if same_line:
            return sorted(same_line, key=lambda item: abs(item.column_no - node.column_no))[0]

    ordinal = int(node.metadata.get("ast_ordinal", 0))
    if ordinal > 0:
        for block in function_blocks:
            if block.branch_ordinal == ordinal:
                return block
    return function_blocks[0]


def _match_llvm_block(
    node: ClangAstNode,
    blocks: list[LlvmIrBlock],
    *,
    allow_ordinal_fallback: bool = True,
) -> LlvmIrBlock | None:
    function_blocks = [block for block in blocks if block.function_name == node.function_name]
    if not function_blocks:
        return None

    if node.line_no > 0:
        same_line = [block for block in function_blocks if block.line_no == node.line_no and block.line_no > 0]
        if same_line:
            return sorted(same_line, key=lambda item: abs(item.column_no - node.column_no))[0]

    if allow_ordinal_fallback:
        ordinal = int(node.metadata.get("ast_ordinal", 0))
        if ordinal > 0:
            for block in function_blocks:
                if block.branch_ordinal == ordinal:
                    return block
        return function_blocks[0]
    return None


def _join_backend_tags(*parts: str) -> str:
    return "+".join([part for part in parts if part])


def _walk_ast(
    node: Any,
    summary: ClangAstSummary,
    *,
    current_function: str,
    branch_counters: dict[str, int],
    source_line_count: int | None = None,
) -> None:
    if not isinstance(node, dict):
        return

    kind = str(node.get("kind", ""))
    node_function = current_function
    if kind == "FunctionDecl" and isinstance(node.get("name"), str):
        node_function = str(node.get("name"))

    location = _extract_location(node)
    if location is not None:
        line_no, column_no = location
        in_main_source = source_line_count is None or 1 <= line_no <= source_line_count
        if not in_main_source:
            pass
        elif kind == "VarDecl" and node_function != "global":
            summary.declarations.append(
                ClangAstNode(
                    kind=kind,
                    function_name=node_function,
                    line_no=line_no,
                    column_no=column_no,
                    detail=str(node.get("name", "")),
                    metadata={
                        "storage_class": str(node.get("storageClass", "")),
                    },
                )
            )
        elif kind == "BinaryOperator" and str(node.get("opcode", "")) == "+":
            summary.additions.append(
                ClangAstNode(
                    kind=kind,
                    function_name=node_function,
                    line_no=line_no,
                    column_no=column_no,
                    metadata={
                        "opcode": "+",
                    },
                )
            )
        elif kind == "IntegerLiteral":
            summary.literals.append(
                ClangAstNode(
                    kind=kind,
                    function_name=node_function,
                    line_no=line_no,
                    column_no=column_no,
                    detail=str(node.get("value", "")),
                    metadata={
                        "value": str(node.get("value", "")),
                    },
                )
            )
        elif kind == "IfStmt" and node_function != "global":
            branch_counters[node_function] += 1
            summary.branches.append(
                ClangAstNode(
                    kind=kind,
                    function_name=node_function,
                    line_no=line_no,
                    column_no=column_no,
                    metadata={"ast_ordinal": branch_counters[node_function]},
                )
            )

    for child in node.get("inner", []):
        _walk_ast(
            child,
            summary,
            current_function=node_function,
            branch_counters=branch_counters,
            source_line_count=source_line_count,
        )


def _extract_location(node: dict[str, Any]) -> tuple[int, int] | None:
    location = _normalize_location(node.get("loc"))
    if location is not None:
        return location

    range_payload = node.get("range")
    if isinstance(range_payload, dict):
        location = _normalize_location(range_payload.get("begin"))
        if location is not None:
            return location
    return None


def _normalize_location(payload: Any) -> tuple[int, int] | None:
    if not isinstance(payload, dict):
        return None

    if "line" in payload and "col" in payload:
        try:
            return int(payload["line"]), int(payload["col"])
        except (TypeError, ValueError):
            return None

    for key in ("spellingLoc", "expansionLoc"):
        nested = payload.get(key)
        location = _normalize_location(nested)
        if location is not None:
            return location
    return None
