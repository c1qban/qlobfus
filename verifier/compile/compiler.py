from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from verifier.models import CompileResult


DEFAULT_COMPILERS = [
    "clang",
    "clang.exe",
    r"C:\Program Files\LLVM\bin\clang.exe",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\bin\clang.exe",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clang.exe",
    "gcc",
    "gcc.exe",
    "cc",
]


def choose_compiler(preferred: str | None = None) -> str | None:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(DEFAULT_COMPILERS)

    for candidate in candidates:
        path_candidate = Path(candidate)
        if path_candidate.is_absolute() and path_candidate.exists():
            return str(path_candidate)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def compile_c_source(
    source_path: Path,
    output_path: Path,
    compiler: str | None = None,
    compiler_flags: list[str] | None = None,
    timeout_sec: float = 10.0,
    workdir: Path | None = None,
) -> CompileResult:
    compiler_path = choose_compiler(compiler)
    if compiler_path is None:
        return CompileResult(
            succeeded=False,
            command="",
            returncode=127,
            stderr="No supported C compiler found. Tried clang/gcc fallback chain.",
            output_path=str(output_path),
            notes=["Install clang or gcc, or pass an explicit compiler path."],
        )

    flags = list(compiler_flags or [])
    command = [compiler_path, str(source_path), "-o", str(output_path), *flags]

    try:
        completed = subprocess.run(
            command,
            cwd=str(workdir) if workdir else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CompileResult(
            succeeded=False,
            command=" ".join(command),
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + "\nCompilation timed out.",
            output_path=str(output_path),
            notes=[f"Compilation exceeded timeout of {timeout_sec} seconds."],
        )

    output_exists = output_path.exists()
    notes: list[str] = []
    if completed.returncode == 0 and not output_exists:
        notes.append(
            f"Compiler returned success but the output artifact was not visible at the expected path "
            f"(path_length={len(str(output_path))})."
        )

    return CompileResult(
        succeeded=completed.returncode == 0 and output_exists,
        command=" ".join(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_path=str(output_path),
        notes=notes,
    )
