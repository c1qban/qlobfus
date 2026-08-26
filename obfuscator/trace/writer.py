from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.common import ensure_dir


def append_trace_event(trace_path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(trace_path.parent)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
