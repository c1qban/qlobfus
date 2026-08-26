from __future__ import annotations

from typing import Any

from scripts.common import load_structured_file, resolve_path


def load_manifest(manifest_path: str) -> dict[str, Any]:
    manifest = load_structured_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must decode to an object: {manifest_path}")
    return manifest


def find_manifest_sample(manifest: dict[str, Any], sample_id: str) -> dict[str, Any]:
    for sample in manifest.get("samples", []):
        if isinstance(sample, dict) and str(sample.get("sample_id")) == sample_id:
            return sample
    raise KeyError(f"Sample not found in manifest: {sample_id}")


def extract_manifest_test_cases(sample: dict[str, Any]) -> list[dict[str, Any]]:
    harness = sample.get("harness", {})
    inline_cases = harness.get("test_cases", [])
    if inline_cases:
        return list(inline_cases)

    harness_path = harness.get("path")
    if not harness_path:
        return []

    payload = load_structured_file(resolve_path(str(harness_path)))
    if isinstance(payload, dict):
        raw_cases = payload.get("test_cases", [])
    elif isinstance(payload, list):
        raw_cases = payload
    else:
        raise ValueError(f"Harness payload must decode to an object or list: {harness_path}")

    if not isinstance(raw_cases, list):
        raise ValueError(f"Harness test_cases must be a list: {harness_path}")

    return [dict(case) for case in raw_cases]

