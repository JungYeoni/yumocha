"""Input provenance and output inventory for provisional #81 artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(
    path: str | Path,
    *,
    role: str,
    schema: Sequence[str] | Mapping[str, Any],
    unit: str,
    **metadata: Any,
) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"계보 대상 파일이 없습니다: {target}")
    description: dict[str, Any] = {
        "role": role,
        "path": str(target.resolve()),
        "filename": target.name,
        "size": target.stat().st_size,
        "sha256": file_sha256(target),
        "schema": list(schema) if not isinstance(schema, Mapping) else dict(schema),
        "unit": unit,
    }
    description.update(metadata)
    return description


def build_manifest(
    *,
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    pipeline: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a manifest with separate input lineage and generated outputs."""

    if not inputs:
        raise ValueError("manifest inputs는 비어 있을 수 없습니다.")
    input_paths = [str(item.get("path", "")) for item in inputs]
    output_paths = [str(item.get("path", "")) for item in outputs]
    if len(input_paths) != len(set(input_paths)):
        raise ValueError("manifest inputs 경로가 중복됩니다.")
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("manifest outputs 경로가 중복됩니다.")
    return {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": dict(pipeline),
        "inputs": [dict(item) for item in inputs],
        "outputs": [dict(item) for item in outputs],
    }


def save_manifest(manifest: Mapping[str, Any], out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
