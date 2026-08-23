from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import serialization


ENDPOINT_CHECKPOINT_FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def save_endpoint_checkpoint(
    directory: str | Path,
    endpoints: Mapping,
    *,
    compatibility: dict,
) -> dict:
    """Atomically persist selected trained endpoints without optimizer state."""
    if not endpoints:
        raise ValueError("endpoint checkpoint cannot be empty")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / "endpoint_params.msgpack"
    temporary_payload = root / "endpoint_params.tmp.msgpack"
    payload = serialization.to_bytes(jax.device_get(dict(endpoints)))
    temporary_payload.write_bytes(payload)
    temporary_payload.replace(payload_path)
    metadata = {
        "format_version": ENDPOINT_CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": "mamba_endpoint_parameters",
        "checkpoint_file": payload_path.name,
        "checkpoint_bytes": int(payload_path.stat().st_size),
        "checkpoint_sha256": _sha256(payload_path),
        "compatibility": compatibility,
    }
    metadata_path = root / "checkpoint.json"
    temporary_metadata = root / "checkpoint.tmp.json"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return metadata


def restore_endpoint_checkpoint(
    directory: str | Path,
    *,
    expected_compatibility: dict,
) -> tuple[dict, dict]:
    """Verify and restore endpoint parameters under an exact contract."""
    root = Path(directory)
    metadata_path = root / "checkpoint.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"endpoint checkpoint metadata is missing: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != ENDPOINT_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported endpoint checkpoint format version")
    if metadata.get("checkpoint_kind") != "mamba_endpoint_parameters":
        raise ValueError("checkpoint is not a Mamba endpoint bundle")
    if _canonical_json(metadata.get("compatibility")) != _canonical_json(
        expected_compatibility
    ):
        raise ValueError("endpoint checkpoint compatibility contract mismatch")
    payload_path = root / metadata["checkpoint_file"]
    if not payload_path.exists():
        raise FileNotFoundError(
            f"endpoint checkpoint payload is missing: {payload_path}"
        )
    if int(payload_path.stat().st_size) != int(metadata["checkpoint_bytes"]):
        raise ValueError("endpoint checkpoint byte-size mismatch")
    if _sha256(payload_path) != metadata["checkpoint_sha256"]:
        raise ValueError("endpoint checkpoint SHA-256 mismatch")
    restored = serialization.msgpack_restore(payload_path.read_bytes())
    endpoints = jax.tree.map(
        lambda value: jnp.asarray(value)
        if hasattr(value, "dtype") and hasattr(value, "shape")
        else value,
        restored,
    )
    if not isinstance(endpoints, dict) or not endpoints:
        raise ValueError("endpoint checkpoint restored an invalid payload")
    return endpoints, metadata
