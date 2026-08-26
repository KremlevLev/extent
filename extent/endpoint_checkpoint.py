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
    retained_arms = sorted(
        {
            str(arm)
            for seed_endpoints in endpoints.values()
            for arm in seed_endpoints
        }
    )
    metadata = {
        "format_version": ENDPOINT_CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": "mamba_endpoint_parameters",
        "checkpoint_file": payload_path.name,
        "checkpoint_bytes": int(payload_path.stat().st_size),
        "checkpoint_sha256": _sha256(payload_path),
        "retained_arms": retained_arms,
        "compatibility": compatibility,
    }
    metadata_path = root / "checkpoint.json"
    temporary_metadata = root / "checkpoint.tmp.json"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return metadata


def endpoint_checkpoint_supports_arms(
    directory: str | Path, required_arms: set[str]
) -> bool:
    """Return whether a checkpoint can resume every requested evaluation arm."""
    metadata_path = Path(directory) / "checkpoint.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    retained = metadata.get("retained_arms")
    return retained is None or required_arms <= set(retained)


def compact_endpoint_checkpoint(
    directory: str | Path,
    *,
    retained_arms: set[str],
    expected_compatibility: dict,
) -> dict:
    """Atomically replace a completed endpoint bundle with selected arms only.

    The new payload is installed under a different filename before metadata is
    switched. An interruption therefore leaves either the old complete bundle
    or the new compact bundle addressable by valid metadata.
    """
    if not retained_arms:
        raise ValueError("at least one endpoint arm must be retained")
    root = Path(directory)
    metadata_path = root / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != ENDPOINT_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported endpoint checkpoint format version")
    if metadata.get("checkpoint_kind") != "mamba_endpoint_parameters":
        raise ValueError("checkpoint is not a Mamba endpoint bundle")
    if _canonical_json(metadata.get("compatibility")) != _canonical_json(
        expected_compatibility
    ):
        raise ValueError("endpoint checkpoint compatibility contract mismatch")
    old_payload = root / metadata["checkpoint_file"]
    if int(old_payload.stat().st_size) != int(metadata["checkpoint_bytes"]):
        raise ValueError("endpoint checkpoint byte-size mismatch")
    if _sha256(old_payload) != metadata["checkpoint_sha256"]:
        raise ValueError("endpoint checkpoint SHA-256 mismatch")
    requested = sorted(str(arm) for arm in retained_arms)
    if metadata.get("retained_arms") == requested:
        return metadata

    restored = serialization.msgpack_restore(old_payload.read_bytes())
    if not isinstance(restored, dict) or not restored:
        raise ValueError("endpoint checkpoint restored an invalid payload")
    selected = {}
    for seed, seed_endpoints in restored.items():
        missing = retained_arms - set(seed_endpoints)
        if missing:
            raise ValueError(f"endpoint seed {seed} is missing arms {sorted(missing)}")
        selected[str(seed)] = {
            arm: seed_endpoints[arm] for arm in requested
        }

    payload_path = root / "endpoint_params.selected.msgpack"
    temporary_payload = root / "endpoint_params.selected.tmp.msgpack"
    temporary_payload.write_bytes(serialization.to_bytes(selected))
    temporary_payload.replace(payload_path)
    compact_metadata = {
        **metadata,
        "checkpoint_file": payload_path.name,
        "checkpoint_bytes": int(payload_path.stat().st_size),
        "checkpoint_sha256": _sha256(payload_path),
        "retained_arms": requested,
        "compacted": True,
    }
    temporary_metadata = root / "checkpoint.compact.tmp.json"
    temporary_metadata.write_text(
        json.dumps(compact_metadata, indent=2), encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    if old_payload.resolve() != payload_path.resolve() and old_payload.exists():
        old_payload.unlink()
    return compact_metadata


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
