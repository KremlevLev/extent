from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import serialization


CHECKPOINT_FORMAT_VERSION = 1


def deterministic_batch_indices(
    step: int,
    batch_size: int,
    example_count: int,
    seed: int,
) -> np.ndarray:
    """Return an exactly resumable shuffled stream without storing a cursor."""
    if step < 0 or seed < 0:
        raise ValueError("step and data seed must be non-negative")
    if batch_size < 1 or example_count < 1:
        raise ValueError("batch size and example count must be positive")
    positions = np.arange(
        step * batch_size, (step + 1) * batch_size, dtype=np.int64
    )
    result = np.empty(batch_size, dtype=np.int64)
    for epoch in np.unique(positions // example_count):
        mask = positions // example_count == epoch
        permutation = np.random.default_rng(
            np.random.SeedSequence([seed, int(epoch)])
        ).permutation(example_count)
        result[mask] = permutation[positions[mask] % example_count]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def save_offline_checkpoint(
    directory: str | Path,
    params,
    opt_state,
    *,
    step: int,
    compatibility: dict,
) -> dict:
    """Atomically save mixer parameters, Lion state, and resume metadata."""
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "training_state.msgpack"
    temporary_checkpoint = root / "training_state.tmp.msgpack"
    payload = serialization.to_bytes(
        {
            "params": jax.device_get(params),
            "opt_state": jax.device_get(opt_state),
        }
    )
    temporary_checkpoint.write_bytes(payload)
    temporary_checkpoint.replace(checkpoint_path)
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "compatibility": compatibility,
    }
    metadata_path = root / "checkpoint.json"
    temporary_metadata = root / "checkpoint.tmp.json"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return metadata


def restore_offline_checkpoint(
    directory: str | Path,
    params_template,
    opt_state_template,
    *,
    expected_compatibility: dict,
):
    """Restore only when data, model, optimizer, and schedule contracts match."""
    root = Path(directory)
    metadata_path = root / "checkpoint.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"offline checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported offline checkpoint format version")
    if _canonical_json(metadata.get("compatibility")) != _canonical_json(
        expected_compatibility
    ):
        raise ValueError("offline checkpoint compatibility contract mismatch")
    checkpoint_path = root / metadata["checkpoint_file"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"offline checkpoint payload is missing: {checkpoint_path}")
    if int(checkpoint_path.stat().st_size) != int(metadata["checkpoint_bytes"]):
        raise ValueError("offline checkpoint byte-size mismatch")
    if _sha256(checkpoint_path) != metadata["checkpoint_sha256"]:
        raise ValueError("offline checkpoint SHA-256 mismatch")
    restored = serialization.from_bytes(
        {"params": params_template, "opt_state": opt_state_template},
        checkpoint_path.read_bytes(),
    )
    return restored["params"], restored["opt_state"], int(metadata["step"]), metadata
