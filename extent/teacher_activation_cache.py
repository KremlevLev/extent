from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ActivationWindowLayout:
    calibration: slice
    training: slice
    evaluation: slice
    total_windows: int


def activation_window_layout(
    calibration_windows: int,
    training_windows: int,
    evaluation_windows: int,
    *,
    allow_evaluation_only: bool = False,
) -> ActivationWindowLayout:
    if allow_evaluation_only:
        if calibration_windows != 0 or training_windows != 0:
            raise ValueError(
                "evaluation-only cache requires zero calibration and training windows"
            )
        if evaluation_windows < 1:
            raise ValueError("evaluation-only cache requires evaluation windows")
    elif min(calibration_windows, training_windows, evaluation_windows) < 1:
        raise ValueError("activation-cache window counts must be positive")
    training_start = calibration_windows
    evaluation_start = training_start + training_windows
    total = evaluation_start + evaluation_windows
    return ActivationWindowLayout(
        calibration=slice(0, training_start),
        training=slice(training_start, evaluation_start),
        evaluation=slice(evaluation_start, total),
        total_windows=total,
    )


def run_host_microbatches(
    runner: Callable[[object, jax.Array], jax.Array],
    params: object,
    inputs: np.ndarray | jax.Array,
    microbatch_windows: int,
    *,
    input_dtype: jnp.dtype,
) -> np.ndarray:
    """Run a frozen layer in bounded batches and return FP32 host activations."""
    if microbatch_windows < 1:
        raise ValueError("microbatch_windows must be positive")
    windows = int(inputs.shape[0])
    output = np.empty(tuple(inputs.shape), dtype=np.float32)
    for start in range(0, windows, microbatch_windows):
        stop = min(start + microbatch_windows, windows)
        batch = jnp.asarray(inputs[start:stop], dtype=input_dtype)
        value = runner(params, batch)
        output[start:stop] = np.asarray(value, dtype=np.float32)
    return output


def run_host_data_parallel(
    runner: Callable[[object, jax.Array], jax.Array],
    params: object,
    inputs: np.ndarray | jax.Array,
    per_device_windows: int,
    device_count: int,
    *,
    input_dtype: jnp.dtype,
) -> np.ndarray:
    """Feed a pmapped frozen layer, padding only the final host batch."""
    if per_device_windows < 1 or device_count < 1:
        raise ValueError("per_device_windows and device_count must be positive")
    windows = int(inputs.shape[0])
    global_batch = per_device_windows * device_count
    output = np.empty(tuple(inputs.shape), dtype=np.float32)
    trailing_shape = tuple(inputs.shape[1:])
    for start in range(0, windows, global_batch):
        stop = min(start + global_batch, windows)
        valid = stop - start
        host = np.zeros((global_batch, *trailing_shape), dtype=np.dtype(input_dtype))
        host[:valid] = np.asarray(inputs[start:stop], dtype=np.dtype(input_dtype))
        sharded = jnp.asarray(
            host.reshape(device_count, per_device_windows, *trailing_shape),
            dtype=input_dtype,
        )
        value = np.asarray(runner(params, sharded), dtype=np.float32).reshape(
            global_batch, *trailing_shape
        )
        output[start:stop] = value[:valid]
    return output


def checkpoint_shard_last_use(
    weight_map: Mapping[str, str], target_layer: int
) -> dict[str, int]:
    """Return the last required layer for each shard; -1 means embeddings only."""
    if target_layer < 0:
        raise ValueError("target_layer must be non-negative")
    embedding = "model.embed_tokens.weight"
    if embedding not in weight_map:
        raise KeyError(f"checkpoint index is missing {embedding}")
    result = {weight_map[embedding]: -1}
    for layer in range(target_layer + 1):
        prefix = f"model.layers.{layer}."
        layer_shards = {
            shard for name, shard in weight_map.items() if name.startswith(prefix)
        }
        if not layer_shards:
            raise KeyError(f"checkpoint index has no tensors for layer {layer}")
        for shard in layer_shards:
            result[shard] = max(result.get(shard, -1), layer)
    return result


def atomic_save_array(path: str | Path, value: np.ndarray, dtype: np.dtype) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp.npy")
    np.save(temporary, np.asarray(value, dtype=dtype), allow_pickle=False)
    temporary.replace(output)
    return output


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_artifact(path: str | Path) -> dict:
    file = Path(path)
    value = np.load(file, mmap_mode="r", allow_pickle=False)
    return {
        "path": str(file.resolve()),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": int(file.stat().st_size),
        "sha256": file_sha256(file),
    }


def load_activation_cache(
    manifest_path: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    verify_hashes: bool = True,
) -> tuple[dict, dict[str, np.ndarray], dict[str, Path]]:
    """Load and validate a portable activation cache from manifest + NPY files."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not manifest.get("passed"):
        raise ValueError("activation-cache manifest is not marked passed")
    records = manifest.get("artifacts", {})
    required = ("token_ids", "residual_input", "normalized_input", "attention_target")
    missing = [name for name in required if name not in records]
    if missing:
        raise KeyError(f"activation-cache manifest is missing artifacts: {missing}")
    search_dir = Path(artifact_dir) if artifact_dir else None
    arrays: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for name in required:
        record = records[name]
        recorded = Path(record["path"])
        candidates = []
        if search_dir is not None:
            candidates.append(search_dir / recorded.name)
        candidates.extend((manifest_file.parent / recorded.name, recorded))
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"cannot locate {name}; searched {[str(value) for value in candidates]}"
            )
        if verify_hashes and file_sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for activation artifact {name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != list(record["shape"]):
            raise ValueError(
                f"shape mismatch for {name}: {list(value.shape)} != {record['shape']}"
            )
        if str(value.dtype) != record["dtype"]:
            raise ValueError(
                f"dtype mismatch for {name}: {value.dtype} != {record['dtype']}"
            )
        arrays[name] = value
        paths[name] = path.resolve()
    return manifest, arrays, paths


def validate_external_evaluation_cache(
    training_manifest: Mapping,
    evaluation_manifest: Mapping,
    *,
    required_token_offset: int | None = None,
    required_evaluation_windows: int | None = None,
    required_dataset_split: str | None = None,
    allow_cross_split: bool = False,
    allow_sequence_length_mismatch: bool = False,
) -> slice:
    """Validate a disjoint evaluation-only cache against a training cache."""
    shared = ["source", "dataset", "target_layer"]
    if not allow_sequence_length_mismatch:
        shared.append("sequence_length")
    mismatches = {
        key: (training_manifest.get(key), evaluation_manifest.get(key))
        for key in shared
        if training_manifest.get(key) != evaluation_manifest.get(key)
    }
    if mismatches:
        raise ValueError(f"training/evaluation cache mismatch: {mismatches}")
    if not evaluation_manifest.get("evaluation_only"):
        raise ValueError("external evaluation cache must be marked evaluation_only")
    training_split = str(training_manifest.get("dataset_split", "train"))
    evaluation_split = str(evaluation_manifest.get("dataset_split", "train"))
    if required_dataset_split is not None and evaluation_split != required_dataset_split:
        raise ValueError("external evaluation dataset split differs from the frozen protocol")
    if evaluation_split != training_split and not allow_cross_split:
        raise ValueError("external evaluation cache uses an unauthorized dataset split")
    layout = evaluation_manifest["window_layout"]
    calibration = slice(*layout["calibration"])
    training = slice(*layout["training"])
    evaluation = slice(*layout["evaluation"])
    if calibration.start != calibration.stop or training.start != training.stop:
        raise ValueError("external evaluation cache must not contain train/calibration windows")
    evaluation_windows = evaluation.stop - evaluation.start
    if evaluation_windows < 1:
        raise ValueError("external evaluation cache has no evaluation windows")
    if (
        required_evaluation_windows is not None
        and evaluation_windows != required_evaluation_windows
    ):
        raise ValueError(
            "external evaluation window count differs from the frozen protocol"
        )
    offset = int(evaluation_manifest.get("token_offset", -1))
    if required_token_offset is not None and offset != required_token_offset:
        raise ValueError("external evaluation token offset differs from the frozen protocol")
    training_range = training_manifest.get("token_range")
    evaluation_range = evaluation_manifest.get("token_range")
    if not training_range or not evaluation_range:
        raise ValueError("cache manifests must record token_range provenance")
    evaluation_start, evaluation_stop = map(int, evaluation_range)
    expected_tokens = evaluation_windows * int(evaluation_manifest["sequence_length"])
    if evaluation_start != offset or evaluation_stop - evaluation_start != expected_tokens:
        raise ValueError("external evaluation token_range is inconsistent with its layout")
    disjoint = evaluation_split != training_split or (
        evaluation_stop <= int(training_range[0])
        or int(training_range[1]) <= evaluation_start
    )
    if not disjoint:
        raise ValueError("external evaluation token range overlaps the training cache")
    return evaluation
