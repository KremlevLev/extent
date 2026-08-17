from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
) -> ActivationWindowLayout:
    if min(calibration_windows, training_windows, evaluation_windows) < 1:
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
