from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from flax.core import freeze

from singularity.qwen3_teacher import Qwen3TeacherConfig
from singularity.weight_mapping import MappingEntry, QwenCheckpointReader, teacher_qwen_mappings


@dataclass(frozen=True)
class ParityMetrics:
    max_abs: float
    mean_abs: float
    rmse: float
    relative_l2: float
    cosine_similarity: float

    def passes(self, *, max_abs_tolerance: float, relative_l2_tolerance: float) -> bool:
        return self.max_abs <= max_abs_tolerance and self.relative_l2 <= relative_l2_tolerance


def layer_mapping_entries(config: Qwen3TeacherConfig, layer_index: int) -> list[MappingEntry]:
    if not 0 <= layer_index < config.num_layers:
        raise ValueError(f"layer_index must be in [0, {config.num_layers}), got {layer_index}")
    prefix = f"layers_{layer_index}/"
    return [entry for entry in teacher_qwen_mappings(config) if entry.target.startswith(prefix)]


def required_layer_shards(
    weight_map: Mapping[str, str],
    config: Qwen3TeacherConfig,
    layer_index: int,
) -> tuple[str, ...]:
    entries = layer_mapping_entries(config, layer_index)
    missing = [entry.source for entry in entries if entry.source not in weight_map]
    if missing:
        raise KeyError(f"checkpoint index is missing layer tensors: {missing}")
    return tuple(sorted({weight_map[entry.source] for entry in entries}))


def load_layer_arrays(
    reader: QwenCheckpointReader,
    config: Qwen3TeacherConfig,
    layer_index: int,
) -> dict[str, np.ndarray]:
    """Load one decoder layer once, casting checkpoint BF16 values to FP32."""
    return {
        entry.source: np.asarray(reader.read(entry.source), dtype=np.float32)
        for entry in layer_mapping_entries(config, layer_index)
    }


def load_mixer_arrays(
    reader: QwenCheckpointReader,
    config: Qwen3TeacherConfig,
    layer_index: int,
) -> dict[str, np.ndarray]:
    """Load only input norm and attention tensors for mixer experiments."""
    prefix = f"model.layers.{layer_index}."
    entries = [
        entry
        for entry in layer_mapping_entries(config, layer_index)
        if entry.source == f"{prefix}input_layernorm.weight"
        or entry.source.startswith(f"{prefix}self_attn.")
    ]
    return {
        entry.source: np.asarray(reader.read(entry.source), dtype=np.float32)
        for entry in entries
    }


def jax_layer_params(
    arrays: Mapping[str, np.ndarray],
    config: Qwen3TeacherConfig,
    layer_index: int,
) -> Any:
    prefix = f"layers_{layer_index}/"
    flat: dict[tuple[str, ...], jax.Array] = {}
    for entry in layer_mapping_entries(config, layer_index):
        value = arrays[entry.source]
        if entry.transform == "transpose":
            value = value.T
        elif entry.transform != "identity":
            raise ValueError(f"unsupported transform: {entry.transform}")
        relative_target = entry.target.removeprefix(prefix)
        flat[tuple(relative_target.split("/"))] = jnp.asarray(value, dtype=jnp.float32)
    return freeze(traverse_util.unflatten_dict(flat))


def jax_attention_params(
    arrays: Mapping[str, np.ndarray],
    config: Qwen3TeacherConfig,
    layer_index: int,
) -> Any:
    """Materialize only one attention module, avoiding unused MLP device arrays."""
    layer_prefix = f"layers_{layer_index}/self_attn/"
    flat: dict[tuple[str, ...], jax.Array] = {}
    for entry in layer_mapping_entries(config, layer_index):
        if not entry.target.startswith(layer_prefix):
            continue
        value = arrays[entry.source]
        if entry.transform == "transpose":
            value = value.T
        elif entry.transform != "identity":
            raise ValueError(f"unsupported transform: {entry.transform}")
        relative_target = entry.target.removeprefix(layer_prefix)
        flat[tuple(relative_target.split("/"))] = jnp.asarray(
            value, dtype=jnp.float32
        )
    return freeze(traverse_util.unflatten_dict(flat))


def torch_layer_state(
    arrays: Mapping[str, np.ndarray],
    config: Qwen3TeacherConfig,
    layer_index: int,
) -> dict[str, Any]:
    import torch

    prefix = f"model.layers.{layer_index}."
    return {
        entry.source.removeprefix(prefix): torch.from_numpy(arrays[entry.source])
        for entry in layer_mapping_entries(config, layer_index)
    }


def parity_metrics(reference: np.ndarray, candidate: np.ndarray) -> ParityMetrics:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"parity shapes differ: {reference.shape} != {candidate.shape}")
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference.ravel())
    candidate_norm = np.linalg.norm(candidate.ravel())
    denominator = max(reference_norm, np.finfo(np.float64).tiny)
    cosine_denominator = max(reference_norm * candidate_norm, np.finfo(np.float64).tiny)
    return ParityMetrics(
        max_abs=float(np.max(np.abs(difference))),
        mean_abs=float(np.mean(np.abs(difference))),
        rmse=float(np.sqrt(np.mean(np.square(difference)))),
        relative_l2=float(np.linalg.norm(difference.ravel()) / denominator),
        cosine_similarity=float(np.dot(reference.ravel(), candidate.ravel()) / cosine_denominator),
    )


def ensure_layer_checkpoint(
    output_dir: str | Path,
    weight_map: Mapping[str, str],
    config: Qwen3TeacherConfig,
    layer_index: int,
    *,
    repo_id: str,
    revision: str,
) -> tuple[Path, tuple[str, ...]]:
    """Download only config/index and shards containing the requested layer."""
    from huggingface_hub import hf_hub_download

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shards = required_layer_shards(weight_map, config, layer_index)
    for filename in ("config.json", "model.safetensors.index.json", *shards):
        hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            local_dir=output,
        )
    return output, shards
