"""Qwen safetensors -> hybrid initialization scaffolding.

The functions operate on one array/layer at a time so a 14B checkpoint never has
to be materialized twice in host RAM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import jax
import numpy as np
from flax import traverse_util
from flax.core import FrozenDict, freeze

from singularity.config import HybridConfig


class QwenShapeConfig(Protocol):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


@dataclass(frozen=True)
class MappingEntry:
    source: str
    target: str
    transform: str = "transpose"


@dataclass(frozen=True)
class MappingValidationReport:
    tensor_count: int
    parameter_count: int


def direct_qwen_mappings(config: QwenShapeConfig) -> list[MappingEntry]:
    entries = [
        MappingEntry("model.embed_tokens.weight", "embed_tokens/embedding", "identity"),
        MappingEntry("model.norm.weight", "norm/scale", "identity"),
        MappingEntry("lm_head.weight", "lm_head/kernel", "transpose"),
    ]
    for layer in range(config.num_layers):
        source = f"model.layers.{layer}"
        target = f"layers_{layer}"
        entries.extend(
            [
                MappingEntry(f"{source}.input_layernorm.weight", f"{target}/input_layernorm/scale", "identity"),
                MappingEntry(f"{source}.post_attention_layernorm.weight", f"{target}/post_attention_layernorm/scale", "identity"),
                MappingEntry(f"{source}.mlp.gate_proj.weight", f"{target}/mlp/gate_proj/kernel"),
                MappingEntry(f"{source}.mlp.up_proj.weight", f"{target}/mlp/up_proj/kernel"),
                MappingEntry(f"{source}.mlp.down_proj.weight", f"{target}/mlp/down_proj/kernel"),
            ]
        )
    return entries


def teacher_qwen_mappings(config: QwenShapeConfig) -> list[MappingEntry]:
    """Map every tensor in the pinned dense Qwen3 checkpoint to the JAX teacher."""
    entries = direct_qwen_mappings(config)
    for layer in range(config.num_layers):
        source = f"model.layers.{layer}.self_attn"
        target = f"layers_{layer}/self_attn"
        entries.extend(
            [
                MappingEntry(f"{source}.q_proj.weight", f"{target}/q_proj/kernel"),
                MappingEntry(f"{source}.k_proj.weight", f"{target}/k_proj/kernel"),
                MappingEntry(f"{source}.v_proj.weight", f"{target}/v_proj/kernel"),
                MappingEntry(f"{source}.o_proj.weight", f"{target}/o_proj/kernel"),
                MappingEntry(f"{source}.q_norm.weight", f"{target}/q_norm/scale", "identity"),
                MappingEntry(f"{source}.k_norm.weight", f"{target}/k_norm/scale", "identity"),
            ]
        )
    return entries


def truncated_svd(
    weight: np.ndarray,
    rank: int,
    *,
    exact_threshold: int = 2_048,
    oversample: int = 8,
    power_iterations: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor [input, output] into low-rank factors without a huge full SVD."""
    matrix = weight.astype(np.float32)
    if max(matrix.shape) <= exact_threshold:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    else:
        width = min(rank + oversample, min(matrix.shape))
        random = np.random.default_rng(seed).standard_normal((matrix.shape[1], width), dtype=np.float32)
        q, _ = np.linalg.qr(matrix @ random, mode="reduced")
        for _ in range(power_iterations):
            q_right, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
            q, _ = np.linalg.qr(matrix @ q_right, mode="reduced")
        small = q.T @ matrix
        u_small, singular, vh = np.linalg.svd(small, full_matrices=False)
        u = q @ u_small
    rank = min(rank, singular.size)
    root = np.sqrt(singular[:rank])
    return (u[:, :rank] * root).astype(weight.dtype), (root[:, None] * vh[:rank]).astype(weight.dtype)


class QwenCheckpointReader:
    """Lazy safetensors reader: only one requested tensor is resident at a time."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        with (self.model_dir / "model.safetensors.index.json").open("r", encoding="utf-8") as stream:
            self.weight_map: dict[str, str] = json.load(stream)["weight_map"]

    def read(self, name: str) -> np.ndarray:
        from safetensors import safe_open

        if name not in self.weight_map:
            raise KeyError(f"tensor not present in checkpoint: {name}")
        with safe_open(str(self.model_dir / self.weight_map[name]), framework="np", device="cpu") as shard:
            return shard.get_tensor(name)

    def read_rows(self, name: str, rows: np.ndarray) -> np.ndarray:
        """Gather selected checkpoint rows without materializing a float32 copy."""
        import torch
        from safetensors import safe_open

        if name not in self.weight_map:
            raise KeyError(f"tensor not present in checkpoint: {name}")
        indices = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        with safe_open(
            str(self.model_dir / self.weight_map[name]), framework="pt", device="cpu"
        ) as shard:
            return shard.get_tensor(name)[indices].float().numpy()

    def shape(self, name: str) -> tuple[int, ...]:
        """Read only a tensor's safetensors header/slice metadata."""
        from safetensors import safe_open

        if name not in self.weight_map:
            raise KeyError(f"tensor not present in checkpoint: {name}")
        with safe_open(str(self.model_dir / self.weight_map[name]), framework="np", device="cpu") as shard:
            return tuple(shard.get_slice(name).get_shape())

    def qkvo(self, layer: int) -> dict[str, np.ndarray]:
        prefix = f"model.layers.{layer}.self_attn"
        return {name: self.read(f"{prefix}.{name}_proj.weight").T for name in ("q", "k", "v", "o")}


def fit_matrix(source: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Deterministically tile/crop a projection for transplant ablations."""
    row_repeats = (target_shape[0] + source.shape[0] - 1) // source.shape[0]
    col_repeats = (target_shape[1] + source.shape[1] - 1) // source.shape[1]
    tiled = np.tile(source, (row_repeats, col_repeats))
    scale = np.sqrt(source.shape[1] / target_shape[1])
    return (tiled[: target_shape[0], : target_shape[1]] * scale).astype(source.dtype)


def mamba_transplant_in_projection(
    q_kernel: np.ndarray,
    k_kernel: np.ndarray,
    v_kernel: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Experimental Q/K/V seed for Mamba in_proj; suitable as an ablation baseline.

    This is intentionally labelled a heuristic, not a pretrained-equivalence map.
    It repeats V for x/z and interleaves K/Q over the remaining controller slots.
    """
    width = target_shape[1]
    first = width // 2
    xz = fit_matrix(np.concatenate((v_kernel, q_kernel), axis=1), (target_shape[0], first))
    controllers = fit_matrix(np.concatenate((k_kernel, q_kernel, v_kernel), axis=1), (target_shape[0], width - first))
    return np.concatenate((xz, controllers), axis=1)


def expected_qwen_shape(entry: MappingEntry, config: QwenShapeConfig) -> tuple[int, ...]:
    """Infer direct-source shapes from the pinned dense Qwen3 architecture."""
    name = entry.source
    if name == "model.embed_tokens.weight" or name == "lm_head.weight":
        return (config.vocab_size, config.hidden_size)
    if name == "model.norm.weight" or name.endswith("layernorm.weight"):
        return (config.hidden_size,)
    if name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
        return (config.intermediate_size, config.hidden_size)
    if name.endswith("mlp.down_proj.weight"):
        return (config.hidden_size, config.intermediate_size)
    if name.endswith("self_attn.q_proj.weight"):
        return (config.num_attention_heads * config.head_dim, config.hidden_size)
    if name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
        return (config.num_key_value_heads * config.head_dim, config.hidden_size)
    if name.endswith("self_attn.o_proj.weight"):
        return (config.hidden_size, config.num_attention_heads * config.head_dim)
    if name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
        return (config.head_dim,)
    raise KeyError(f"no expected shape rule for {name}")


def _transformed_shape(shape: tuple[int, ...], transform: str) -> tuple[int, ...]:
    if transform == "identity":
        return shape
    if transform == "transpose" and len(shape) == 2:
        return shape[::-1]
    raise ValueError(f"unsupported transform {transform!r} for shape {shape}")


def validate_mapping_plan(
    config: QwenShapeConfig,
    abstract_params: Mapping[str, Any],
    weight_map: Mapping[str, str],
    entries: list[MappingEntry],
    *,
    require_complete_source: bool = False,
) -> MappingValidationReport:
    """Validate a mapping using only abstract target arrays and checkpoint metadata."""
    flat_targets = traverse_util.flatten_dict(abstract_params)
    parameter_count = 0
    seen_sources: set[str] = set()
    seen_targets: set[tuple[str, ...]] = set()
    for entry in entries:
        if entry.source in seen_sources:
            raise ValueError(f"source tensor mapped more than once: {entry.source}")
        if entry.source not in weight_map:
            raise KeyError(f"source tensor missing from index: {entry.source}")
        target_path = tuple(entry.target.split("/"))
        if target_path in seen_targets:
            raise ValueError(f"target tensor mapped more than once: {entry.target}")
        if target_path not in flat_targets:
            raise KeyError(f"target tensor missing from hybrid model: {entry.target}")
        source_shape = expected_qwen_shape(entry, config)
        expected_target = _transformed_shape(source_shape, entry.transform)
        actual_target = tuple(flat_targets[target_path].shape)
        if actual_target != expected_target:
            raise ValueError(
                f"shape mismatch for {entry.source} -> {entry.target}: "
                f"expected {expected_target}, target is {actual_target}"
            )
        parameter_count += int(np.prod(actual_target))
        seen_sources.add(entry.source)
        seen_targets.add(target_path)
    if require_complete_source:
        unmapped_sources = set(weight_map) - seen_sources
        unmapped_targets = set(flat_targets) - seen_targets
        if unmapped_sources or unmapped_targets:
            raise ValueError(
                f"mapping is incomplete: {len(unmapped_sources)} source and "
                f"{len(unmapped_targets)} target tensors remain"
            )
    return MappingValidationReport(len(entries), parameter_count)


def validate_direct_mapping_plan(
    config: HybridConfig,
    abstract_params: Mapping[str, Any],
    weight_map: Mapping[str, str],
) -> MappingValidationReport:
    """Validate all tensors preserved unchanged in the hybrid student."""
    return validate_mapping_plan(
        config, abstract_params, weight_map, direct_qwen_mappings(config)
    )


def validate_teacher_mapping_plan(
    config: QwenShapeConfig,
    abstract_params: Mapping[str, Any],
    weight_map: Mapping[str, str],
) -> MappingValidationReport:
    """Require a one-to-one mapping of the complete Qwen3 teacher checkpoint."""
    return validate_mapping_plan(
        config,
        abstract_params,
        weight_map,
        teacher_qwen_mappings(config),
        require_complete_source=True,
    )


def stream_direct_qwen_weights(
    params: Mapping[str, Any],
    reader: QwenCheckpointReader,
    config: HybridConfig,
    *,
    progress: Callable[[int, int, MappingEntry], None] | None = None,
) -> tuple[Mapping[str, Any], MappingValidationReport]:
    """Replace preserved tensors one at a time, retaining target device sharding."""
    report = validate_direct_mapping_plan(config, params, reader.weight_map)
    was_frozen = isinstance(params, FrozenDict)
    flat = dict(traverse_util.flatten_dict(params))
    entries = direct_qwen_mappings(config)
    for index, entry in enumerate(entries, start=1):
        target_path = tuple(entry.target.split("/"))
        target = flat[target_path]
        source = reader.read(entry.source)
        actual_source_shape = tuple(source.shape)
        expected_source_shape = expected_qwen_shape(entry, config)
        if actual_source_shape != expected_source_shape:
            raise ValueError(
                f"checkpoint tensor {entry.source} has shape {actual_source_shape}; "
                f"expected {expected_source_shape}"
            )
        value = source if entry.transform == "identity" else source.T
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(f"transformed {entry.source} does not fit {entry.target}")
        if isinstance(target, jax.Array):
            # Materialize each addressable shard from host memory directly. A
            # temporary full tensor must never land on one accelerator first.
            host_value = np.ascontiguousarray(value, dtype=np.dtype(target.dtype))
            value = jax.make_array_from_callback(
                target.shape,
                target.sharding,
                lambda index, host_value=host_value: host_value[index],
            )
        else:
            value = np.asarray(value, dtype=target.dtype)
        flat[target_path] = value
        del source
        if progress is not None:
            progress(index, len(entries), entry)
    result = traverse_util.unflatten_dict(flat)
    return (freeze(result) if was_frozen else result), report


def validate_local_direct_shapes(
    reader: QwenCheckpointReader,
    config: HybridConfig,
) -> None:
    """Check safetensors headers for every directly transferred tensor."""
    for entry in direct_qwen_mappings(config):
        actual = reader.shape(entry.source)
        expected = expected_qwen_shape(entry, config)
        if actual != expected:
            raise ValueError(
                f"checkpoint tensor {entry.source} has shape {actual}; expected {expected}"
            )
