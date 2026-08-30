"""Streaming full-model mixer materialization into existing global JAX shards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from flax import traverse_util
from flax.core import FrozenDict, freeze
import jax
import numpy as np

from extent.config import HybridConfig
from extent.mamba3_transplant import build_qwen3_to_mamba3_transplant_variants
from extent.qwen3_parity import jax_attention_params
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.rorope_bkv_conversion import map_qwen3_to_rorope_bkv


EXACT_LIFT_VARIANT = "INIT-K-balanced-qkvo-lift"


@dataclass(frozen=True)
class MaterializedMixerReport:
    layer_index: int
    target_mixer: str
    method: str
    tensor_count: int
    parameter_count: int
    calibration_tokens: int = 0
    cache_reduction_fraction: float | None = None
    calibration_relative_l2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _host_tree_into_target_shards(host_tree: Mapping, target_tree: Mapping) -> Mapping:
    """Cast a host subtree and place every array directly into target sharding."""
    host_flat = traverse_util.flatten_dict(host_tree)
    target_flat = traverse_util.flatten_dict(target_tree)
    if set(host_flat) != set(target_flat):
        missing = sorted(set(target_flat) - set(host_flat))
        extra = sorted(set(host_flat) - set(target_flat))
        raise ValueError(f"mixer parameter contract mismatch: missing={missing}, extra={extra}")
    result = {}
    for path, target in target_flat.items():
        host = np.asarray(jax.device_get(host_flat[path]), dtype=np.dtype(target.dtype))
        if host.ndim:
            host = np.ascontiguousarray(host)
        if tuple(host.shape) != tuple(target.shape):
            raise ValueError(f"shape mismatch at {'/'.join(path)}: {host.shape} != {target.shape}")
        if isinstance(target, jax.Array):
            result[path] = jax.make_array_from_callback(
                target.shape,
                target.sharding,
                lambda index, host=host: host[index],
            )
        else:
            result[path] = host
    tree = traverse_util.unflatten_dict(result)
    return freeze(tree) if isinstance(target_tree, FrozenDict) else tree


def _replace_layer_mixer(params: Mapping, layer_index: int, name: str, mixer: Mapping) -> Mapping:
    was_frozen = isinstance(params, FrozenDict)
    mutable = dict(params)
    layer_name = f"layers_{layer_index}"
    layer = dict(mutable[layer_name])
    layer[name] = mixer
    mutable[layer_name] = freeze(layer) if isinstance(mutable[layer_name], FrozenDict) else layer
    return freeze(mutable) if was_frozen else mutable


def materialize_exact_lift_layer(
    params: Mapping,
    arrays: Mapping[str, np.ndarray],
    source: Qwen3TeacherConfig,
    config: HybridConfig,
    layer_index: int,
) -> tuple[Mapping, MaterializedMixerReport]:
    target = params[f"layers_{layer_index}"]["mamba"]
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        variants, reports = build_qwen3_to_mamba3_transplant_variants(
            target, arrays, source, config.mamba, layer_index
        )
        selected_host = jax.device_get(variants[EXACT_LIFT_VARIANT])
        del variants
    selected = _host_tree_into_target_shards(selected_host, target)
    flat = traverse_util.flatten_dict(selected)
    report = MaterializedMixerReport(
        layer_index=layer_index,
        target_mixer="mamba3_mimo",
        method=reports[EXACT_LIFT_VARIANT].variant,
        tensor_count=len(flat),
        parameter_count=sum(int(np.prod(value.shape)) for value in flat.values()),
    )
    return _replace_layer_mixer(params, layer_index, "mamba", selected), report


def materialize_rorope_bkv_layer(
    params: Mapping,
    arrays: Mapping[str, np.ndarray],
    calibration_inputs: np.ndarray,
    source: Qwen3TeacherConfig,
    config: HybridConfig,
    layer_index: int,
) -> tuple[Mapping, MaterializedMixerReport]:
    target = params[f"layers_{layer_index}"]["self_attn"]
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        mapped, conversion = map_qwen3_to_rorope_bkv(
            arrays,
            source,
            layer_index,
            jax.numpy.asarray(calibration_inputs, dtype=jax.numpy.float32),
            latent_rank=config.mla.kv_lora_rank,
            svd_seed=layer_index,
        )
        mapped_host = jax.device_get(mapped)
    selected = _host_tree_into_target_shards(mapped_host, target)
    flat = traverse_util.flatten_dict(selected)
    report = MaterializedMixerReport(
        layer_index=layer_index,
        target_mixer="rorope_bkv_mla",
        method=conversion.method,
        tensor_count=len(flat),
        parameter_count=sum(int(np.prod(value.shape)) for value in flat.values()),
        calibration_tokens=conversion.calibration_tokens,
        cache_reduction_fraction=conversion.cache_reduction_fraction,
        calibration_relative_l2=conversion.calibration_joint_reconstruction_relative_l2,
    )
    return _replace_layer_mixer(params, layer_index, "self_attn", selected), report


def materialize_qwen_gqa_layer(
    params: Mapping,
    arrays: Mapping[str, np.ndarray],
    source: Qwen3TeacherConfig,
    layer_index: int,
) -> tuple[Mapping, MaterializedMixerReport]:
    """Copy one retained Qwen GQA mixer into the hybrid's target shards."""
    target = params[f"layers_{layer_index}"]["self_attn"]
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        mapped_host = jax.device_get(
            jax_attention_params(arrays, source, layer_index)
        )
    selected = _host_tree_into_target_shards(mapped_host, target)
    flat = traverse_util.flatten_dict(selected)
    report = MaterializedMixerReport(
        layer_index=layer_index,
        target_mixer="qwen3_gqa",
        method="DIRECT-QWEN3-GQA",
        tensor_count=len(flat),
        parameter_count=sum(int(np.prod(value.shape)) for value in flat.values()),
    )
    return _replace_layer_mixer(params, layer_index, "self_attn", selected), report
