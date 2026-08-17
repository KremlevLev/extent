from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import numpy as np
from flax.core import freeze

from singularity.config import MLAConfig
from singularity.qwen3_teacher import Qwen3TeacherConfig
from singularity.weight_mapping import truncated_svd


@dataclass(frozen=True)
class MLAConversionReport:
    method: str
    layer_index: int
    source_cache_elements_per_token: int
    target_cache_elements_per_token: int
    cache_reduction_fraction: float
    joint_reconstruction_relative_l2: float
    rope_aggregation_relative_l2: float


@dataclass(frozen=True)
class MLAJointFactors:
    joint: np.ndarray
    down: np.ndarray
    up: np.ndarray
    rope_dim: int
    partial_rope_strategy: str

    @property
    def max_rank(self) -> int:
        return self.down.shape[1]


def partial_rope_indices(
    head_dim: int,
    rope_dim: int,
    strategy: str = "high",
) -> tuple[np.ndarray, np.ndarray]:
    """Return content and RoPE indices while keeping split-half pairs together."""
    if head_dim % 2 or rope_dim % 2 or not 0 < rope_dim <= head_dim:
        raise ValueError("head_dim and rope_dim must be positive even compatible widths")
    pairs = head_dim // 2
    selected_pairs = rope_dim // 2
    if strategy == "high":
        first = np.arange(selected_pairs)
    elif strategy == "low":
        first = np.arange(pairs - selected_pairs, pairs)
    else:
        raise ValueError("partial RoPE strategy must be high or low")
    rope = np.concatenate((first, first + pairs))
    content = np.setdiff1d(np.arange(head_dim), rope, assume_unique=True)
    return content, rope


def qwen3_mla_conversion_config(
    source: Qwen3TeacherConfig,
    *,
    kv_lora_rank: int,
    rope_dim: int,
    grouped_rope: bool,
    partial_rope_strategy: str = "high",
) -> MLAConfig:
    return MLAConfig(
        num_heads=source.num_attention_heads,
        num_kv_heads=source.num_key_value_heads,
        num_key_rope_heads=source.num_key_value_heads if grouped_rope else 1,
        q_lora_rank=0,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=source.head_dim - rope_dim,
        qk_rope_head_dim=rope_dim,
        v_head_dim=source.head_dim,
        rope_theta=source.rope_theta,
        rope_original_head_dim=source.head_dim,
        partial_rope_strategy=partial_rope_strategy,
        use_qk_norm=True,
        use_kv_latent_norm=False,
        qk_head_chunk_size=4 if source.num_attention_heads % 4 == 0 else 0,
    )


def _kernel(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(arrays[name], dtype=np.float32).T


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float32).tiny)
    return float(np.linalg.norm(candidate - reference) / denominator)


def factorize_qwen3_joint_kv(
    arrays: Mapping[str, np.ndarray],
    source: Qwen3TeacherConfig,
    layer_index: int,
    *,
    rope_dim: int,
    max_rank: int,
    partial_rope_strategy: str = "high",
    seed: int = 0,
) -> MLAJointFactors:
    """Compute one maximum-rank factorization reusable by every lower-rank sweep."""
    prefix = f"model.layers.{layer_index}.self_attn"
    content_indices, _ = partial_rope_indices(
        source.head_dim, rope_dim, partial_rope_strategy
    )
    key = _kernel(arrays, f"{prefix}.k_proj.weight").reshape(
        source.hidden_size, source.num_key_value_heads, source.head_dim
    )
    value = _kernel(arrays, f"{prefix}.v_proj.weight").reshape(
        source.hidden_size, source.num_key_value_heads, source.head_dim
    )
    joint = np.concatenate((key[:, :, content_indices], value), axis=-1).reshape(
        source.hidden_size, -1
    )
    down, up = truncated_svd(joint, max_rank, seed=seed)
    return MLAJointFactors(
        joint=joint,
        down=down,
        up=up,
        rope_dim=rope_dim,
        partial_rope_strategy=partial_rope_strategy,
    )


def convert_qwen3_gqa_to_mla_joint_svd(
    arrays: Mapping[str, np.ndarray],
    source: Qwen3TeacherConfig,
    target: MLAConfig,
    layer_index: int,
    *,
    seed: int = 0,
    factors: MLAJointFactors | None = None,
) -> tuple[dict, MLAConversionReport]:
    """MHA2MLA-style joint KV SVD with either grouped or shared RoPE keys."""
    if target.q_lora_rank != 0 or target.use_kv_latent_norm:
        raise ValueError("source-faithful SVD conversion requires direct Q and no latent RMSNorm")
    if target.num_kv_heads != source.num_key_value_heads:
        raise ValueError("this baseline preserves the source GQA KV-head count")
    prefix = f"model.layers.{layer_index}.self_attn"
    content_indices, rope_indices = partial_rope_indices(
        source.head_dim, target.qk_rope_head_dim, target.partial_rope_strategy
    )

    query = _kernel(arrays, f"{prefix}.q_proj.weight").reshape(
        source.hidden_size, source.num_attention_heads, source.head_dim
    )
    key = _kernel(arrays, f"{prefix}.k_proj.weight").reshape(
        source.hidden_size, source.num_key_value_heads, source.head_dim
    )
    value = _kernel(arrays, f"{prefix}.v_proj.weight").reshape(
        source.hidden_size, source.num_key_value_heads, source.head_dim
    )
    query_ordered = np.concatenate(
        (query[:, :, content_indices], query[:, :, rope_indices]), axis=-1
    ).reshape(source.hidden_size, -1)
    key_content = key[:, :, content_indices]
    key_rope = key[:, :, rope_indices]
    joint = np.concatenate((key_content, value), axis=-1).reshape(
        source.hidden_size, -1
    )
    if factors is None:
        factors = factorize_qwen3_joint_kv(
            arrays,
            source,
            layer_index,
            rope_dim=target.qk_rope_head_dim,
            max_rank=target.kv_lora_rank,
            partial_rope_strategy=target.partial_rope_strategy,
            seed=seed,
        )
    if (
        factors.rope_dim != target.qk_rope_head_dim
        or factors.partial_rope_strategy != target.partial_rope_strategy
        or factors.joint.shape != joint.shape
    ):
        raise ValueError("precomputed joint factors do not match the MLA target")
    if target.kv_lora_rank > factors.max_rank:
        raise ValueError(
            f"requested rank {target.kv_lora_rank} exceeds factor rank {factors.max_rank}"
        )
    down = factors.down[:, : target.kv_lora_rank]
    up = factors.up[: target.kv_lora_rank]

    if target.num_key_rope_heads == source.num_key_value_heads:
        cached_rope = key_rope
    elif target.num_key_rope_heads == 1:
        cached_rope = np.mean(key_rope, axis=1, keepdims=True)
    else:
        raise ValueError("key RoPE heads must be one or match source KV heads")
    expanded_cached_rope = np.repeat(
        cached_rope, source.num_key_value_heads // target.num_key_rope_heads, axis=1
    )
    rope_error = _relative_l2(key_rope, expanded_cached_rope)
    kv_wa = np.concatenate((down, cached_rope.reshape(source.hidden_size, -1)), axis=-1)

    norm_order = np.concatenate((content_indices, rope_indices))
    q_norm = np.asarray(arrays[f"{prefix}.q_norm.weight"], dtype=np.float32)[norm_order]
    k_norm = np.asarray(arrays[f"{prefix}.k_norm.weight"], dtype=np.float32)[norm_order]
    params = freeze(
        {
            "query_proj": {"kernel": jnp.asarray(query_ordered)},
            "kv_wa_proj": {"kernel": jnp.asarray(kv_wa)},
            "kv_wb_proj": {"kernel": jnp.asarray(up)},
            "out_proj": {"kernel": jnp.asarray(_kernel(arrays, f"{prefix}.o_proj.weight"))},
            "q_norm": {"scale": jnp.asarray(q_norm)},
            "k_norm": {"scale": jnp.asarray(k_norm)},
        }
    )
    source_cache = 2 * source.num_key_value_heads * source.head_dim
    target_cache = (
        target.kv_lora_rank
        + target.num_key_rope_heads * target.qk_rope_head_dim
    )
    report = MLAConversionReport(
        method=(
            "joint_svd_grouped_rope"
            if target.num_key_rope_heads > 1
            else "joint_svd_shared_rope"
        ),
        layer_index=layer_index,
        source_cache_elements_per_token=source_cache,
        target_cache_elements_per_token=target_cache,
        cache_reduction_fraction=1.0 - target_cache / source_cache,
        joint_reconstruction_relative_l2=_relative_l2(joint, down @ up),
        rope_aggregation_relative_l2=rope_error,
    )
    return params, report


def qwen3_decoder_common_params(
    arrays: Mapping[str, np.ndarray], layer_index: int
) -> dict:
    """Copy the non-mixer tensors shared by teacher and converted decoder layers."""
    prefix = f"model.layers.{layer_index}"
    return {
        "input_layernorm": {
            "scale": jnp.asarray(arrays[f"{prefix}.input_layernorm.weight"], dtype=jnp.float32)
        },
        "post_attention_layernorm": {
            "scale": jnp.asarray(
                arrays[f"{prefix}.post_attention_layernorm.weight"], dtype=jnp.float32
            )
        },
        "mlp": {
            "gate_proj": {
                "kernel": jnp.asarray(arrays[f"{prefix}.mlp.gate_proj.weight"].T, dtype=jnp.float32)
            },
            "up_proj": {
                "kernel": jnp.asarray(arrays[f"{prefix}.mlp.up_proj.weight"].T, dtype=jnp.float32)
            },
            "down_proj": {
                "kernel": jnp.asarray(arrays[f"{prefix}.mlp.down_proj.weight"].T, dtype=jnp.float32)
            },
        },
    }
