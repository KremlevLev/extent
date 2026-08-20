from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze

from extent.activation_compression import (
    bkv_balance_ratio,
    fit_activation_pca_jax,
)
from extent.layers.common import RMSNorm
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.rorope import fit_rorope_rotations, rotate_rorope_key


@dataclass(frozen=True)
class RoRoPEBKVMappingReport:
    method: str
    layer_index: int
    calibration_tokens: int
    latent_rank: int
    bkv_balance_ratio: float
    calibration_joint_reconstruction_relative_l2: float
    source_cache_elements_per_token: int
    target_cache_elements_per_token: int
    cache_reduction_fraction: float


def _kernel(arrays: Mapping[str, np.ndarray], name: str) -> jax.Array:
    return jnp.asarray(np.asarray(arrays[name], dtype=np.float32).T)


def map_qwen3_to_rorope_bkv(
    arrays: Mapping[str, np.ndarray],
    source: Qwen3TeacherConfig,
    layer_index: int,
    calibration_inputs: jax.Array,
    *,
    latent_rank: int = 448,
    svd_seed: int = 0,
) -> tuple[dict, RoRoPEBKVMappingReport]:
    """Fit a cache-producing RoRoPE+BKV module from normalized Qwen3 inputs."""
    if calibration_inputs.ndim != 3 or calibration_inputs.shape[-1] != source.hidden_size:
        raise ValueError("calibration inputs must be [batch, tokens, hidden_size]")
    tokens = calibration_inputs.shape[0] * calibration_inputs.shape[1]
    joint_width = (2 * source.num_key_value_heads - 1) * source.head_dim
    if not 0 < latent_rank <= min(tokens, joint_width):
        raise ValueError("latent rank exceeds the calibration matrix")
    prefix = f"model.layers.{layer_index}.self_attn"
    key = calibration_inputs @ _kernel(arrays, f"{prefix}.k_proj.weight")
    value = calibration_inputs @ _kernel(arrays, f"{prefix}.v_proj.weight")
    key = key.reshape(
        calibration_inputs.shape[0],
        calibration_inputs.shape[1],
        source.num_key_value_heads,
        source.head_dim,
    )
    value = value.reshape(
        calibration_inputs.shape[0],
        calibration_inputs.shape[1],
        source.num_key_value_heads,
        source.head_dim,
    )
    key = RMSNorm(source.head_dim, source.rms_norm_eps, jnp.float32).apply(
        {"params": {"scale": jnp.asarray(arrays[f"{prefix}.k_norm.weight"])}},
        key,
    )
    rotations, _ = fit_rorope_rotations(np.asarray(key))
    rotated_key = rotate_rorope_key(key, jnp.asarray(rotations))
    key_nope = rotated_key[:, :, 1:].reshape(-1, (source.num_key_value_heads - 1) * source.head_dim)
    value_flat = value.reshape(-1, source.num_key_value_heads * source.head_dim)
    balance = bkv_balance_ratio(np.asarray(key_nope), np.asarray(value_flat))
    joint = jnp.concatenate((key_nope / balance, value_flat), axis=-1)
    basis = fit_activation_pca_jax(joint, latent_rank, seed=svd_seed)
    reconstructed = (joint @ basis.T) @ basis
    relative_l2 = float(
        jnp.linalg.norm(reconstructed - joint)
        / jnp.maximum(jnp.linalg.norm(joint), jnp.finfo(jnp.float32).tiny)
    )
    if not np.isfinite(relative_l2) or not np.isfinite(balance):
        raise FloatingPointError("non-finite RoRoPE-BKV calibration result")

    params = freeze(
        {
            "q_proj": {"kernel": _kernel(arrays, f"{prefix}.q_proj.weight")},
            "k_proj": {"kernel": _kernel(arrays, f"{prefix}.k_proj.weight")},
            "v_proj": {"kernel": _kernel(arrays, f"{prefix}.v_proj.weight")},
            "o_proj": {"kernel": _kernel(arrays, f"{prefix}.o_proj.weight")},
            "q_norm": {
                "scale": jnp.asarray(arrays[f"{prefix}.q_norm.weight"], dtype=jnp.float32)
            },
            "k_norm": {
                "scale": jnp.asarray(arrays[f"{prefix}.k_norm.weight"], dtype=jnp.float32)
            },
            "rorope_rotations": jnp.asarray(rotations),
            "joint_basis": basis,
            "bkv_balance_ratio": jnp.asarray(balance, dtype=jnp.float32),
        }
    )
    source_cache = 2 * source.num_key_value_heads * source.head_dim
    target_cache = source.head_dim + latent_rank
    report = RoRoPEBKVMappingReport(
        method="qwen3_rorope_fold1_bkv_activation_pca",
        layer_index=layer_index,
        calibration_tokens=tokens,
        latent_rank=latent_rank,
        bkv_balance_ratio=balance,
        calibration_joint_reconstruction_relative_l2=relative_l2,
        source_cache_elements_per_token=source_cache,
        target_cache_elements_per_token=target_cache,
        cache_reduction_fraction=1.0 - target_cache / source_cache,
    )
    return params, report
