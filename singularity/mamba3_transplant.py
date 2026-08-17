from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

from singularity.config import Mamba3Config
from singularity.qwen3_teacher import Qwen3TeacherConfig
from singularity.weight_mapping import fit_matrix


MAMBA_IN_LLAMA_REFERENCE_COMMIT = "b03f123152eeba5f2ae9d8694f4a001147e0a14c"


@dataclass(frozen=True)
class Mamba3TransplantReport:
    variant: str
    description: str
    copied_in_projection_columns: int
    total_in_projection_columns: int
    copied_output_projection: bool
    mimo_channel_rule: str
    prior_reference_commit: str | None = None


def mamba3_projection_slices(
    hidden_size: int, config: Mamba3Config
) -> dict[str, slice]:
    inner = int(hidden_size * config.expand)
    heads = inner // config.head_dim
    widths = {
        "z": inner,
        "x": inner,
        "b": config.mimo_rank * config.groups * config.d_state,
        "c": config.mimo_rank * config.groups * config.d_state,
        "dt": heads,
        "a": heads,
        "trap": heads,
        "angle": int(config.d_state * config.rope_fraction) // 2,
    }
    result = {}
    offset = 0
    for name, width in widths.items():
        result[name] = slice(offset, offset + width)
        offset += width
    return result


def _kernel(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(arrays[name], dtype=np.float32).T


def _head_group_projection(
    kernel: np.ndarray,
    source_heads: int,
    source_head_dim: int,
    target_rank: int,
    target_state_dim: int,
    *,
    copy_channels: bool,
) -> np.ndarray:
    heads = kernel.reshape(kernel.shape[0], source_heads, source_head_dim)
    if copy_channels:
        groups = [np.arange(source_heads)] * target_rank
    else:
        groups = np.array_split(np.arange(source_heads), target_rank)
    channels = []
    for group in groups:
        pooled = heads[:, group].mean(axis=1)
        channels.append(fit_matrix(pooled, (kernel.shape[0], target_state_dim)))
    return np.concatenate(channels, axis=1).astype(np.float32)


def build_qwen3_to_mamba3_transplant_variants(
    base_params: Mapping,
    arrays: Mapping[str, np.ndarray],
    source: Qwen3TeacherConfig,
    config: Mamba3Config,
    layer_index: int,
) -> tuple[dict[str, FrozenDict], dict[str, Mamba3TransplantReport]]:
    """Build controlled INIT-A..E mixer initializations from one random base."""
    if config.groups != 1:
        raise ValueError("the Qwen3 transplant currently requires one Mamba B/C group")
    prefix = f"model.layers.{layer_index}.self_attn"
    q = _kernel(arrays, f"{prefix}.q_proj.weight")
    k = _kernel(arrays, f"{prefix}.k_proj.weight")
    v = _kernel(arrays, f"{prefix}.v_proj.weight")
    o = _kernel(arrays, f"{prefix}.o_proj.weight")
    q_norm = np.asarray(arrays[f"{prefix}.q_norm.weight"], dtype=np.float32)
    k_norm = np.asarray(arrays[f"{prefix}.k_norm.weight"], dtype=np.float32)
    slices = mamba3_projection_slices(source.hidden_size, config)
    in_shape = tuple(base_params["in_proj"]["kernel"].shape)
    out_shape = tuple(base_params["out_proj"]["kernel"].shape)
    total_width = in_shape[1]
    bc_width = config.mimo_rank * config.groups * config.d_state

    def fresh() -> dict:
        return unfreeze(base_params) if isinstance(base_params, FrozenDict) else unfreeze(freeze(base_params))

    def replace(
        *,
        copy_out: bool,
        copy_bc_norm: bool = False,
        x_projection: np.ndarray | None = None,
        b_projection: np.ndarray | None = None,
        c_projection: np.ndarray | None = None,
    ) -> FrozenDict:
        params = fresh()
        kernel = np.asarray(params["in_proj"]["kernel"], dtype=np.float32).copy()
        if x_projection is not None:
            kernel[:, slices["x"]] = x_projection
        if b_projection is not None:
            kernel[:, slices["b"]] = b_projection
        if c_projection is not None:
            kernel[:, slices["c"]] = c_projection
        params["in_proj"]["kernel"] = jnp.asarray(kernel)
        if copy_out:
            params["out_proj"]["kernel"] = jnp.asarray(fit_matrix(o, out_shape))
        if copy_bc_norm:
            indices = np.floor(
                np.arange(config.d_state) * source.head_dim / config.d_state
            ).astype(np.int32)
            params["b_norm"]["scale"] = jnp.asarray(k_norm[indices])
            params["c_norm"]["scale"] = jnp.asarray(q_norm[indices])
        return freeze(params)

    x_projection = fit_matrix(v, (source.hidden_size, slices["x"].stop - slices["x"].start))
    prior_b = fit_matrix(k, (source.hidden_size, bc_width))
    prior_c = fit_matrix(q, (source.hidden_size, bc_width))
    siso_b = _head_group_projection(
        k,
        source.num_key_value_heads,
        source.head_dim,
        config.mimo_rank,
        config.d_state,
        copy_channels=True,
    )
    siso_c = _head_group_projection(
        q,
        source.num_attention_heads,
        source.head_dim,
        config.mimo_rank,
        config.d_state,
        copy_channels=True,
    )
    mimo_b = _head_group_projection(
        k,
        source.num_key_value_heads,
        source.head_dim,
        config.mimo_rank,
        config.d_state,
        copy_channels=False,
    )
    mimo_c = _head_group_projection(
        q,
        source.num_attention_heads,
        source.head_dim,
        config.mimo_rank,
        config.d_state,
        copy_channels=False,
    )

    variants = {
        "INIT-A-random": replace(copy_out=False),
        "INIT-B-output-only": replace(copy_out=True),
        "INIT-C-prior-qkvo-port": replace(
            copy_out=True,
            copy_bc_norm=True,
            x_projection=x_projection,
            b_projection=prior_b,
            c_projection=prior_c,
        ),
        "INIT-D-siso-copied": replace(
            copy_out=True,
            copy_bc_norm=True,
            x_projection=x_projection,
            b_projection=siso_b,
            c_projection=siso_c,
        ),
        "INIT-E-mimo-aware": replace(
            copy_out=True,
            copy_bc_norm=True,
            x_projection=x_projection,
            b_projection=mimo_b,
            c_projection=mimo_c,
        ),
    }
    copied_qkvo = (
        slices["x"].stop - slices["x"].start + 2 * bc_width
    )
    reports = {
        "INIT-A-random": Mamba3TransplantReport(
            "INIT-A-random", "Canonical random Mamba-3 initialization.", 0, total_width, False, "random"
        ),
        "INIT-B-output-only": Mamba3TransplantReport(
            "INIT-B-output-only", "Only the Qwen attention output projection is resized and copied.", 0, total_width, True, "random"
        ),
        "INIT-C-prior-qkvo-port": Mamba3TransplantReport(
            "INIT-C-prior-qkvo-port",
            "Mamba-in-the-Llama-style V->x, K->B, Q->C, O->out port adapted to mismatched widths.",
            copied_qkvo,
            total_width,
            True,
            "flat resize",
            MAMBA_IN_LLAMA_REFERENCE_COMMIT,
        ),
        "INIT-D-siso-copied": Mamba3TransplantReport(
            "INIT-D-siso-copied",
            "Head-pooled K/Q state projections copied identically into every MIMO channel.",
            copied_qkvo,
            total_width,
            True,
            "one pooled channel copied across rank",
        ),
        "INIT-E-mimo-aware": Mamba3TransplantReport(
            "INIT-E-mimo-aware",
            "Disjoint Q/K head groups initialize distinct MIMO state channels.",
            copied_qkvo,
            total_width,
            True,
            "disjoint source-head groups",
        ),
    }
    for name, params in variants.items():
        if tuple(params["in_proj"]["kernel"].shape) != in_shape:
            raise ValueError(f"{name} changed the Mamba input-projection shape")
        if not all(
            np.all(np.isfinite(np.asarray(value))) for value in jax.tree.leaves(params)
        ):
            raise FloatingPointError(f"{name} produced non-finite parameters")
    return variants, reports
