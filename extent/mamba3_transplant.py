from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze

from extent.config import Mamba3Config
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.weight_mapping import fit_matrix


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


def _match_rms(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match one transplanted slice to the RMS of its random-base control."""
    value = np.asarray(value, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    value_rms = float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))
    reference_rms = float(
        np.sqrt(np.mean(np.square(reference), dtype=np.float64))
    )
    if not np.isfinite(value_rms) or value_rms <= np.finfo(np.float32).tiny:
        raise ValueError("cannot RMS-match a zero or non-finite transplant slice")
    return (value * (reference_rms / value_rms)).astype(np.float32)


def build_qwen3_to_mamba3_transplant_variants(
    base_params: Mapping,
    arrays: Mapping[str, np.ndarray],
    source: Qwen3TeacherConfig,
    config: Mamba3Config,
    layer_index: int,
) -> tuple[dict[str, FrozenDict], dict[str, Mamba3TransplantReport]]:
    """Build controlled mixer initializations from one canonical random base."""
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
    base_in = np.asarray(base_params["in_proj"]["kernel"], dtype=np.float32)
    matched_x = _match_rms(x_projection, base_in[:, slices["x"]])
    matched_b = _match_rms(prior_b, base_in[:, slices["b"]])
    matched_c = _match_rms(prior_c, base_in[:, slices["c"]])

    def interpolate(
        name: str, matched: np.ndarray, fraction: float
    ) -> np.ndarray:
        random_slice = base_in[:, slices[name]]
        return (
            (1.0 - fraction) * random_slice + fraction * matched
        ).astype(np.float32)

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
    single_b = np.zeros_like(siso_b)
    single_c = np.zeros_like(siso_c)
    single_b[:, : config.d_state] = siso_b[:, : config.d_state]
    single_c[:, : config.d_state] = siso_c[:, : config.d_state]
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

    def exact_lift(params: FrozenDict, *, balanced: bool) -> FrozenDict:
        """Embed one SISO parameterization into the canonical MIMO contract.

        The single-channel and balanced forms compute the same recurrence.  In
        the balanced form every B/C channel is copied, ``mimo_x`` and
        ``mimo_o`` are 1/R, and D is multiplied by R so the skip path is not
        accidentally attenuated.  This compensates for BC RMSNorm preventing
        direct 1/sqrt(R) scaling of B/C projections.
        """
        lifted = unfreeze(params)
        rank = config.mimo_rank
        if balanced:
            lifted["mimo_x"] = jnp.full_like(lifted["mimo_x"], 1.0 / rank)
            lifted["mimo_o"] = jnp.full_like(lifted["mimo_o"], 1.0 / rank)
            lifted["mimo_z"] = jnp.ones_like(lifted["mimo_z"])
            lifted["D"] = lifted["D"] * rank
            for name in ("b_bias", "c_bias"):
                lifted[name] = jnp.broadcast_to(
                    lifted[name][:, :1, :], lifted[name].shape
                )
        else:
            for name in ("mimo_x", "mimo_o", "mimo_z"):
                value = jnp.zeros_like(lifted[name])
                value = value.at[:, 0, :].set(1.0)
                lifted[name] = value
            for name in ("b_bias", "c_bias"):
                value = jnp.zeros_like(lifted[name])
                value = value.at[:, 0, :].set(lifted[name][:, 0, :])
                lifted[name] = value
        return freeze(lifted)

    single_lift = exact_lift(
        replace(
            copy_out=True,
            copy_bc_norm=True,
            x_projection=x_projection,
            b_projection=single_b,
            c_projection=single_c,
        ),
        balanced=False,
    )
    balanced_lift = exact_lift(
        replace(
            copy_out=True,
            copy_bc_norm=True,
            x_projection=x_projection,
            b_projection=siso_b,
            c_projection=siso_c,
        ),
        balanced=True,
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
        "INIT-F-variance-matched-qkvo": replace(
            copy_out=True,
            copy_bc_norm=True,
            x_projection=matched_x,
            b_projection=matched_b,
            c_projection=matched_c,
        ),
        "INIT-G-vm-qkvo-blend-0.25": replace(
            copy_out=True,
            copy_bc_norm=True,
            x_projection=interpolate("x", matched_x, 0.25),
            b_projection=interpolate("b", matched_b, 0.25),
            c_projection=interpolate("c", matched_c, 0.25),
        ),
        "INIT-H-vm-qkvo-blend-0.5": replace(
            copy_out=True,
            copy_bc_norm=True,
            x_projection=interpolate("x", matched_x, 0.5),
            b_projection=interpolate("b", matched_b, 0.5),
            c_projection=interpolate("c", matched_c, 0.5),
        ),
        "INIT-J-single-channel-qkvo-lift": single_lift,
        "INIT-K-balanced-qkvo-lift": balanced_lift,
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
        "INIT-F-variance-matched-qkvo": Mamba3TransplantReport(
            "INIT-F-variance-matched-qkvo",
            "The flat QKVO port with x/B/C slices independently RMS-matched to their canonical random controls.",
            copied_qkvo,
            total_width,
            True,
            "flat resize with per-slice RMS matching",
            MAMBA_IN_LLAMA_REFERENCE_COMMIT,
        ),
        "INIT-G-vm-qkvo-blend-0.25": Mamba3TransplantReport(
            "INIT-G-vm-qkvo-blend-0.25",
            "A 25% interpolation from the canonical random x/B/C slices toward the RMS-matched QKVO port.",
            copied_qkvo,
            total_width,
            True,
            "25% variance-matched flat port plus 75% random base",
            MAMBA_IN_LLAMA_REFERENCE_COMMIT,
        ),
        "INIT-H-vm-qkvo-blend-0.5": Mamba3TransplantReport(
            "INIT-H-vm-qkvo-blend-0.5",
            "A 50% interpolation from the canonical random x/B/C slices toward the RMS-matched QKVO port.",
            copied_qkvo,
            total_width,
            True,
            "50% variance-matched flat port plus 50% random base",
            MAMBA_IN_LLAMA_REFERENCE_COMMIT,
        ),
        "INIT-J-single-channel-qkvo-lift": Mamba3TransplantReport(
            "INIT-J-single-channel-qkvo-lift",
            "A pooled QKVO SISO donor embedded into exactly one active MIMO input/output channel.",
            copied_qkvo,
            total_width,
            True,
            "exact single-active-channel SISO-to-MIMO embedding",
            MAMBA_IN_LLAMA_REFERENCE_COMMIT,
        ),
        "INIT-K-balanced-qkvo-lift": Mamba3TransplantReport(
            "INIT-K-balanced-qkvo-lift",
            "The same pooled QKVO SISO donor distributed exactly across all MIMO channels with skip-path compensation.",
            copied_qkvo,
            total_width,
            True,
            "exact balanced-rank SISO-to-MIMO embedding; x/o=1/R and D'=R*D",
            MAMBA_IN_LLAMA_REFERENCE_COMMIT,
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
