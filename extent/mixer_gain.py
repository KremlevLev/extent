"""Foldable, per-Mamba output gains for full-model calibration."""

from __future__ import annotations

import jax.numpy as jnp
from flax.core import unfreeze


def scaled_mamba_parameters(params, layer_indices, gains):
    """Return a parameter tree whose Mamba output kernels have selected gains.

    Multiplying the output projection is exactly equivalent to multiplying the
    Mamba branch contribution before its residual addition. No inference-time
    module or extra cache is needed after folding the gains into the kernels.
    """
    if gains.ndim != 1 or gains.shape[0] != len(layer_indices):
        raise ValueError("one gain is required for each replaced layer")
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("layer indices must be unique")
    tree = unfreeze(params)
    for position, layer in enumerate(layer_indices):
        kernel = tree[f"layers_{layer}"]["mamba"]["out_proj"]["kernel"]
        tree[f"layers_{layer}"]["mamba"]["out_proj"]["kernel"] = (
            kernel.astype(jnp.float32) * gains[position]
        ).astype(kernel.dtype)
    return tree


def effective_gains(raw, arm, count):
    """Map unconstrained optimizer coordinates into the registered [0.5, 1.5] band."""
    if arm == "GLOBAL":
        if raw.shape != (1,):
            raise ValueError("GLOBAL requires exactly one scalar")
        coordinates = jnp.broadcast_to(raw, (count,))
    elif arm == "PER-LAYER":
        if raw.shape != (count,):
            raise ValueError("PER-LAYER requires one scalar per layer")
        coordinates = raw
    else:
        raise ValueError(f"unknown gain arm: {arm}")
    return 1.0 + 0.5 * jnp.tanh(coordinates.astype(jnp.float32))
