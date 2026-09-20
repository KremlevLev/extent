"""Gradient intervention for isolating Mamba-3 timestep dynamics."""

from __future__ import annotations

from flax.core import FrozenDict, freeze, unfreeze
import jax.numpy as jnp

from extent.mamba3_transplant import mamba3_projection_slices


def freeze_mamba_dt_gradients(grads, config):
    """Zero raw-dt projection and dt-bias gradients in every Mamba layer."""
    was_frozen = isinstance(grads, FrozenDict)
    mutable = unfreeze(grads) if was_frozen else unfreeze(freeze(grads))
    dt_slice = mamba3_projection_slices(config.hidden_size, config.mamba)["dt"]
    for layer in config.mamba_layer_indices:
        mixer = mutable[f"layers_{layer}"]["mamba"]
        kernel = mixer["in_proj"]["kernel"]
        mixer["in_proj"]["kernel"] = kernel.at[:, dt_slice].set(
            jnp.zeros_like(kernel[:, dt_slice])
        )
        mixer["dt_bias"] = jnp.zeros_like(mixer["dt_bias"])
    return freeze(mutable) if was_frozen else mutable
