"""Lion update scaling for the copied Qwen backbone during Mamba recovery."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scale_copied_backbone_updates(updates, *, scale: float):
    """Scale Lion updates after sign/momentum; scaling its gradients is ineffective."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("backbone update scale must be in [0, 1]")

    def transform(path, update):
        is_mamba = any(
            getattr(component, "key", None) == "mamba" for component in path
        )
        return update if is_mamba else update * jnp.asarray(scale, dtype=update.dtype)

    return jax.tree_util.tree_map_with_path(transform, updates)
