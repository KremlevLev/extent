"""Small trainable corrections that fold into a transplanted hybrid's kernels."""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp
from flax.core import unfreeze

ARMS = ("OUT-LORA", "INOUT-LORA", "MLP-READOUT-LORA", "HEAD-GAIN")


def correction_paths(order, arm):
    if arm not in ARMS:
        raise ValueError(f"unknown recovery subspace: {arm}")
    for layer in order:
        root = f"layers_{layer}"
        if arm == "MLP-READOUT-LORA":
            yield (root, "mlp", "down_proj", "kernel")
        else:
            yield (root, "mamba", "out_proj", "kernel")
            if arm == "INOUT-LORA":
                yield (root, "mamba", "in_proj", "kernel")


def _leaf(tree, path):
    for component in path:
        tree = tree[component]
    return tree


def initialize_corrections(base, order, arm, *, seed, rank=8, head_dim=64):
    """Zero-change initial state; OUT/INOUT share exactly the same out factors."""
    if rank < 1 or head_dim < 1:
        raise ValueError("rank and head_dim must be positive")
    result = {}
    for path in correction_paths(order, arm):
        kernel = _leaf(base, path)
        name = "/".join(path)
        if arm == "HEAD-GAIN":
            if kernel.shape[0] % head_dim:
                raise ValueError("Mamba output width must divide into heads")
            result[name] = {"raw": jnp.zeros((kernel.shape[0] // head_dim,), jnp.float32)}
        else:
            layer = int(path[0].removeprefix("layers_"))
            projection = {"out_proj": 0, "in_proj": 1, "down_proj": 2}[path[2]]
            key = jax.random.fold_in(jax.random.key(seed), layer * 3 + projection)
            result[name] = {
                "a": jax.random.normal(key, (kernel.shape[0], rank), jnp.float32)
                     / math.sqrt(kernel.shape[0]),
                "b": jnp.zeros((rank, kernel.shape[1]), jnp.float32),
            }
    return result


def apply_corrections(base, coordinates, *, head_dim=64, protected_input_columns=None):
    """FP32 correction folded into stored kernels; no new inference module."""
    tree = unfreeze(base)
    for name, factors in coordinates.items():
        path = name.split("/")
        parent = tree
        for component in path[:-1]:
            parent = parent[component]
        kernel = parent[path[-1]]
        if "raw" in factors:
            gains = 1.0 + 0.5 * jnp.tanh(factors["raw"])
            scale = jnp.repeat(gains, head_dim)[:, None]
            if scale.shape[0] != kernel.shape[0]:
                raise ValueError("head gains do not match output kernel")
            updated = kernel.astype(jnp.float32) * scale
        else:
            b = factors["b"]
            if protected_input_columns is not None and path[-2] == "in_proj":
                b = b * (jnp.arange(b.shape[1]) < protected_input_columns)[None, :]
            updated = kernel.astype(jnp.float32) + factors["a"] @ b
        parent[path[-1]] = updated.astype(kernel.dtype)
    return tree


def coordinate_count(coordinates):
    return sum(int(leaf.size) for leaf in jax.tree.leaves(coordinates))


def coordinates_finite(coordinates):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf))
                             for leaf in jax.tree.leaves(coordinates)]))
