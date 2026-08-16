from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_v5e_mesh(devices: list[jax.Device] | None = None) -> Mesh:
    """Create the 1x4x2 data/fsdp/tensor mesh intended for one v5e-8 slice.

    On CPU/single-device bring-up the unavailable axes collapse to size one.
    """
    devices = list(jax.devices() if devices is None else devices)
    count = len(devices)
    tensor = 2 if count % 2 == 0 else 1
    fsdp = min(4, count // tensor)
    while count % (tensor * fsdp):
        fsdp -= 1
    data = count // (tensor * fsdp)
    array = np.asarray(devices, dtype=object).reshape(data, fsdp, tensor)
    return Mesh(array, ("data", "fsdp", "tensor"))


def parameter_partition_spec(path: tuple[str, ...], shape: tuple[int, ...]) -> P:
    """Conservative FSDP+tensor rules for Flax kernels [input, output]."""
    name = "/".join(path)
    if len(shape) < 2:
        return P()
    if "embed_tokens/embedding" in name:
        return P("fsdp", "tensor")
    if name.endswith("lm_head/kernel"):
        return P("fsdp", "tensor")
    if any(token in name for token in ("down_proj/kernel", "out_proj/kernel")):
        return P("tensor", "fsdp")
    if name.endswith("kernel"):
        return P("fsdp", "tensor")
    return P()


def named_sharding_tree(params: Mapping, mesh: Mesh) -> Mapping:
    """Return a sharding pytree with the same nested structure as params."""
    from flax import traverse_util

    flat = traverse_util.flatten_dict(params)
    specs = {
        path: NamedSharding(mesh, parameter_partition_spec(path, value.shape))
        for path, value in flat.items()
    }
    return traverse_util.unflatten_dict(specs)


def activation_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P("data", None, "tensor"))


def batch_sharding(mesh: Mesh) -> NamedSharding:
    """Shard batches over data replicas and replicate over model axes."""
    return NamedSharding(mesh, P("data", None))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def sharding_tree_from_arrays(tree: Any) -> Any:
    """Capture the concrete layouts chosen by GSPMD for reuse in jitted steps."""
    return jax.tree.map(lambda value: value.sharding, tree)


def count_partitioned_arrays(tree: Any) -> tuple[int, int]:
    arrays = [leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)]
    partitioned = sum(not leaf.sharding.is_fully_replicated for leaf in arrays)
    return partitioned, len(arrays)


def validate_partition_specs(
    params: Mapping,
    mesh: Mesh | None = None,
    *,
    axis_sizes: Mapping[str, int] | None = None,
) -> None:
    """Fail before compilation when a tensor dimension cannot be evenly sharded."""
    from flax import traverse_util

    if axis_sizes is None:
        if mesh is None:
            raise ValueError("mesh or axis_sizes is required")
        axis_sizes = mesh.shape
    for path, value in traverse_util.flatten_dict(params).items():
        spec = parameter_partition_spec(path, value.shape)
        for dimension, axes in zip(value.shape, spec):
            if axes is None:
                continue
            axes = (axes,) if isinstance(axes, str) else axes
            parts = int(np.prod([axis_sizes[axis] for axis in axes]))
            if dimension % parts:
                raise ValueError(
                    f"{'/'.join(path)} shape {value.shape} is not divisible by {axes}={parts}"
                )
