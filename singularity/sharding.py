from __future__ import annotations

from collections.abc import Mapping

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
