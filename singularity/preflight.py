from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from flax import traverse_util

from singularity.sharding import parameter_partition_spec, validate_partition_specs


@dataclass(frozen=True)
class TensorBudget:
    path: str
    shape: tuple[int, ...]
    parameters: int
    global_bytes: int
    bytes_per_device: int
    partitions: int


@dataclass(frozen=True)
class PreflightReport:
    parameter_count: int
    tensor_count: int
    partitioned_tensor_count: int
    global_weight_bytes: int
    weight_bytes_per_device: int
    training_bytes_per_device: int
    mesh_size: int
    largest_tensors: tuple[TensorBudget, ...]

    @property
    def global_weight_gib(self) -> float:
        return self.global_weight_bytes / 2**30

    @property
    def weight_gib_per_device(self) -> float:
        return self.weight_bytes_per_device / 2**30

    @property
    def training_gib_per_device(self) -> float:
        return self.training_bytes_per_device / 2**30


def _partition_count(spec: tuple[Any, ...], axis_sizes: Mapping[str, int]) -> int:
    count = 1
    for axes in spec:
        if axes is None:
            continue
        axes = (axes,) if isinstance(axes, str) else axes
        count *= int(np.prod([axis_sizes[axis] for axis in axes]))
    return count


def build_preflight_report(
    abstract_params: Any,
    axis_sizes: Mapping[str, int],
    *,
    training_copies: int = 3,
    largest: int = 12,
) -> PreflightReport:
    """Compute exact persistent-memory cost for weights/grads/Lion momentum."""
    validate_partition_specs(abstract_params, axis_sizes=axis_sizes)
    budgets: list[TensorBudget] = []
    for path, value in traverse_util.flatten_dict(abstract_params).items():
        spec = parameter_partition_spec(path, value.shape)
        partitions = _partition_count(tuple(spec), axis_sizes)
        parameters = int(np.prod(value.shape))
        global_bytes = parameters * np.dtype(value.dtype).itemsize
        budgets.append(
            TensorBudget(
                path="/".join(path),
                shape=tuple(value.shape),
                parameters=parameters,
                global_bytes=global_bytes,
                bytes_per_device=global_bytes // partitions,
                partitions=partitions,
            )
        )
    global_bytes = sum(item.global_bytes for item in budgets)
    per_device = sum(item.bytes_per_device for item in budgets)
    mesh_size = int(np.prod(list(axis_sizes.values())))
    return PreflightReport(
        parameter_count=sum(item.parameters for item in budgets),
        tensor_count=len(budgets),
        partitioned_tensor_count=sum(item.partitions > 1 for item in budgets),
        global_weight_bytes=global_bytes,
        weight_bytes_per_device=per_device,
        training_bytes_per_device=per_device * training_copies,
        mesh_size=mesh_size,
        largest_tensors=tuple(sorted(budgets, key=lambda item: item.bytes_per_device, reverse=True)[:largest]),
    )


def allocated_bytes_by_device(params: Any) -> dict[str, int]:
    """Measure physical parameter bytes after real sharded initialization."""
    totals: dict[str, int] = {}
    for value in traverse_util.flatten_dict(params).values():
        for shard in value.addressable_shards:
            device = str(shard.device)
            totals[device] = totals.get(device, 0) + shard.data.size * shard.data.dtype.itemsize
    return totals
