from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
from jax.sharding import Mesh

from singularity.sharding import (
    batch_sharding,
    named_sharding_tree,
    replicated_sharding,
    validate_partition_specs,
)


@dataclass(frozen=True)
class ShardedParameters:
    params: Any
    layout: Any
    abstract_params: Any


def abstract_parameter_tree(
    model: nn.Module,
    *,
    batch_size: int = 1,
    sequence_length: int = 1,
) -> Any:
    """Trace parameter shapes without allocating model weights or activations."""
    tokens = jax.ShapeDtypeStruct((batch_size, sequence_length), jax.numpy.int32)
    variables = jax.eval_shape(model.init, jax.random.key(0), tokens)
    return variables["params"]


def initialize_sharded_parameters(
    model: nn.Module,
    rng: jax.Array,
    input_ids: jax.Array,
    mesh: Mesh,
) -> ShardedParameters:
    """Create parameters directly in their final global device shards."""
    abstract_params = abstract_parameter_tree(
        model,
        batch_size=input_ids.shape[0],
        sequence_length=input_ids.shape[1],
    )
    validate_partition_specs(abstract_params, mesh)
    layout = named_sharding_tree(abstract_params, mesh)
    init_model = jax.jit(
        model.init,
        in_shardings=(replicated_sharding(mesh), batch_sharding(mesh)),
        out_shardings={"params": layout},
    )
    variables = init_model(
        jax.device_put(rng, replicated_sharding(mesh)),
        input_ids,
    )
    return ShardedParameters(variables["params"], layout, abstract_params)
