from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import optax
from jax.sharding import Mesh

from extent.sharding import (
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


@dataclass(frozen=True)
class ShardedOptimizerState:
    opt_state: Any
    layout: Any
    abstract_opt_state: Any


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


def optimizer_state_layout(
    tx: optax.GradientTransformation,
    abstract_params: Any,
    param_layout: Any,
    mesh: Mesh,
) -> tuple[Any, Any]:
    """Build layouts that shard optimizer moments like their parameters."""
    replicated = replicated_sharding(mesh)
    abstract_opt_state = jax.eval_shape(tx.init, abstract_params)
    layout_items = []
    for item in abstract_opt_state:
        if hasattr(item, "mu"):
            replacements = {"mu": param_layout}
            if hasattr(item, "nu"):
                replacements["nu"] = param_layout
            if hasattr(item, "count"):
                replacements["count"] = replicated
            layout_items.append(item._replace(**replacements))
        else:
            layout_items.append(jax.tree.map(lambda _: replicated, item))
    return abstract_opt_state, tuple(layout_items)


def initialize_sharded_optimizer_state(
    tx: optax.GradientTransformation,
    params: Any,
    abstract_params: Any,
    param_layout: Any,
    mesh: Mesh,
) -> ShardedOptimizerState:
    """Initialize optimizer state directly in shards without allocating gradients."""
    abstract_opt_state, layout = optimizer_state_layout(
        tx, abstract_params, param_layout, mesh
    )
    opt_state = jax.jit(
        tx.init,
        in_shardings=(param_layout,),
        out_shardings=layout,
    )(params)
    return ShardedOptimizerState(opt_state, layout, abstract_opt_state)
