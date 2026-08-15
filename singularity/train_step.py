from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import optax
from jax.sharding import Mesh, NamedSharding

from singularity.model import causal_lm_loss
from singularity.optimizer import cast_grads_bf16
from singularity.sharding import (
    batch_sharding,
    named_sharding_tree,
    replicated_sharding,
    validate_partition_specs,
)


def create_train_state(model: nn.Module, params: optax.Params, tx: optax.GradientTransformation) -> TrainState:
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step() -> Callable:
    """Return a donation-friendly step; wrap with shard_map/pjit at the launcher."""
    @jax.jit
    def train_step(state: TrainState, batch: dict[str, jax.Array]):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            return causal_lm_loss(logits, batch.get("labels", batch["input_ids"]), batch.get("loss_mask"))

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = cast_grads_bf16(grads)
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss, "grad_norm": optax.tree.norm(grads)}

    return train_step


@dataclass(frozen=True)
class ShardedRuntime:
    state: TrainState
    train_step: Callable
    batch_layout: NamedSharding
    state_layout: Any


def initialize_sharded_runtime(
    model: nn.Module,
    tx: optax.GradientTransformation,
    rng: jax.Array,
    example_batch: dict[str, jax.Array],
    mesh: Mesh,
    *,
    donate_state: bool = True,
) -> ShardedRuntime:
    """Initialize parameters directly into global shards and compile one train step.

    ``example_batch`` must already be a global JAX array tree. Only its
    ``input_ids`` shape is used during parameter initialization.
    """
    batch_layout = batch_sharding(mesh)
    replicated = replicated_sharding(mesh)
    input_ids = example_batch["input_ids"]
    abstract_rng = jax.ShapeDtypeStruct(rng.shape, rng.dtype)
    abstract_tokens = jax.ShapeDtypeStruct(input_ids.shape, input_ids.dtype)
    abstract_variables = jax.eval_shape(model.init, abstract_rng, abstract_tokens)
    validate_partition_specs(abstract_variables["params"], mesh)
    param_layout = named_sharding_tree(abstract_variables["params"], mesh)

    init_model = jax.jit(
        model.init,
        in_shardings=(replicated, batch_layout),
        out_shardings={"params": param_layout},
    )
    variables = init_model(jax.device_put(rng, replicated), input_ids)

    def init_state(params):
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    abstract_state = jax.eval_shape(init_state, abstract_variables["params"])
    opt_layout_items = []
    for item in abstract_state.opt_state:
        if hasattr(item, "mu"):
            # optax.ScaleByLionState: momentum has exactly the parameter pytree.
            replacements = {"mu": param_layout}
            if hasattr(item, "count"):
                replacements["count"] = replicated
            opt_layout_items.append(item._replace(**replacements))
        else:
            opt_layout_items.append(jax.tree.map(lambda _: replicated, item))
    state_layout = abstract_state.replace(
        step=replicated,
        params=param_layout,
        opt_state=tuple(opt_layout_items),
    )
    # Explicit out layouts prevent Lion momentum from becoming fully replicated.
    state = jax.jit(
        init_state,
        in_shardings=(param_layout,),
        out_shardings=state_layout,
    )(variables["params"])
    batch_layout_tree = jax.tree.map(lambda _: batch_layout, example_batch)
    metric_layout = {"loss": replicated, "grad_norm": replicated}

    def distributed_step(current_state: TrainState, batch: dict[str, jax.Array]):
        def loss_fn(params):
            logits = current_state.apply_fn(
                {"params": params},
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            return causal_lm_loss(logits, batch["labels"], batch["loss_mask"])

        loss, grads = jax.value_and_grad(loss_fn)(current_state.params)
        grads = cast_grads_bf16(grads)
        new_state = current_state.apply_gradients(grads=grads)
        return new_state, {"loss": loss, "grad_norm": optax.tree.norm(grads)}

    compiled_step = jax.jit(
        distributed_step,
        in_shardings=(state_layout, batch_layout_tree),
        out_shardings=(state_layout, metric_layout),
        donate_argnums=(0,) if donate_state else (),
    )
    return ShardedRuntime(state, compiled_step, batch_layout, state_layout)


def shard_host_batch(batch: dict[str, Any], mesh: Mesh) -> dict[str, jax.Array]:
    """Place a small host batch as a global array replicated over model axes."""
    layout = batch_sharding(mesh)
    data_axis = mesh.shape["data"]
    output = {}
    for name, value in batch.items():
        value = jax.numpy.asarray(value)
        if value.shape[0] % data_axis:
            raise ValueError(f"{name} batch={value.shape[0]} must be divisible by data axis={data_axis}")
        output[name] = jax.device_put(value, layout)
    return output
