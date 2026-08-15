from __future__ import annotations

from collections.abc import Callable

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import optax

from singularity.model import causal_lm_loss
from singularity.optimizer import cast_grads_bf16


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
        return state, {"loss": loss, "grad_norm": optax.global_norm(grads)}

    return train_step
