"""Composition-aware recovery of a small trainable decoder segment."""
import jax
import jax.numpy as jnp
import optax

from extent.layerwise_distillation import relative_mse
from extent.optimizer import cast_grads_bf16, gradient_health


def make_block_recovery_step(forward, tx, *, bf16_gradients=True):
    """Differentiate through the entire segment, updating only candidate subtrees."""
    def step(params, opt_state, frozen, inputs, targets):
        def loss_fn(candidate):
            prediction = forward(candidate, frozen, inputs)
            return relative_mse(
                prediction.astype(jnp.float32) - inputs.astype(jnp.float32),
                targets.astype(jnp.float32) - inputs.astype(jnp.float32),
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        health = gradient_health(grads)
        updates, opt_state = tx.update(
            cast_grads_bf16(grads) if bf16_gradients else grads,
            opt_state, params,
        )
        return optax.apply_updates(params, updates), opt_state, {"loss": loss, **health}

    return step
