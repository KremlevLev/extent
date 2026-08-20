from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from extent.optimizer import cast_grads_bf16, gradient_health


def create_teacher_mixer_runner(module, positions, attention_mask) -> Callable:
    """Compile a Flax teacher apply without crossing JIT with NumPy helpers."""

    @jax.jit
    def run(params, inputs):
        return module.apply(
            {"params": params}, inputs, positions, attention_mask
        )

    return run


def relative_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Scale-free mixer distillation objective, accumulated in FP32."""
    error = prediction.astype(jnp.float32) - target.astype(jnp.float32)
    numerator = jnp.sum(jnp.square(error), dtype=jnp.float32)
    denominator = jnp.sum(
        jnp.square(target.astype(jnp.float32)), dtype=jnp.float32
    )
    return numerator / jnp.maximum(denominator, jnp.finfo(jnp.float32).tiny)


def create_layerwise_train_step(
    apply_fn: Callable[[optax.Params, jax.Array], jax.Array],
    tx: optax.GradientTransformation,
    *,
    bf16_gradients: bool,
) -> Callable:
    """Build one compiled Lion step for a trainable replacement mixer."""

    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        def loss_fn(candidate):
            return relative_mse(apply_fn(candidate, inputs), targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        health = gradient_health(grads)
        optimizer_grads = cast_grads_bf16(grads) if bf16_gradients else grads
        updates, opt_state = tx.update(optimizer_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {"loss": loss, **health}
        return params, opt_state, metrics

    return train_step
