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


def apply_parameter_offset(
    params: optax.Params,
    offset: optax.Updates,
    scale: jax.Array | float,
) -> optax.Params:
    """Apply a differentiable, externally scheduled parameter-space prior."""
    return jax.tree.map(
        lambda parameter, delta: parameter
        + delta.astype(parameter.dtype) * jnp.asarray(scale, parameter.dtype),
        params,
        offset,
    )


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


def create_decoder_aware_train_step(
    apply_mixer: Callable[[optax.Params, jax.Array], jax.Array],
    apply_decoder_tail: Callable[[optax.Params, jax.Array, jax.Array], jax.Array],
    tx: optax.GradientTransformation,
    *,
    decoder_loss_weight: float,
    bf16_gradients: bool,
) -> Callable:
    """Train a mixer against both attention output and the frozen decoder tail."""
    if decoder_loss_weight < 0:
        raise ValueError("decoder loss weight must be non-negative")
    weight = jnp.asarray(decoder_loss_weight, dtype=jnp.float32)
    normalization = jnp.asarray(1.0 + decoder_loss_weight, dtype=jnp.float32)

    @jax.jit
    def train_step(
        params,
        opt_state,
        tail_params,
        residual_inputs,
        normalized_inputs,
        mixer_targets,
    ):
        decoder_targets = jax.lax.stop_gradient(
            apply_decoder_tail(tail_params, residual_inputs, mixer_targets)
        )

        def loss_fn(candidate):
            mixer_predictions = apply_mixer(candidate, normalized_inputs)
            decoder_predictions = apply_decoder_tail(
                tail_params, residual_inputs, mixer_predictions
            )
            mixer_loss = relative_mse(mixer_predictions, mixer_targets)
            decoder_loss = relative_mse(decoder_predictions, decoder_targets)
            objective = (mixer_loss + weight * decoder_loss) / normalization
            return objective, (mixer_loss, decoder_loss)

        (loss, (mixer_loss, decoder_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        health = gradient_health(grads)
        optimizer_grads = cast_grads_bf16(grads) if bf16_gradients else grads
        updates, opt_state = tx.update(optimizer_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {
            "loss": loss,
            "mixer_loss": mixer_loss,
            "decoder_loss": decoder_loss,
            **health,
        }

    return train_step


def create_layerwise_prior_train_step(
    apply_fn: Callable[[optax.Params, jax.Array], jax.Array],
    tx: optax.GradientTransformation,
    *,
    bf16_gradients: bool,
) -> Callable:
    """Build a Lion step whose effective parameters include a decaying prior."""

    @jax.jit
    def train_step(
        params,
        opt_state,
        inputs,
        targets,
        prior_offset,
        prior_scale,
    ):
        def loss_fn(candidate):
            effective = apply_parameter_offset(
                candidate, prior_offset, prior_scale
            )
            return relative_mse(apply_fn(effective, inputs), targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        health = gradient_health(grads)
        optimizer_grads = cast_grads_bf16(grads) if bf16_gradients else grads
        updates, opt_state = tx.update(optimizer_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {
            "loss": loss,
            "prior_scale": jnp.asarray(prior_scale, jnp.float32),
            **health,
        }
        return params, opt_state, metrics

    return train_step
