"""Equal-budget local recovery on teacher, hybrid, or mixed residual streams."""
from __future__ import annotations

from collections.abc import Mapping
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict, freeze, unfreeze

from extent.layerwise_distillation import relative_mse
from extent.optimizer import cast_grads_bf16, gradient_health


def replace_mamba(layer_params: Mapping, mamba_params: Mapping):
    mutable = unfreeze(layer_params) if isinstance(layer_params, FrozenDict) else dict(layer_params)
    mutable["mamba"] = mamba_params
    return freeze(mutable) if isinstance(layer_params, FrozenDict) else mutable


def make_conditional_recovery_step(student_layer, tx, *, bf16_gradients=True):
    """Train only Mamba against frozen conditional teacher decoder outputs."""
    def step(mamba_params, opt_state, frozen_layer_params, residual_inputs, decoder_targets):
        def loss_fn(candidate):
            prediction = student_layer.apply(
                {"params": replace_mamba(frozen_layer_params, candidate)},
                residual_inputs,
                jnp.arange(residual_inputs.shape[1], dtype=jnp.int32)[None],
                None,
            )
            # Remove the shared residual identity before normalization. Without
            # this, a large input stream can make a poor learned contribution
            # look artificially accurate (the exact pathology exposed by EXP-070).
            return relative_mse(
                prediction.astype(jnp.float32) - residual_inputs.astype(jnp.float32),
                decoder_targets.astype(jnp.float32) - residual_inputs.astype(jnp.float32),
            )
        loss, grads = jax.value_and_grad(loss_fn)(mamba_params)
        health = gradient_health(grads)
        updates, opt_state = tx.update(
            cast_grads_bf16(grads) if bf16_gradients else grads,
            opt_state, mamba_params,
        )
        return optax.apply_updates(mamba_params, updates), opt_state, {"loss": loss, **health}
    return step


def recovery_examples(arm, teacher_inputs, teacher_targets, hybrid_inputs, hybrid_targets):
    if arm == "TEACHER":
        return teacher_inputs, teacher_targets
    if arm == "ONPOLICY":
        return hybrid_inputs, hybrid_targets
    if arm == "MIXED":
        return (jnp.concatenate((teacher_inputs, hybrid_inputs), axis=0),
                jnp.concatenate((teacher_targets, hybrid_targets), axis=0))
    raise ValueError(f"unknown recovery arm: {arm}")
