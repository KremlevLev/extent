"""Exact-dual, Mamba-3-aware bridge training primitives.

The bridge deliberately uses the canonical Mamba-3 parameter tree in its
quadratic SSD-dual execution mode.  The deployable recurrent module consumes
the same tree, so the bridge-to-Mamba transition is an execution-mode switch,
not another learned projection or weight conversion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
import optax

from extent.config import Mamba3Config
from extent.mamba3_transplant import mamba3_projection_slices
from extent.optimizer import cast_grads_bf16, gradient_health


def zero_complex_projection(
    params: Mapping, hidden_size: int, config: Mamba3Config
) -> Mapping:
    """Return the same parameter tree with the input-dependent phase disabled."""
    was_frozen = isinstance(params, FrozenDict)
    mutable = unfreeze(params) if was_frozen else unfreeze(freeze(params))
    angle = mamba3_projection_slices(hidden_size, config)["angle"]
    kernel = mutable["in_proj"]["kernel"]
    mutable["in_proj"]["kernel"] = kernel.at[:, angle].set(0)
    return freeze(mutable) if was_frozen else mutable


def _mask_complex_gradient(
    grads: Mapping, hidden_size: int, config: Mamba3Config
) -> Mapping:
    was_frozen = isinstance(grads, FrozenDict)
    mutable = unfreeze(grads) if was_frozen else unfreeze(freeze(grads))
    angle = mamba3_projection_slices(hidden_size, config)["angle"]
    kernel = mutable["in_proj"]["kernel"]
    mutable["in_proj"]["kernel"] = kernel.at[:, angle].set(0)
    return freeze(mutable) if was_frozen else mutable


def scale_free_output_loss(prediction: jax.Array, target: jax.Array) -> jax.Array:
    prediction = prediction.astype(jnp.float32)
    target = target.astype(jnp.float32)
    squared_error = jnp.sum(jnp.square(prediction - target))
    target_energy = jnp.maximum(
        jnp.sum(jnp.square(target)), jnp.finfo(jnp.float32).tiny
    )
    cosine_denominator = jnp.maximum(
        jnp.linalg.norm(prediction) * jnp.linalg.norm(target),
        jnp.finfo(jnp.float32).tiny,
    )
    relative_mse = squared_error / target_energy
    cosine = 1.0 - jnp.vdot(prediction.reshape(-1), target.reshape(-1)).real / cosine_denominator
    return relative_mse + cosine


def token_whitened_output_loss(
    prediction: jax.Array, target: jax.Array
) -> jax.Array:
    """Equalize token contributions using frozen teacher RMS scales.

    Deep Qwen layers showed million-fold differences in uncalibrated bridge
    loss despite RMS-normalized inputs.  Dividing both sides by the teacher's
    per-token output RMS preserves direction and relative error while stopping
    a few high-energy tokens from defining the construction objective.
    """
    prediction = prediction.astype(jnp.float32)
    target = target.astype(jnp.float32)
    scale = jax.lax.stop_gradient(
        jnp.sqrt(jnp.mean(jnp.square(target), axis=-1, keepdims=True) + 1e-6)
    )
    return scale_free_output_loss(prediction / scale, target / scale)


def create_exact_dual_bridge_train_step(
    apply_dual: Callable,
    tx: optax.GradientTransformation,
    *,
    hidden_size: int,
    config: Mamba3Config,
    bf16_gradients: bool,
    freeze_complex: bool = False,
    loss_mode: str = "global_scale_free",
) -> Callable:
    """Train canonical Mamba-3 parameters in their exact quadratic dual form."""
    if loss_mode not in {"global_scale_free", "token_whitened"}:
        raise ValueError("unknown exact-dual bridge loss mode")
    bridge_loss = (
        scale_free_output_loss
        if loss_mode == "global_scale_free"
        else token_whitened_output_loss
    )

    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        def loss_fn(candidate):
            prediction = apply_dual(candidate, inputs)
            return bridge_loss(prediction, targets), prediction

        (loss, prediction), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        if freeze_complex:
            grads = _mask_complex_gradient(grads, hidden_size, config)
        health = gradient_health(grads)
        optimizer_grads = cast_grads_bf16(grads) if bf16_gradients else grads
        updates, opt_state = tx.update(optimizer_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if freeze_complex:
            params = zero_complex_projection(params, hidden_size, config)
        relative_l2 = jnp.linalg.norm(
            prediction.astype(jnp.float32) - targets.astype(jnp.float32)
        ) / jnp.maximum(
            jnp.linalg.norm(targets.astype(jnp.float32)),
            jnp.finfo(jnp.float32).tiny,
        )
        return params, opt_state, {
            "loss": loss,
            "relative_l2": relative_l2,
            **health,
        }

    return train_step


def dual_recurrent_parity(
    dual_apply: Callable,
    recurrent_apply: Callable,
    params: Mapping,
    inputs: jax.Array,
) -> dict[str, float | bool]:
    """Measure the required zero-shock bridge-to-recurrence transition."""
    dual = np.asarray(dual_apply(params, inputs), np.float32)
    recurrent = np.asarray(recurrent_apply(params, inputs), np.float32)
    difference = dual - recurrent
    denominator = max(float(np.linalg.norm(dual)), np.finfo(np.float32).tiny)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
        "finite": bool(np.all(np.isfinite(dual)) and np.all(np.isfinite(recurrent))),
    }
