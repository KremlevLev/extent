from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from flax.core import FrozenDict, freeze


def decay_mask(params: optax.Params) -> optax.Params:
    """Exclude norms, biases and recurrent stability scalars from weight decay."""
    flat = traverse_util.flatten_dict(params)
    excluded = {"scale", "bias", "dt_bias", "D", "b_bias", "c_bias"}
    mask = traverse_util.unflatten_dict(
        {path: path[-1] not in excluded for path in flat}
    )
    return freeze(mask) if isinstance(params, FrozenDict) else mask


def global_norm_fp32(tree: optax.Updates) -> jax.Array:
    """FP32 global L2 norm, with scaled reduction on sum-of-squares overflow.

    Preserve the original reduction for ordinary finite sums, including resumed
    runs. Large finite gradients can overflow that sum while their norm is still
    representable. Scaling before squaring handles this without hiding NaN/Inf
    gradients or changing the intended global clipping threshold.
    """
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.array(0.0, jnp.float32)
    squared = [jnp.sum(jnp.square(value.astype(jnp.float32))) for value in leaves]
    total = jnp.sum(jnp.stack(squared), dtype=jnp.float32)

    def scaled_norm(_):
        maximum = jnp.max(jnp.stack([
            jnp.max(jnp.abs(value.astype(jnp.float32)), initial=0.0)
            for value in leaves
        ]))
        denominator = jnp.where(maximum > 0, maximum, 1.0)
        scaled_squared = [
            jnp.sum(jnp.square(value.astype(jnp.float32) / denominator))
            for value in leaves
        ]
        return maximum * jnp.sqrt(jnp.sum(jnp.stack(scaled_squared), dtype=jnp.float32))

    return jax.lax.cond(jnp.isfinite(total), lambda _: jnp.sqrt(total), scaled_norm, None)


def clip_by_global_norm_fp32(max_norm: float) -> optax.GradientTransformation:
    """Optax-compatible clipping that never reduces squared gradients in BF16."""
    def init_fn(_):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        norm = global_norm_fp32(updates)
        scale = jnp.minimum(1.0, jnp.asarray(max_norm, jnp.float32) / jnp.maximum(norm, 1e-12))
        clipped = jax.tree.map(lambda value: value * scale.astype(value.dtype), updates)
        return clipped, state

    return optax.GradientTransformation(init_fn, update_fn)


def gradient_health(grads: optax.Updates) -> dict[str, jax.Array]:
    """Small replicated diagnostics used to distinguish norm issues from NaNs."""
    leaves = jax.tree.leaves(grads)
    finite_by_leaf = jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])
    max_by_leaf = jnp.stack([
        jnp.max(jnp.where(jnp.isfinite(value), jnp.abs(value.astype(jnp.float32)), 0.0))
        for value in leaves
    ])
    return {
        "grad_norm": global_norm_fp32(grads),
        "grads_finite": jnp.all(finite_by_leaf),
        "nonfinite_grad_leaves": jnp.sum(~finite_by_leaf, dtype=jnp.int32),
        "max_abs_grad": jnp.max(max_by_leaf),
    }


def gradient_health_tree(grads: optax.Updates) -> dict[str, optax.Updates]:
    """Per-parameter diagnostics, materialized only after a failed smoke step."""
    return {
        "finite": jax.tree.map(lambda value: jnp.all(jnp.isfinite(value)), grads),
        "max_abs": jax.tree.map(
            lambda value: jnp.max(
                jnp.where(jnp.isfinite(value), jnp.abs(value.astype(jnp.float32)), 0.0)
            ),
            grads,
        ),
    }


def create_lion(
    learning_rate: float = 1e-4,
    warmup_steps: int = 2_000,
    total_steps: int = 100_000,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
) -> optax.GradientTransformation:
    """Memory-aware Lion: one BF16 momentum buffer and BF16 accumulated grads."""
    warmup_steps = min(warmup_steps, total_steps)
    schedule: Callable[[jax.Array], jax.Array] = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.1,
    )
    optimizer = optax.chain(
        clip_by_global_norm_fp32(max_grad_norm),
        optax.scale_by_lion(b1=0.9, b2=0.99, mu_dtype=jnp.bfloat16),
        optax.add_decayed_weights(weight_decay, mask=decay_mask),
        optax.scale_by_learning_rate(schedule),
    )
    if accumulation_steps > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=accumulation_steps, use_grad_mean=True)
    return optimizer


def create_adamw(
    learning_rate: float = 1e-4,
    warmup_steps: int = 2_000,
    total_steps: int = 100_000,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
) -> optax.GradientTransformation:
    """Memory-aware AdamW control with BF16 moments for TPU ablations."""
    warmup_steps = min(warmup_steps, total_steps)
    schedule: Callable[[jax.Array], jax.Array] = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.1,
    )
    optimizer = optax.chain(
        clip_by_global_norm_fp32(max_grad_norm),
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, mu_dtype=jnp.bfloat16),
        optax.add_decayed_weights(weight_decay, mask=decay_mask),
        optax.scale_by_learning_rate(schedule),
    )
    if accumulation_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=accumulation_steps, use_grad_mean=True
        )
    return optimizer


def cast_grads_bf16(grads: optax.Updates) -> optax.Updates:
    """Explicit contract requested for TPU experiments; call before ``tx.update``."""
    return jax.tree.map(
        lambda value: value.astype(jnp.bfloat16) if jnp.issubdtype(value.dtype, jnp.inexact) else value,
        grads,
    )
