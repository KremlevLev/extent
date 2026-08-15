from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import traverse_util


def decay_mask(params: optax.Params) -> optax.Params:
    """Exclude norms, biases and recurrent stability scalars from weight decay."""
    flat = traverse_util.flatten_dict(params)
    excluded = {"scale", "bias", "dt_logit", "D", "b_bias", "c_bias"}
    return traverse_util.unflatten_dict({path: path[-1] not in excluded for path in flat})


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
        optax.clip_by_global_norm(max_grad_norm),
        optax.scale_by_lion(b1=0.9, b2=0.99, mu_dtype=jnp.bfloat16),
        optax.add_decayed_weights(weight_decay, mask=decay_mask),
        optax.scale_by_learning_rate(schedule),
    )
    if accumulation_steps > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=accumulation_steps, use_grad_mean=True)
    return optimizer


def cast_grads_bf16(grads: optax.Updates) -> optax.Updates:
    """Explicit contract requested for TPU experiments; call before ``tx.update``."""
    return jax.tree.map(
        lambda value: value.astype(jnp.bfloat16) if jnp.issubdtype(value.dtype, jnp.inexact) else value,
        grads,
    )
