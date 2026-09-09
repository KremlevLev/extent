"""Whole-model Qwen-to-hybrid distillation objectives for M3Q experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import optax

from extent.optimizer import cast_grads_bf16, gradient_health


def causal_cross_entropy(logits: jax.Array, token_ids: jax.Array) -> jax.Array:
    targets = token_ids[:, 1:]
    log_probabilities = jax.nn.log_softmax(
        logits[:, :-1].astype(jnp.float32), axis=-1
    )
    return -jnp.mean(
        jnp.take_along_axis(
            log_probabilities, targets[..., None], axis=-1
        ).squeeze(-1),
        dtype=jnp.float32,
    )


def forward_kl(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    *,
    temperature: float,
) -> jax.Array:
    """Teacher-to-student KL over next-token distributions in FP32."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scale = jnp.asarray(temperature, jnp.float32)
    teacher = teacher_logits[:, :-1].astype(jnp.float32) / scale
    student = student_logits[:, :-1].astype(jnp.float32) / scale
    teacher_log_probs = jax.nn.log_softmax(teacher, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)
    student_log_probs = jax.nn.log_softmax(student, axis=-1)
    token_kl = jnp.sum(
        teacher_probs * (teacher_log_probs - student_log_probs), axis=-1
    )
    return jnp.mean(token_kl, dtype=jnp.float32) * jnp.square(scale)


def hidden_state_relative_mse(
    student_states: Sequence[jax.Array],
    teacher_states: Sequence[jax.Array],
    layer_indices: Sequence[int],
) -> jax.Array:
    """Mean scale-free hidden loss over the registered replaced layers."""
    if not layer_indices:
        raise ValueError("hidden alignment requires at least one layer")
    losses = []
    for index in layer_indices:
        error = (
            student_states[index].astype(jnp.float32)
            - teacher_states[index].astype(jnp.float32)
        )
        target = teacher_states[index].astype(jnp.float32)
        losses.append(
            jnp.sum(jnp.square(error), dtype=jnp.float32)
            / jnp.maximum(
                jnp.sum(jnp.square(target), dtype=jnp.float32),
                jnp.finfo(jnp.float32).tiny,
            )
        )
    return jnp.mean(jnp.stack(losses), dtype=jnp.float32)


def hidden_delta_relative_mse(
    student_states: Sequence[jax.Array],
    teacher_states: Sequence[jax.Array],
    layer_indices: Sequence[int],
) -> jax.Array:
    """Match decoder-block contributions instead of accumulated residual states.

    For every layer after layer zero, the target is ``h_l - h_(l-1)``.  This
    prevents the already-matched residual stream from dominating the loss and
    is the full-model analogue of the successful local contribution objective
    used by the sequential M3Q recovery campaigns.  Layer zero falls back to
    its output state because the embedding state is intentionally not exposed
    by the model API.
    """
    if not layer_indices:
        raise ValueError("hidden delta alignment requires at least one layer")
    losses = []
    for index in layer_indices:
        if index < 0 or index >= len(student_states):
            raise ValueError(f"hidden layer index is out of range: {index}")
        if index == 0:
            student_delta = student_states[index].astype(jnp.float32)
            teacher_delta = teacher_states[index].astype(jnp.float32)
        else:
            student_delta = (
                student_states[index].astype(jnp.float32)
                - student_states[index - 1].astype(jnp.float32)
            )
            teacher_delta = (
                teacher_states[index].astype(jnp.float32)
                - teacher_states[index - 1].astype(jnp.float32)
            )
        losses.append(
            jnp.sum(jnp.square(student_delta - teacher_delta), dtype=jnp.float32)
            / jnp.maximum(
                jnp.sum(jnp.square(teacher_delta), dtype=jnp.float32),
                jnp.finfo(jnp.float32).tiny,
            )
        )
    return jnp.mean(jnp.stack(losses), dtype=jnp.float32)


def make_prediction_distill_step(
    student_apply: Callable,
    teacher_apply: Callable,
    tx: optax.GradientTransformation,
    *,
    temperature: float,
    cross_entropy_weight: float,
    bf16_gradients: bool,
    trainable_mask=None,
) -> Callable:
    """Create the standard exact-init end-to-end KL baseline step."""
    if cross_entropy_weight < 0:
        raise ValueError("cross entropy weight must be non-negative")

    def step(student_params, opt_state, teacher_params, token_ids):
        teacher_logits = jax.lax.stop_gradient(
            teacher_apply(teacher_params, token_ids, False)[0]
        )

        def loss_fn(candidate):
            student_logits, _ = student_apply(candidate, token_ids, False)
            kl = forward_kl(
                student_logits, teacher_logits, temperature=temperature
            )
            cross_entropy = causal_cross_entropy(student_logits, token_ids)
            loss = kl + cross_entropy_weight * cross_entropy
            return loss, (kl, cross_entropy)

        (loss, (kl, cross_entropy)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(student_params)
        if trainable_mask is not None:
            grads = jax.tree.map(
                lambda grad, trainable: (
                    grad if trainable else jnp.zeros_like(grad)
                ),
                grads,
                trainable_mask,
            )
        health = gradient_health(grads)
        updates, opt_state = tx.update(
            cast_grads_bf16(grads) if bf16_gradients else grads,
            opt_state,
            student_params,
        )
        student_params = optax.apply_updates(student_params, updates)
        return student_params, opt_state, {
            "loss": loss,
            "prediction_kl": kl,
            "cross_entropy": cross_entropy,
            "hidden_loss": jnp.asarray(0.0, jnp.float32),
            **health,
        }

    return step


def make_hidden_bridge_distill_step(
    student_apply: Callable,
    teacher_apply: Callable,
    tx: optax.GradientTransformation,
    *,
    layer_indices: Sequence[int],
    temperature: float,
    cross_entropy_weight: float,
    prediction_weight: float,
    hidden_weight: float,
    bf16_gradients: bool,
    hidden_mode: str = "state",
    trainable_mask=None,
) -> Callable:
    """Create M3Q's joint hidden-state bridge and prediction-KL stage."""
    if min(cross_entropy_weight, prediction_weight, hidden_weight) < 0:
        raise ValueError("distillation weights must be non-negative")
    if hidden_mode not in {"state", "delta"}:
        raise ValueError("hidden_mode must be 'state' or 'delta'")
    selected_layers = tuple(int(index) for index in layer_indices)
    hidden_objective = (
        hidden_state_relative_mse
        if hidden_mode == "state"
        else hidden_delta_relative_mse
    )

    def step(student_params, opt_state, teacher_params, token_ids):
        teacher_logits, teacher_states = teacher_apply(
            teacher_params, token_ids, True
        )
        teacher_logits = jax.lax.stop_gradient(teacher_logits)
        teacher_states = jax.tree.map(jax.lax.stop_gradient, teacher_states)

        def loss_fn(candidate):
            student_logits, student_states = student_apply(
                candidate, token_ids, True
            )
            kl = forward_kl(
                student_logits, teacher_logits, temperature=temperature
            )
            cross_entropy = causal_cross_entropy(student_logits, token_ids)
            hidden = hidden_objective(
                student_states, teacher_states, selected_layers
            )
            loss = (
                prediction_weight * kl
                + hidden_weight * hidden
                + cross_entropy_weight * cross_entropy
            )
            return loss, (kl, cross_entropy, hidden)

        (loss, (kl, cross_entropy, hidden)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(student_params)
        if trainable_mask is not None:
            grads = jax.tree.map(
                lambda grad, trainable: (
                    grad if trainable else jnp.zeros_like(grad)
                ),
                grads,
                trainable_mask,
            )
        health = gradient_health(grads)
        updates, opt_state = tx.update(
            cast_grads_bf16(grads) if bf16_gradients else grads,
            opt_state,
            student_params,
        )
        student_params = optax.apply_updates(student_params, updates)
        return student_params, opt_state, {
            "loss": loss,
            "prediction_kl": kl,
            "cross_entropy": cross_entropy,
            "hidden_loss": hidden,
            **health,
        }

    return step


def full_model_eval_metrics(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    token_ids: jax.Array,
    *,
    temperature: float,
) -> dict[str, jax.Array]:
    student_nll = causal_cross_entropy(student_logits, token_ids)
    teacher_nll = causal_cross_entropy(teacher_logits, token_ids)
    kl = forward_kl(student_logits, teacher_logits, temperature=temperature)
    agreement = jnp.mean(
        jnp.argmax(student_logits[:, :-1], axis=-1)
        == jnp.argmax(teacher_logits[:, :-1], axis=-1),
        dtype=jnp.float32,
    )
    return {
        "student_nll": student_nll,
        "teacher_nll": teacher_nll,
        "excess_nll": student_nll - teacher_nll,
        "prediction_kl": kl,
        "top1_agreement": agreement,
        "finite": jnp.all(jnp.asarray([
            jnp.isfinite(student_nll),
            jnp.isfinite(teacher_nll),
            jnp.isfinite(kl),
        ])),
    }
