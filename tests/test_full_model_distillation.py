import jax
import jax.numpy as jnp
import numpy as np
import optax

from extent.full_model_distillation import (
    causal_cross_entropy,
    forward_kl,
    full_model_eval_metrics,
    hidden_delta_relative_mse,
    hidden_state_relative_mse,
    make_hidden_bridge_distill_step,
    make_prediction_distill_step,
)


def _apply(params, tokens, return_hidden):
    hidden = tokens[..., None].astype(jnp.float32) * params["scale"]
    logits = jnp.concatenate((hidden, -hidden, hidden * 0.5), axis=-1)
    states = (hidden, hidden * 2.0)
    return logits, states if return_hidden else ()


def test_full_model_losses_are_exact_for_identical_teacher():
    tokens = jnp.asarray([[0, 1, 2, 1]], jnp.int32)
    logits, states = _apply({"scale": jnp.asarray(1.0)}, tokens, True)
    assert float(forward_kl(logits, logits, temperature=2.0)) < 1e-6
    assert float(hidden_state_relative_mse(states, states, (0, 1))) == 0.0
    assert float(hidden_delta_relative_mse(states, states, (0, 1))) == 0.0
    assert np.isfinite(float(causal_cross_entropy(logits, tokens)))
    metrics = full_model_eval_metrics(
        logits, logits, tokens, temperature=2.0
    )
    assert float(metrics["excess_nll"]) == 0.0
    assert float(metrics["top1_agreement"]) == 1.0


def test_prediction_and_hidden_bridge_steps_update_only_student():
    student = {"scale": jnp.asarray(0.5, jnp.float32)}
    teacher = {"scale": jnp.asarray(1.0, jnp.float32)}
    tokens = jnp.asarray([[0, 1, 2, 1]], jnp.int32)
    tx = optax.sgd(1e-2)
    prediction_step = jax.jit(make_prediction_distill_step(
        _apply, _apply, tx, temperature=1.0,
        cross_entropy_weight=0.1, bf16_gradients=False,
    ))
    hidden_step = jax.jit(make_hidden_bridge_distill_step(
        _apply, _apply, tx, layer_indices=(0, 1), temperature=1.0,
        cross_entropy_weight=0.1, prediction_weight=1.0,
        hidden_weight=1.0, bf16_gradients=False,
    ))

    predicted, _, prediction_metrics = prediction_step(
        student, tx.init(student), teacher, tokens
    )
    bridged, _, bridge_metrics = hidden_step(
        student, tx.init(student), teacher, tokens
    )
    jax.block_until_ready((prediction_metrics, bridge_metrics))
    assert bool(prediction_metrics["grads_finite"])
    assert bool(bridge_metrics["grads_finite"])
    assert float(prediction_metrics["hidden_loss"]) == 0.0
    assert float(bridge_metrics["hidden_loss"]) > 0.0
    assert float(predicted["scale"]) != float(student["scale"])
    assert float(bridged["scale"]) != float(student["scale"])
    assert float(teacher["scale"]) == 1.0


def test_prediction_step_respects_static_trainable_mask():
    student = {
        "train": jnp.asarray(0.5, jnp.float32),
        "frozen": jnp.asarray(0.5, jnp.float32),
    }
    teacher = {"train": jnp.asarray(1.0), "frozen": jnp.asarray(1.0)}

    def apply(params, tokens, return_hidden):
        scale = params["train"] + params["frozen"]
        hidden = tokens[..., None].astype(jnp.float32) * scale
        logits = jnp.concatenate((hidden, -hidden, hidden * 0.5), axis=-1)
        return logits, ()

    tokens = jnp.asarray([[0, 1, 2, 1]], jnp.int32)
    tx = optax.sgd(1e-2)
    step = make_prediction_distill_step(
        apply, apply, tx, temperature=1.0, cross_entropy_weight=0.1,
        bf16_gradients=False, trainable_mask={"train": True, "frozen": False},
    )
    updated, _, _ = step(student, tx.init(student), teacher, tokens)
    assert float(updated["train"]) != float(student["train"])
    assert float(updated["frozen"]) == float(student["frozen"])


def test_hidden_delta_is_block_contribution_not_residual_error():
    teacher = (
        jnp.asarray([[[10.0]]]),
        jnp.asarray([[[12.0]]]),
        jnp.asarray([[[15.0]]]),
    )
    student = (
        jnp.asarray([[[20.0]]]),
        jnp.asarray([[[22.0]]]),
        jnp.asarray([[[26.0]]]),
    )
    # Layer one contributes +2 in both networks despite different residuals.
    assert float(hidden_delta_relative_mse(student, teacher, (1,))) == 0.0
    assert float(hidden_state_relative_mse(student, teacher, (1,))) > 0.0
    # Layer two contributions differ (+4 versus +3).
    assert np.isclose(
        float(hidden_delta_relative_mse(student, teacher, (2,))), 1.0 / 9.0
    )


def test_hidden_bridge_step_respects_static_trainable_mask():
    student = {
        "train": jnp.asarray(0.5, jnp.float32),
        "frozen": jnp.asarray(0.5, jnp.float32),
    }
    teacher = {"train": jnp.asarray(1.0), "frozen": jnp.asarray(1.0)}

    def apply(params, tokens, return_hidden):
        scale = params["train"] + params["frozen"]
        hidden = tokens[..., None].astype(jnp.float32) * scale
        logits = jnp.concatenate((hidden, -hidden, hidden * 0.5), axis=-1)
        states = (hidden, hidden * 2.0)
        return logits, states if return_hidden else ()

    tokens = jnp.asarray([[0, 1, 2, 1]], jnp.int32)
    tx = optax.sgd(1e-2)
    step = make_hidden_bridge_distill_step(
        apply, apply, tx, layer_indices=(0, 1), temperature=1.0,
        cross_entropy_weight=0.0, prediction_weight=0.0,
        hidden_weight=1.0, bf16_gradients=False, hidden_mode="delta",
        trainable_mask={"train": True, "frozen": False},
    )
    updated, _, metrics = step(student, tx.init(student), teacher, tokens)
    assert float(updated["train"]) != float(student["train"])
    assert float(updated["frozen"]) == float(student["frozen"])
    assert float(metrics["hidden_loss"]) > 0.0
