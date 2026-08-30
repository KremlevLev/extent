from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze

from extent.layerwise_distillation import (
    create_counterfactual_contribution_train_step,
    create_decoder_aware_train_step,
    create_homotopy_decoder_train_step,
    apply_parameter_offset,
    create_layerwise_prior_train_step,
    create_layerwise_train_step,
    create_teacher_mixer_runner,
    relative_mse,
)
from extent.qwen3_teacher import Qwen3GQAAttention, tiny_qwen3_teacher_config
from extent.optimizer import create_lion
from scripts.qwen_mamba3_distill_pilot import (
    _json_default,
    _write_json,
    _write_json_with_output_mirror,
)


def test_relative_mse_is_scale_free_and_zero_for_exact_target():
    target = jnp.asarray([[1.0, -2.0]], jnp.float32)
    assert float(relative_mse(target, target)) == 0.0
    np.testing.assert_allclose(relative_mse(jnp.zeros_like(target), target), 1.0)


def test_layerwise_train_step_updates_parameters_and_reports_finite_gradients():
    params = {"kernel": jnp.eye(3, dtype=jnp.float32)}
    inputs = jnp.asarray([[1.0, 2.0, 3.0]], jnp.float32)
    targets = 2.0 * inputs
    apply_fn = lambda candidate, value: value @ candidate["kernel"]
    tx = optax.lion(learning_rate=1e-2)
    step = create_layerwise_train_step(apply_fn, tx, bf16_gradients=False)
    updated, _, metrics = step(params, tx.init(params), inputs, targets)
    jax.block_until_ready(metrics)
    assert bool(metrics["grads_finite"])
    assert np.isfinite(float(metrics["grad_norm"]))
    assert not np.array_equal(np.asarray(updated["kernel"]), np.asarray(params["kernel"]))


def test_decoder_aware_step_reports_both_objectives_and_updates_only_mixer():
    params = {"kernel": jnp.eye(2, dtype=jnp.float32)}
    tail_params = {"scale": jnp.asarray(2.0, dtype=jnp.float32)}
    residual = jnp.asarray([[1.0, -1.0]], dtype=jnp.float32)
    normalized = jnp.asarray([[2.0, 1.0]], dtype=jnp.float32)
    mixer_target = jnp.asarray([[0.5, -0.25]], dtype=jnp.float32)
    apply_mixer = lambda candidate, value: value @ candidate["kernel"]
    apply_tail = (
        lambda frozen, residual_value, mixer_value: residual_value
        + frozen["scale"] * mixer_value
    )
    tx = optax.sgd(learning_rate=1e-2)
    step = create_decoder_aware_train_step(
        apply_mixer,
        apply_tail,
        tx,
        decoder_loss_weight=1.0,
        bf16_gradients=False,
    )
    updated, _, metrics = step(
        params,
        tx.init(params),
        tail_params,
        residual,
        normalized,
        mixer_target,
    )
    jax.block_until_ready(metrics)
    assert bool(metrics["grads_finite"])
    assert float(metrics["mixer_loss"]) > 0
    assert float(metrics["decoder_loss"]) > 0
    np.testing.assert_allclose(
        metrics["loss"],
        0.5 * (metrics["mixer_loss"] + metrics["decoder_loss"]),
    )
    assert not np.array_equal(updated["kernel"], params["kernel"])
    np.testing.assert_array_equal(tail_params["scale"], jnp.asarray(2.0))


def test_decoder_aware_step_rejects_negative_decoder_weight():
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        create_decoder_aware_train_step(
            lambda params, value: value,
            lambda params, residual, mixer: residual + mixer,
            optax.sgd(1e-2),
            decoder_loss_weight=-1.0,
            bf16_gradients=False,
        )


def test_homotopy_step_trains_mixer_and_reaches_deployable_decoder_path():
    params = {"kernel": jnp.eye(2, dtype=jnp.float32)}
    tail_params = {"scale": jnp.asarray(2.0, dtype=jnp.float32)}
    residual = jnp.asarray([[1.0, -1.0]], dtype=jnp.float32)
    normalized = jnp.asarray([[2.0, 1.0]], dtype=jnp.float32)
    mixer_target = jnp.asarray([[0.5, -0.25]], dtype=jnp.float32)
    apply_mixer = lambda candidate, value: value @ candidate["kernel"]
    apply_tail = lambda frozen, residual_value, mixer_value: (
        residual_value + frozen["scale"] * mixer_value
    )
    tx = optax.sgd(learning_rate=1e-2)
    step = create_homotopy_decoder_train_step(
        apply_mixer,
        apply_tail,
        tx,
        decoder_loss_weight=1.0,
        bf16_gradients=False,
    )

    _, _, teacher_bridge = step(
        params, tx.init(params), tail_params, residual, normalized, mixer_target, 0.0
    )
    updated, _, deployable = step(
        params, tx.init(params), tail_params, residual, normalized, mixer_target, 1.0
    )
    jax.block_until_ready((teacher_bridge, deployable))

    assert float(teacher_bridge["homotopy_decoder_loss"]) == 0.0
    assert float(deployable["homotopy_decoder_loss"]) > 0.0
    assert float(deployable["homotopy_alpha"]) == 1.0
    assert bool(deployable["grads_finite"])
    assert not np.array_equal(updated["kernel"], params["kernel"])


def test_counterfactual_contribution_removes_residual_baseline():
    params = {"kernel": jnp.eye(2, dtype=jnp.float32)}
    tail_params = {"scale": jnp.asarray(3.0, dtype=jnp.float32)}
    residual = jnp.asarray([[1000.0, -1000.0]], dtype=jnp.float32)
    normalized = jnp.asarray([[2.0, 1.0]], dtype=jnp.float32)
    mixer_target = jnp.asarray([[0.5, -0.25]], dtype=jnp.float32)
    apply_mixer = lambda candidate, value: value @ candidate["kernel"]
    apply_tail = (
        lambda frozen, residual_value, mixer_value: residual_value
        + frozen["scale"] * mixer_value
    )
    tx = optax.sgd(learning_rate=1e-2)
    step = create_counterfactual_contribution_train_step(
        apply_mixer,
        apply_tail,
        tx,
        mixer_loss_weight=1.0,
        bf16_gradients=False,
    )
    updated, _, metrics = step(
        params,
        tx.init(params),
        tail_params,
        residual,
        normalized,
        mixer_target,
    )
    jax.block_until_ready(metrics)
    assert bool(metrics["grads_finite"])
    assert float(metrics["contribution_loss"]) > 0
    np.testing.assert_allclose(
        metrics["loss"],
        0.5 * (metrics["mixer_loss"] + metrics["contribution_loss"]),
    )
    assert not np.array_equal(updated["kernel"], params["kernel"])


def test_counterfactual_contribution_rejects_negative_mixer_weight():
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        create_counterfactual_contribution_train_step(
            lambda params, value: value,
            lambda params, residual, mixer: residual + mixer,
            optax.sgd(1e-2),
            mixer_loss_weight=-1.0,
            bf16_gradients=False,
        )


def test_transient_prior_step_optimizes_base_through_effective_parameters():
    params = {"kernel": jnp.eye(2, dtype=jnp.float32)}
    offset = {"kernel": jnp.ones((2, 2), dtype=jnp.float32)}
    inputs = jnp.asarray([[1.0, 2.0]], jnp.float32)
    targets = 2.0 * inputs
    apply_fn = lambda candidate, value: value @ candidate["kernel"]
    tx = optax.sgd(learning_rate=1e-2)
    step = create_layerwise_prior_train_step(
        apply_fn, tx, bf16_gradients=False
    )
    effective = apply_parameter_offset(params, offset, 0.5)
    np.testing.assert_allclose(
        effective["kernel"], params["kernel"] + 0.5 * offset["kernel"]
    )
    updated, _, metrics = step(
        params, tx.init(params), inputs, targets, offset, 0.5
    )
    jax.block_until_ready(metrics)
    assert float(metrics["prior_scale"]) == 0.5
    assert bool(metrics["grads_finite"])
    assert not np.array_equal(np.asarray(updated["kernel"]), np.asarray(params["kernel"]))


def test_teacher_mixer_runner_is_jit_safe():
    config = tiny_qwen3_teacher_config()
    module = Qwen3GQAAttention(config)
    inputs = jnp.ones((1, 3, config.hidden_size), jnp.float32)
    positions = jnp.arange(3, dtype=jnp.int32)[None]
    mask = jnp.ones((1, 3), jnp.bool_)
    params = module.init(jax.random.key(4), inputs, positions, mask)["params"]
    output = create_teacher_mixer_runner(module, positions, mask)(params, inputs)
    jax.block_until_ready(output)
    assert output.shape == inputs.shape
    assert np.all(np.isfinite(np.asarray(output)))


def test_project_lion_initializes_with_frozen_mamba_parameters():
    params = freeze(
        {
            "in_proj": {"kernel": jnp.ones((2, 3), jnp.float32)},
            "b_norm": {"scale": jnp.ones((3,), jnp.float32)},
            "dt_bias": jnp.zeros((3,), jnp.float32),
        }
    )
    tx = create_lion(total_steps=4, warmup_steps=1)
    state = tx.init(params)
    assert jax.tree.structure(state)


def test_distillation_json_accepts_numpy_scalars_and_is_written(tmp_path):
    payload = {
        "passed": np.bool_(True),
        "loss": np.float32(0.5),
        "steps": np.int32(2),
    }
    encoded = json.dumps(payload, default=_json_default)
    assert json.loads(encoded) == {"passed": True, "loss": 0.5, "steps": 2}
    output = tmp_path / "result.json"
    _write_json(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(encoded)


def test_result_json_is_mirrored_to_output_directory(tmp_path):
    requested = tmp_path / "requested" / "result.json"
    output_dir = tmp_path / "output"
    mirror = _write_json_with_output_mirror(
        requested, {"passed": True}, str(output_dir)
    )
    assert mirror == output_dir / "result.json"
    assert json.loads(requested.read_text(encoding="utf-8"))["passed"] is True
    assert json.loads(mirror.read_text(encoding="utf-8"))["passed"] is True
