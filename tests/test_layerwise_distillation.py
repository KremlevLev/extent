from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze

from singularity.layerwise_distillation import (
    create_layerwise_train_step,
    create_teacher_mixer_runner,
    relative_mse,
)
from singularity.qwen3_teacher import Qwen3GQAAttention, tiny_qwen3_teacher_config
from singularity.optimizer import create_lion
from scripts.qwen_mamba3_distill_pilot import _json_default, _write_json


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
