from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from singularity.layerwise_distillation import create_layerwise_train_step, relative_mse


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
