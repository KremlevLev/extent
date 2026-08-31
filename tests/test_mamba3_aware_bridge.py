from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from extent.config import tiny_config
from extent.layers.mamba3 import Mamba3MIMO
from extent.mamba3_aware_bridge import (
    create_exact_dual_bridge_train_step,
    dual_recurrent_parity,
    zero_complex_projection,
)
from extent.mamba3_transplant import mamba3_projection_slices
from extent.optimizer import create_lion


def test_exact_dual_bridge_updates_canonical_tree_and_preserves_zero_shock():
    config = tiny_config()
    inputs = jax.random.normal(
        jax.random.key(201), (2, 5, config.hidden_size), dtype=jnp.float32
    )
    targets = jax.random.normal(
        jax.random.key(202), inputs.shape, dtype=jnp.float32
    )
    recurrent = Mamba3MIMO(
        config.hidden_size,
        config.mamba,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        execution_mode="recurrent",
    )
    dual = Mamba3MIMO(
        config.hidden_size,
        config.mamba,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        execution_mode="dual",
    )
    params = recurrent.init(jax.random.key(203), inputs)["params"]
    tx = create_lion(
        learning_rate=1e-4,
        warmup_steps=1,
        total_steps=2,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    step = create_exact_dual_bridge_train_step(
        lambda candidate, values: dual.apply({"params": candidate}, values),
        tx,
        hidden_size=config.hidden_size,
        config=config.mamba,
        bf16_gradients=False,
    )
    updated, opt_state, metrics = step(params, tx.init(params), inputs, targets)
    updated, _, metrics = step(updated, opt_state, inputs, targets)
    jax.block_until_ready(metrics)
    assert bool(metrics["grads_finite"])
    assert np.isfinite(float(metrics["loss"]))
    assert not np.array_equal(
        np.asarray(updated["in_proj"]["kernel"]),
        np.asarray(params["in_proj"]["kernel"]),
    )
    parity = dual_recurrent_parity(
        lambda candidate, values: dual.apply({"params": candidate}, values),
        lambda candidate, values: recurrent.apply({"params": candidate}, values),
        updated,
        inputs,
    )
    assert parity["finite"]
    assert parity["relative_l2"] < 2e-5


def test_no_complex_ablation_keeps_angle_projection_zero():
    config = tiny_config()
    inputs = jnp.ones((1, 4, config.hidden_size), jnp.float32)
    module = Mamba3MIMO(
        config.hidden_size,
        config.mamba,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        execution_mode="dual",
    )
    params = zero_complex_projection(
        module.init(jax.random.key(204), inputs)["params"],
        config.hidden_size,
        config.mamba,
    )
    tx = create_lion(
        learning_rate=1e-4,
        warmup_steps=1,
        total_steps=2,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    step = create_exact_dual_bridge_train_step(
        lambda candidate, values: module.apply({"params": candidate}, values),
        tx,
        hidden_size=config.hidden_size,
        config=config.mamba,
        bf16_gradients=False,
        freeze_complex=True,
    )
    updated, _, metrics = step(params, tx.init(params), inputs, inputs)
    jax.block_until_ready(metrics)
    angle = mamba3_projection_slices(config.hidden_size, config.mamba)["angle"]
    assert np.count_nonzero(np.asarray(updated["in_proj"]["kernel"][:, angle])) == 0
