from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from extent import tiny_config
from extent.attention_bridge import (
    AttentionBridgeConfig,
    apply_attention_bridge,
    build_bridge_mamba3_initialization,
    create_bridge_train_step,
    create_orientation_train_step,
    explicit_bridge_matrix,
    hedgehog_features,
    initialize_bridge_params,
    mamba3_orientation_matrix,
    qwen3_attention_components,
)
from extent.layers.mamba3 import Mamba3MIMO
from extent.optimizer import create_lion
from extent.qwen3_parity import jax_attention_params, layer_mapping_entries
from extent.qwen3_teacher import Qwen3GQAAttention, tiny_qwen3_teacher_config
from extent.qwen3_teacher import apply_qwen3_rope
from extent.weight_mapping import expected_qwen_shape


def _arrays(source):
    rng = np.random.default_rng(701)
    return {
        entry.source: rng.normal(
            0.0, 0.02, expected_qwen_shape(entry, source)
        ).astype(np.float32)
        for entry in layer_mapping_entries(source, 0)
    }


def test_qwen_probe_and_recurrent_bridge_match_reference_and_explicit_form():
    source = replace(
        tiny_qwen3_teacher_config(),
        param_dtype="float32",
        compute_dtype="float32",
    )
    arrays = _arrays(source)
    params = jax_attention_params(arrays, source, 0)
    inputs = jax.random.normal(jax.random.key(1), (2, 5, source.hidden_size))
    positions = jnp.broadcast_to(jnp.arange(5, dtype=jnp.int32)[None], (2, 5))
    attention_mask = jnp.ones((2, 5), dtype=jnp.bool_)
    components = qwen3_attention_components(
        params, inputs, positions, source, attention_mask
    )
    reference = Qwen3GQAAttention(source).apply(
        {"params": params}, inputs, positions, attention_mask
    )
    np.testing.assert_allclose(components.output, reference, rtol=2e-5, atol=2e-5)

    bridge_config = AttentionBridgeConfig(feature_dim=8)
    bridge_params = initialize_bridge_params(
        jax.random.key(2), source.head_dim, bridge_config
    )
    bridge = apply_attention_bridge(
        bridge_params,
        components,
        positions,
        attention_mask,
        params["o_proj"]["kernel"],
        bridge_config,
    )
    query_features = hedgehog_features(
        components.query, bridge_params["query"]
    )
    key_features = hedgehog_features(components.key, bridge_params["key"])
    query_features = apply_qwen3_rope(
        query_features, positions, bridge_config.rope_theta
    )
    key_features = apply_qwen3_rope(
        key_features, positions, bridge_config.rope_theta
    )
    explicit_matrix = explicit_bridge_matrix(
        query_features,
        key_features,
        positions,
        attention_mask,
        bridge_config.epsilon,
    )
    explicit_values = jnp.einsum(
        "bhqk,bkhd->bqhd", explicit_matrix, components.value
    ).reshape(2, 5, source.hidden_size)
    explicit_output = jnp.einsum(
        "bld,df->blf", explicit_values, params["o_proj"]["kernel"]
    )
    np.testing.assert_allclose(bridge.matrix, explicit_matrix, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(bridge.output, explicit_output, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(
        np.asarray(bridge.matrix).sum(axis=-1), 1.0, rtol=2e-5, atol=2e-6
    )
    assert np.allclose(np.triu(np.asarray(bridge.matrix), k=1), 0.0)


def test_qwen_probe_matches_bfloat16_teacher_path():
    source = tiny_qwen3_teacher_config()
    arrays = _arrays(source)
    params = jax_attention_params(arrays, source, 0)
    inputs = jax.random.normal(
        jax.random.key(21), (1, 5, source.hidden_size), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    positions = jnp.arange(5, dtype=jnp.int32)[None]
    mask = jnp.ones((1, 5), dtype=jnp.bool_)
    candidate = qwen3_attention_components(
        params, inputs, positions, source, mask
    ).output
    reference = Qwen3GQAAttention(source).apply(
        {"params": params}, inputs, positions, mask
    )
    np.testing.assert_allclose(candidate, reference, rtol=3e-3, atol=3e-4)


def test_learned_bridge_initialization_preserves_mamba_contract_and_is_finite():
    source = replace(
        tiny_qwen3_teacher_config(),
        param_dtype="float32",
        compute_dtype="float32",
    )
    mamba_config = tiny_config().mamba
    inputs = jax.random.normal(jax.random.key(3), (1, 5, source.hidden_size))
    module = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    base = module.init(jax.random.key(4), inputs)["params"]
    bridge_params = initialize_bridge_params(
        jax.random.key(5),
        source.head_dim,
        AttentionBridgeConfig(feature_dim=mamba_config.d_state),
    )
    initialized, report = build_bridge_mamba3_initialization(
        base,
        _arrays(source),
        bridge_params,
        source,
        mamba_config,
        0,
    )
    assert report.variant == "INIT-J-apple-linear-bridge"
    assert not report.exact_functional_equivalence
    assert jax.tree.structure(initialized) == jax.tree.structure(base)
    output = module.apply({"params": initialized}, inputs)
    assert output.shape == inputs.shape
    assert np.all(np.isfinite(np.asarray(output)))
    np.testing.assert_array_equal(initialized["D"], 0.0)


def test_mamba3_orientation_proxy_is_causal_normalized_and_differentiable():
    source = replace(
        tiny_qwen3_teacher_config(),
        param_dtype="float32",
        compute_dtype="float32",
    )
    mamba_config = tiny_config().mamba
    inputs = jax.random.normal(jax.random.key(6), (2, 6, source.hidden_size))
    module = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    params = module.init(jax.random.key(7), inputs[:1])["params"]
    matrix = mamba3_orientation_matrix(params, inputs, mamba_config)
    assert matrix.shape == (2, 6, 6)
    np.testing.assert_allclose(
        np.asarray(matrix).sum(axis=-1), 1.0, rtol=2e-5, atol=2e-6
    )
    assert np.allclose(np.triu(np.asarray(matrix), k=1), 0.0)

    loss, grads = jax.value_and_grad(
        lambda candidate: jnp.mean(
            jnp.square(
                mamba3_orientation_matrix(candidate, inputs, mamba_config)
                - jnp.eye(6)[None]
            )
        )
    )(params)
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(grads))


def test_bridge_and_orientation_compiled_train_steps_are_finite():
    source = replace(
        tiny_qwen3_teacher_config(),
        param_dtype="float32",
        compute_dtype="float32",
    )
    arrays = _arrays(source)
    attention_params = jax_attention_params(arrays, source, 0)
    inputs = jax.random.normal(jax.random.key(8), (1, 5, source.hidden_size))
    positions = jnp.arange(5, dtype=jnp.int32)[None]
    mask = jnp.ones((1, 5), dtype=jnp.bool_)
    teacher = qwen3_attention_components(
        attention_params, inputs, positions, source, mask
    )
    bridge_config = AttentionBridgeConfig(feature_dim=8)
    bridge_params = initialize_bridge_params(
        jax.random.key(9), source.head_dim, bridge_config
    )
    bridge_tx = create_lion(
        learning_rate=1e-3,
        warmup_steps=1,
        total_steps=2,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    bridge_step = create_bridge_train_step(
        bridge_tx, bridge_config, matrix_loss_weight=0.1
    )
    bridge_params, _, bridge_metrics = bridge_step(
        bridge_params,
        bridge_tx.init(bridge_params),
        teacher,
        positions,
        mask,
        attention_params["o_proj"]["kernel"],
    )
    jax.block_until_ready(bridge_metrics)
    assert bool(bridge_metrics["grads_finite"])
    assert np.isfinite(float(bridge_metrics["loss"]))

    mamba_config = tiny_config().mamba
    module = Mamba3MIMO(
        source.hidden_size,
        mamba_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    mamba_params = module.init(jax.random.key(10), inputs)["params"]
    orientation_tx = create_lion(
        learning_rate=3e-5,
        warmup_steps=1,
        total_steps=2,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )
    orientation_step = create_orientation_train_step(
        orientation_tx, mamba_config, bf16_gradients=False
    )
    _, _, orientation_metrics = orientation_step(
        mamba_params,
        orientation_tx.init(mamba_params),
        inputs,
        teacher.matrix,
    )
    jax.block_until_ready(orientation_metrics)
    assert bool(orientation_metrics["grads_finite"])
    assert np.isfinite(float(orientation_metrics["loss"]))
