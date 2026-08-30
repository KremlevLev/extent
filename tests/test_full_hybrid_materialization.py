import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from extent import HybridForCausalLM, tiny_config
from extent.full_hybrid_materialization import (
    materialize_exact_lift_layer,
    materialize_rorope_bkv_layer,
    materialize_qwen_gqa_layer,
)
from extent.qwen3_parity import layer_mapping_entries
from extent.qwen3_teacher import tiny_qwen3_teacher_config
from extent.weight_mapping import expected_qwen_shape
from extent.initialization import initialize_sharded_parameters
from extent.sharding import batch_sharding, create_v5e_mesh
from scripts.full_hybrid_materialization_campaign import _backward_probe


def _mixer_arrays(source, layer):
    rng = np.random.default_rng(700 + layer)
    values = {}
    for entry in layer_mapping_entries(source, layer):
        if ".self_attn." not in entry.source and not entry.source.endswith("input_layernorm.weight"):
            continue
        shape = expected_qwen_shape(entry, source)
        values[entry.source] = (
            np.ones(shape, np.float32)
            if entry.source.endswith("norm.weight")
            else rng.normal(0, 0.02, shape).astype(np.float32)
        )
    return values


def test_materializes_selected_mixers_without_changing_contract():
    config = tiny_config()
    source = tiny_qwen3_teacher_config()
    params = HybridForCausalLM(config).init(
        jax.random.key(0), jnp.zeros((1, 4), jnp.int32)
    )["params"]

    params, mamba_report = materialize_exact_lift_layer(
        params, _mixer_arrays(source, 0), source, config, 0
    )
    calibration = np.random.default_rng(4).normal(
        size=(2, 8, source.hidden_size)
    ).astype(np.float32)
    params, mla_report = materialize_rorope_bkv_layer(
        params, _mixer_arrays(source, 1), calibration, source, config, 1
    )

    assert mamba_report.method == "INIT-K-balanced-qkvo-lift"
    assert mla_report.method == "qwen3_rorope_fold1_bkv_activation_pca"
    assert mla_report.calibration_tokens == 16
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(params))
    logits = HybridForCausalLM(config).apply(
        {"params": params}, jnp.zeros((1, 4), jnp.int32)
    )
    assert logits.shape == (1, 4, config.vocab_size)


def test_materializes_retained_qwen_gqa_for_isolated_mamba_ablation():
    base = tiny_config()
    config = replace(
        base, mla=replace(base.mla, implementation="qwen3_gqa")
    )
    source = tiny_qwen3_teacher_config()
    model = HybridForCausalLM(config)
    params = model.init(jax.random.key(5), jnp.zeros((1, 4), jnp.int32))["params"]

    params, report = materialize_qwen_gqa_layer(
        params, _mixer_arrays(source, 1), source, 1
    )

    assert report.target_mixer == "qwen3_gqa"
    assert report.method == "DIRECT-QWEN3-GQA"
    assert report.tensor_count == 6
    logits, hidden = model.apply(
        {"params": params},
        jnp.zeros((1, 4), jnp.int32),
        return_hidden_states=True,
    )
    assert logits.shape == (1, 4, config.vocab_size)
    assert len(hidden) == config.num_layers
    assert all(np.all(np.isfinite(np.asarray(value))) for value in hidden)


def test_full_model_backward_probe_reduces_gradient_tree_to_finite_metrics():
    config = tiny_config()
    model = HybridForCausalLM(config)
    mesh = create_v5e_mesh()
    tokens = jax.device_put(
        np.zeros((mesh.shape["data"], 1), np.int32),
        batch_sharding(mesh),
    )
    initialized = initialize_sharded_parameters(
        model,
        jax.random.key(62),
        tokens,
        mesh,
    )

    result = _backward_probe(
        model,
        initialized.params,
        initialized.layout,
        mesh,
        4,
    )

    assert result["sequence_length"] == 4
    assert result["grads_finite"] is True
    assert result["nonfinite_grad_leaves"] == 0
    assert np.isfinite(result["loss"])
    assert np.isfinite(result["grad_norm"])
    assert np.isfinite(result["max_abs_grad"])
