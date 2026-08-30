import jax
import jax.numpy as jnp
import numpy as np

from extent import HybridForCausalLM, tiny_config
from extent.full_hybrid_materialization import (
    materialize_exact_lift_layer,
    materialize_rorope_bkv_layer,
)
from extent.qwen3_parity import layer_mapping_entries
from extent.qwen3_teacher import tiny_qwen3_teacher_config
from extent.weight_mapping import expected_qwen_shape


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
