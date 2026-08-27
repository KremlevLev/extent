from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from extent import tiny_config
from extent.layers.mamba3 import Mamba3MIMO
from extent.mamba3_transplant import (
    build_qwen3_to_mamba3_transplant_variants,
    mamba3_projection_slices,
)
from extent.qwen3_parity import layer_mapping_entries
from extent.qwen3_teacher import tiny_qwen3_teacher_config
from extent.weight_mapping import expected_qwen_shape


def _arrays(source):
    rng = np.random.default_rng(51)
    return {
        entry.source: rng.normal(
            0.0, 0.02, expected_qwen_shape(entry, source)
        ).astype(np.float32)
        for entry in layer_mapping_entries(source, 0)
    }


def test_transplant_variants_are_controlled_and_mimo_channels_are_distinct():
    source = tiny_qwen3_teacher_config()
    config = tiny_config().mamba
    module = Mamba3MIMO(
        source.hidden_size, config, dtype=jnp.float32, param_dtype=jnp.float32
    )
    base = module.init(
        jax.random.key(5), jnp.ones((1, 4, source.hidden_size), jnp.float32)
    )["params"]
    variants, reports = build_qwen3_to_mamba3_transplant_variants(
        base, _arrays(source), source, config, 0
    )
    slices = mamba3_projection_slices(source.hidden_size, config)
    base_in = np.asarray(base["in_proj"]["kernel"])

    np.testing.assert_array_equal(
        variants["INIT-A-random"]["in_proj"]["kernel"], base_in
    )
    np.testing.assert_array_equal(
        variants["INIT-B-output-only"]["in_proj"]["kernel"], base_in
    )
    assert not np.array_equal(
        variants["INIT-B-output-only"]["out_proj"]["kernel"],
        base["out_proj"]["kernel"],
    )
    for name in ("dt", "a", "trap", "angle", "z"):
        np.testing.assert_array_equal(
            variants["INIT-E-mimo-aware"]["in_proj"]["kernel"][:, slices[name]],
            base_in[:, slices[name]],
        )
    assert not np.array_equal(
        variants["INIT-E-mimo-aware"]["b_norm"]["scale"],
        base["b_norm"]["scale"],
    )
    state = config.d_state
    siso_b = np.asarray(
        variants["INIT-D-siso-copied"]["in_proj"]["kernel"][:, slices["b"]]
    )
    mimo_b = np.asarray(
        variants["INIT-E-mimo-aware"]["in_proj"]["kernel"][:, slices["b"]]
    )
    np.testing.assert_allclose(siso_b[:, :state], siso_b[:, state : 2 * state])
    assert not np.allclose(mimo_b[:, :state], mimo_b[:, state : 2 * state])

    matched = variants["INIT-F-variance-matched-qkvo"]["in_proj"]["kernel"]
    blend_25 = variants["INIT-G-vm-qkvo-blend-0.25"]["in_proj"]["kernel"]
    blend_50 = variants["INIT-H-vm-qkvo-blend-0.5"]["in_proj"]["kernel"]
    for name in ("x", "b", "c"):
        base_slice = base_in[:, slices[name]]
        matched_slice = np.asarray(matched[:, slices[name]])
        np.testing.assert_allclose(
            np.sqrt(np.mean(np.square(matched_slice), dtype=np.float64)),
            np.sqrt(np.mean(np.square(base_slice), dtype=np.float64)),
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            blend_25[:, slices[name]],
            0.75 * base_slice + 0.25 * matched_slice,
            rtol=2e-6,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            blend_50[:, slices[name]],
            0.5 * base_slice + 0.5 * matched_slice,
            rtol=2e-6,
            atol=2e-7,
        )
    for name in ("dt", "a", "trap", "angle", "z"):
        np.testing.assert_array_equal(
            matched[:, slices[name]], base_in[:, slices[name]]
        )
    assert set(variants) == set(reports)
    inputs = jnp.asarray(
        np.random.default_rng(52).normal(size=(1, 5, source.hidden_size)).astype(
            np.float32
        )
    )
    for params in variants.values():
        output = module.apply({"params": params}, inputs)
        assert output.shape == inputs.shape
        assert np.all(np.isfinite(np.asarray(output)))

    single = variants["INIT-J-single-channel-qkvo-lift"]
    balanced = variants["INIT-K-balanced-qkvo-lift"]
    single_output, single_features = module.apply(
        {"params": single}, inputs, return_features=True
    )
    balanced_output, balanced_features = module.apply(
        {"params": balanced}, inputs, return_features=True
    )
    np.testing.assert_allclose(
        balanced_features, single_features, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        balanced_output, single_output, rtol=2e-5, atol=2e-6
    )
    rank = config.mimo_rank
    np.testing.assert_array_equal(
        single["mimo_x"][:, 1:, :], 0.0
    )
    np.testing.assert_allclose(balanced["mimo_x"], 1.0 / rank)
    np.testing.assert_allclose(balanced["mimo_o"], 1.0 / rank)
    np.testing.assert_allclose(balanced["D"], single["D"] * rank)
