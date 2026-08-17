from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from singularity import tiny_config
from singularity.layers.mamba3 import Mamba3MIMO
from singularity.mamba3_transplant import (
    build_qwen3_to_mamba3_transplant_variants,
    mamba3_projection_slices,
)
from singularity.qwen3_parity import layer_mapping_entries
from singularity.qwen3_teacher import tiny_qwen3_teacher_config
from singularity.weight_mapping import expected_qwen_shape


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
