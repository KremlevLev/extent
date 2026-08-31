from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch

from extent.config import Mamba3Config, tiny_config
from extent.layers.mamba3 import (
    MAMBA3_REFERENCE_COMMIT,
    Mamba3MIMO,
    mamba3_reference_scan,
)
from tests.mamba3_torch_reference import (
    MAMBA3_REFERENCE_COMMIT as TORCH_REFERENCE_COMMIT,
    torch_mamba3_mimo_scan,
)


def _fixture(seed: int = 41):
    rng = np.random.default_rng(seed)
    batch, length, heads, head_dim, rank, state_dim = 2, 7, 3, 6, 2, 8
    rotary_pairs = 2

    def normal(shape, scale=0.2):
        return rng.normal(0.0, scale, shape).astype(np.float32)

    return {
        "x": normal((batch, length, heads, head_dim)),
        "z": normal((batch, length, heads, head_dim)),
        "b": normal((batch, length, rank, heads, state_dim)),
        "c": normal((batch, length, rank, heads, state_dim)),
        "dt": rng.uniform(0.001, 0.1, (batch, length, heads)).astype(np.float32),
        "decay": -rng.uniform(0.05, 1.5, (batch, length, heads)).astype(np.float32),
        "trap": normal((batch, length, heads), 1.0),
        # Values beyond [-1, 1] deliberately catch an incorrect tanh here.
        "angle_step": normal((batch, length, rotary_pairs), 1.5),
        "mimo_x": normal((heads, rank, head_dim)),
        "mimo_z": normal((heads, rank, head_dim)),
        "mimo_o": normal((heads, rank, head_dim)),
        "skip": normal((heads,)),
        "rotary_pairs": rotary_pairs,
    }


def test_jax_scan_matches_independent_torch_mamba3_mimo_reference():
    fixture = _fixture()
    rotary_pairs = fixture.pop("rotary_pairs")
    jax_output = mamba3_reference_scan(
        *(jnp.asarray(fixture[name]) for name in fixture), rotary_pairs
    )
    torch_output = torch_mamba3_mimo_scan(
        *(torch.from_numpy(fixture[name]) for name in fixture), rotary_pairs
    )
    np.testing.assert_allclose(
        np.asarray(jax_output), torch_output.numpy(), atol=2e-6, rtol=2e-6
    )
    assert MAMBA3_REFERENCE_COMMIT == TORCH_REFERENCE_COMMIT


def test_mamba3_module_uses_official_parameter_shapes_and_names():
    config = tiny_config()
    module = Mamba3MIMO(
        config.hidden_size, config.mamba, dtype=jnp.float32, param_dtype=jnp.float32
    )
    params = module.init(
        jax.random.key(9),
        jnp.ones((1, 3, config.hidden_size), dtype=jnp.float32),
    )["params"]
    heads = int(config.hidden_size * config.mamba.expand) // config.mamba.head_dim
    expected = (heads, config.mamba.mimo_rank, config.mamba.head_dim)
    assert params["mimo_x"].shape == expected
    assert params["mimo_z"].shape == expected
    assert params["mimo_o"].shape == expected
    assert params["dt_bias"].shape == (heads,)
    assert "mimo_out" not in params
    assert "dt_logit" not in params
    inputs = jnp.ones((1, 3, config.hidden_size), dtype=jnp.float32)
    output = module.apply({"params": params}, inputs)
    feature_output, features = module.apply(
        {"params": params}, inputs, return_features=True
    )
    np.testing.assert_allclose(feature_output, output, atol=0.0, rtol=0.0)
    assert features.shape == (
        1,
        3,
        int(config.hidden_size * config.mamba.expand),
    )


def test_quadratic_dual_and_recurrent_module_are_functionally_identical():
    config = Mamba3Config(
        d_state=8,
        expand=1.0,
        head_dim=8,
        groups=1,
        mimo_rank=3,
        rope_fraction=0.5,
    )
    inputs = jax.random.normal(jax.random.key(101), (2, 7, 16), dtype=jnp.float32)
    recurrent = Mamba3MIMO(
        16,
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        execution_mode="recurrent",
    )
    dual = Mamba3MIMO(
        16,
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        execution_mode="dual",
    )
    params = recurrent.init(jax.random.key(102), inputs)["params"]
    recurrent_output = recurrent.apply({"params": params}, inputs)
    dual_output = dual.apply({"params": params}, inputs)
    np.testing.assert_allclose(
        np.asarray(dual_output),
        np.asarray(recurrent_output),
        rtol=2e-5,
        atol=2e-5,
    )
