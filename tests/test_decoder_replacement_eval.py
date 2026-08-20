from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from extent.decoder_replacement_eval import (
    Qwen3DecoderTail,
    create_batched_teacher_tail_runner,
)


def test_decoder_tail_is_identity_when_mixer_and_mlp_are_zero():
    module = Qwen3DecoderTail(
        hidden_size=4,
        intermediate_size=8,
        rms_norm_eps=1e-6,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    residual = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 4)
    mixer = jnp.zeros_like(residual)
    params = module.init(jax.random.key(0), residual, mixer)["params"]
    params = jax.tree.map(jnp.zeros_like, params)
    output = module.apply({"params": params}, residual, mixer)
    np.testing.assert_array_equal(output, residual)


def test_teacher_tail_runner_uses_cached_mixer_output_before_residual_mlp():
    module = Qwen3DecoderTail(
        hidden_size=4,
        intermediate_size=8,
        rms_norm_eps=1e-6,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    residual = jnp.ones((1, 2, 4), dtype=jnp.float32)
    mixer = jnp.full_like(residual, 0.25)
    params = module.init(jax.random.key(1), residual, mixer)["params"]
    params = jax.tree.map(jnp.zeros_like, params)
    runner = create_batched_teacher_tail_runner(module)
    output = runner(params, residual, mixer)
    np.testing.assert_array_equal(output, residual + mixer)
