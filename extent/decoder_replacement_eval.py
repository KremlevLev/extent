from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from extent.layers.common import RMSNorm, SwiGLU


class Qwen3DecoderTail(nn.Module):
    """Frozen Qwen residual/MLP path after either attention or a replacement mixer."""

    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float
    dtype: jnp.dtype
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, residual_input: jax.Array, mixer_output: jax.Array) -> jax.Array:
        post_mixer = residual_input + mixer_output
        normalized = RMSNorm(
            self.hidden_size,
            self.rms_norm_eps,
            self.param_dtype,
            name="post_attention_layernorm",
        )(post_mixer)
        mlp = SwiGLU(
            self.hidden_size,
            self.intermediate_size,
            self.dtype,
            self.param_dtype,
            name="mlp",
        )(normalized)
        return post_mixer + mlp


def create_batched_replacement_runner(
    mamba,
    decoder_tail: Qwen3DecoderTail,
) -> Callable:
    """Compile the exact frozen decoder tail around a trainable mixer."""

    @jax.jit
    def run(mamba_params, tail_params, residual_input, normalized_input):
        mixer_output = mamba.apply({"params": mamba_params}, normalized_input)
        decoder_output = decoder_tail.apply(
            {"params": tail_params}, residual_input, mixer_output
        )
        return mixer_output, decoder_output

    return run


def create_batched_teacher_tail_runner(decoder_tail: Qwen3DecoderTail) -> Callable:
    """Compile the same tail using cached frozen-attention outputs."""

    @jax.jit
    def run(tail_params, residual_input, attention_target):
        return decoder_tail.apply(
            {"params": tail_params}, residual_input, attention_target
        )

    return run
