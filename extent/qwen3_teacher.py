from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from extent.layers.common import RMSNorm, SwiGLU, dtype_from_name


@dataclass(frozen=True)
class Qwen3TeacherConfig:
    vocab_size: int = 151_936
    hidden_size: int = 5_120
    intermediate_size: int = 17_408
    num_layers: int = 40
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40_960
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    param_dtype: str = "bfloat16"
    compute_dtype: str = "bfloat16"
    logits_dtype: str = "float32"
    remat_policy: str = "full"

    def __post_init__(self) -> None:
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("Qwen3 query heads must span hidden_size")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim % 2:
            raise ValueError("RoPE head_dim must be even")


def tiny_qwen3_teacher_config() -> Qwen3TeacherConfig:
    return Qwen3TeacherConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        remat_policy="none",
    )


def apply_qwen3_rope(
    tensor: jax.Array,
    positions: jax.Array,
    theta: float,
) -> jax.Array:
    """Qwen3/Hugging Face split-half rotary embedding in FP32."""
    dim = tensor.shape[-1]
    inv_freq = theta ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    frequencies = positions.astype(jnp.float32)[..., None] * inv_freq
    cos = jnp.concatenate((jnp.cos(frequencies), jnp.cos(frequencies)), axis=-1)
    sin = jnp.concatenate((jnp.sin(frequencies), jnp.sin(frequencies)), axis=-1)
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    first, second = jnp.split(tensor.astype(jnp.float32), 2, axis=-1)
    rotated_half = jnp.concatenate((-second, first), axis=-1)
    return (tensor.astype(jnp.float32) * cos + rotated_half * sin).astype(tensor.dtype)


class Qwen3GQAAttention(nn.Module):
    config: Qwen3TeacherConfig

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.config
        dtype = dtype_from_name(cfg.compute_dtype)
        param_dtype = dtype_from_name(cfg.param_dtype)
        batch, length, _ = inputs.shape

        def dense(features: int, name: str) -> nn.Dense:
            return nn.Dense(
                features,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(0.02),
                name=name,
            )

        query = dense(cfg.num_attention_heads * cfg.head_dim, "q_proj")(inputs)
        key = dense(cfg.num_key_value_heads * cfg.head_dim, "k_proj")(inputs)
        value = dense(cfg.num_key_value_heads * cfg.head_dim, "v_proj")(inputs)
        query = query.reshape(batch, length, cfg.num_attention_heads, cfg.head_dim)
        key = key.reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim)
        value = value.reshape(batch, length, cfg.num_key_value_heads, cfg.head_dim)

        query = RMSNorm(
            cfg.head_dim, cfg.rms_norm_eps, param_dtype, name="q_norm"
        )(query)
        key = RMSNorm(
            cfg.head_dim, cfg.rms_norm_eps, param_dtype, name="k_norm"
        )(key)
        query = apply_qwen3_rope(query, positions, cfg.rope_theta)
        key = apply_qwen3_rope(key, positions, cfg.rope_theta)

        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        key = jnp.repeat(key, groups, axis=2)
        value = jnp.repeat(value, groups, axis=2)
        logits = jnp.einsum(
            "bqhd,bkhd->bhqk",
            query,
            key,
            preferred_element_type=jnp.float32,
        ) * (cfg.head_dim**-0.5)

        causal = positions[:, None, :, None] >= positions[:, None, None, :]
        mask = causal
        if attention_mask is not None:
            supplied = attention_mask.astype(jnp.bool_)
            if supplied.ndim == 2:
                supplied = supplied[:, None, None, :]
            mask = mask & supplied
        logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
        probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probabilities, value)
        attended = attended.reshape(batch, length, cfg.num_attention_heads * cfg.head_dim)
        return dense(cfg.hidden_size, "o_proj")(attended)


class Qwen3DecoderLayer(nn.Module):
    config: Qwen3TeacherConfig

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.config
        dtype = dtype_from_name(cfg.compute_dtype)
        param_dtype = dtype_from_name(cfg.param_dtype)
        residual = inputs
        normalized = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, param_dtype, name="input_layernorm"
        )(inputs)
        inputs = residual + Qwen3GQAAttention(cfg, name="self_attn")(
            normalized, positions, attention_mask
        )
        residual = inputs
        normalized = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, param_dtype, name="post_attention_layernorm"
        )(inputs)
        mlp = SwiGLU(
            cfg.hidden_size,
            cfg.intermediate_size,
            dtype,
            param_dtype,
            name="mlp",
        )(normalized)
        return residual + mlp


class Qwen3ForCausalLM(nn.Module):
    config: Qwen3TeacherConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        positions: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        *,
        return_hidden_states: bool = False,
    ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, ...]]:
        cfg = self.config
        dtype = dtype_from_name(cfg.compute_dtype)
        param_dtype = dtype_from_name(cfg.param_dtype)
        batch, length = input_ids.shape
        positions = jnp.arange(length, dtype=jnp.int32)[None, :] if positions is None else positions
        if positions.shape[0] == 1 and batch != 1:
            positions = jnp.broadcast_to(positions, (batch, length))

        inputs = nn.Embed(
            cfg.vocab_size,
            cfg.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=nn.initializers.normal(0.02),
            name="embed_tokens",
        )(input_ids)
        hidden_states = []
        layer_type = Qwen3DecoderLayer
        if cfg.remat_policy == "full":
            layer_type = nn.remat(Qwen3DecoderLayer, prevent_cse=False)
        for index in range(cfg.num_layers):
            inputs = layer_type(cfg, name=f"layers_{index}")(
                inputs, positions, attention_mask
            )
            if return_hidden_states:
                hidden_states.append(inputs)
        inputs = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, param_dtype, name="norm"
        )(inputs)
        logits = nn.Dense(
            cfg.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.normal(0.02),
            name="lm_head",
        )(inputs).astype(dtype_from_name(cfg.logits_dtype))
        if return_hidden_states:
            return logits, tuple(hidden_states)
        return logits

