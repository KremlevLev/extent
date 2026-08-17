"""Multi-head latent attention with MaxText-compatible parameter names."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from singularity.config import MLAConfig
from singularity.layers.common import RMSNorm


def apply_partial_rope(
    x: jax.Array,
    positions: jax.Array,
    theta: float,
    original_head_dim: int,
    strategy: str,
) -> jax.Array:
    """Apply Qwen split-half RoPE while preserving selected source frequencies."""
    pairs = x.shape[-1] // 2
    original_pairs = original_head_dim // 2
    if strategy == "high":
        frequencies = jnp.arange(pairs, dtype=jnp.float32)
    elif strategy == "low":
        frequencies = jnp.arange(original_pairs - pairs, original_pairs, dtype=jnp.float32)
    else:
        raise ValueError(f"unsupported partial RoPE strategy: {strategy}")
    inv_freq = theta ** (-(2.0 * frequencies) / original_head_dim)
    phase = positions.astype(jnp.float32)[..., None] * inv_freq
    while phase.ndim < x.ndim:
        phase = jnp.expand_dims(phase, -2)
    first, second = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    cos, sin = jnp.cos(phase), jnp.sin(phase)
    rotated = jnp.concatenate((first * cos - second * sin, second * cos + first * sin), axis=-1)
    return rotated.astype(x.dtype)


class MultiHeadLatentAttention(nn.Module):
    """Reference MLA path matching the field names used in MaxText base.yml.

    ``attention_mask`` is boolean and broadcastable to [batch, heads, query, key].
    A future MaxText kernel adapter can consume the same projected q/k/v tensors.
    """

    hidden_size: int
    config: MLAConfig
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        positions: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.config
        batch, length, _ = inputs.shape
        positions = jnp.arange(length)[None, :] if positions is None else positions
        if positions.shape[0] == 1 and batch != 1:
            positions = jnp.broadcast_to(positions, (batch, length))

        dense = lambda features, name: nn.Dense(
            features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(0.02),
            name=name,
        )

        if cfg.q_lora_rank > 0:
            q_latent = dense(cfg.q_lora_rank, "query_wa_proj")(inputs)
            q_latent = RMSNorm(
                cfg.q_lora_rank,
                param_dtype=self.param_dtype,
                name="q_layernorm",
            )(q_latent)
            q = dense(
                cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
                "query_wb_proj",
            )(q_latent)
        else:
            q = dense(
                cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
                "query_proj",
            )(inputs)

        # One shared latent plus one or more RoPE keys form the compressed cache.
        rope_width = cfg.num_key_rope_heads * cfg.qk_rope_head_dim
        kv_and_rope = dense(cfg.kv_lora_rank + rope_width, "kv_wa_proj")(inputs)
        kv_latent, k_rope = jnp.split(kv_and_rope, (cfg.kv_lora_rank,), axis=-1)
        if cfg.use_kv_latent_norm:
            kv_latent = RMSNorm(
                cfg.kv_lora_rank, param_dtype=self.param_dtype, name="kv_layernorm"
            )(kv_latent)
        kv = dense(
            cfg.num_kv_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            "kv_wb_proj",
        )(kv_latent)

        q = q.reshape(
            batch,
            length,
            cfg.num_heads,
            cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        )
        if cfg.use_qk_norm:
            q = RMSNorm(q.shape[-1], param_dtype=self.param_dtype, name="q_norm")(q)
        kv = kv.reshape(batch, length, cfg.num_kv_heads, cfg.qk_nope_head_dim + cfg.v_head_dim)
        k_nope, value = jnp.split(kv, (cfg.qk_nope_head_dim,), axis=-1)
        kv_groups = cfg.num_heads // cfg.num_kv_heads
        k_nope = jnp.repeat(k_nope, kv_groups, axis=2)
        value = jnp.repeat(value, kv_groups, axis=2)
        k_rope = k_rope.reshape(
            batch, length, cfg.num_key_rope_heads, cfg.qk_rope_head_dim
        )
        rope_groups = cfg.num_heads // cfg.num_key_rope_heads
        k_rope = jnp.repeat(k_rope, rope_groups, axis=2)
        key = jnp.concatenate((k_nope, k_rope), axis=-1)
        if cfg.use_qk_norm:
            key = RMSNorm(key.shape[-1], param_dtype=self.param_dtype, name="k_norm")(key)
        q_nope, q_rope = jnp.split(q, (cfg.qk_nope_head_dim,), axis=-1)
        k_nope, k_rope = jnp.split(key, (cfg.qk_nope_head_dim,), axis=-1)
        q_rope = apply_partial_rope(
            q_rope,
            positions,
            cfg.rope_theta,
            cfg.rope_original_head_dim,
            cfg.partial_rope_strategy,
        )
        k_rope = apply_partial_rope(
            k_rope,
            positions,
            cfg.rope_theta,
            cfg.rope_original_head_dim,
            cfg.partial_rope_strategy,
        )
        key = jnp.concatenate((k_nope, k_rope), axis=-1)
        query = jnp.concatenate((q_nope, q_rope), axis=-1)

        scale = (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5
        causal = jnp.arange(length)[None, :] <= jnp.arange(length)[:, None]
        mask = causal[None, None]
        if attention_mask is not None:
            supplied = attention_mask.astype(jnp.bool_)
            if supplied.ndim == 2:
                supplied = supplied[:, None, None, :]
            mask = mask & supplied

        def attend(q_chunk: jax.Array, k_chunk: jax.Array, v_chunk: jax.Array) -> jax.Array:
            chunk_logits = jnp.einsum(
                "bqhd,bkhd->bhqk", q_chunk, k_chunk, preferred_element_type=jnp.float32
            ) * scale
            chunk_logits = jnp.where(mask, chunk_logits, jnp.finfo(jnp.float32).min)
            probabilities = jax.nn.softmax(chunk_logits, axis=-1).astype(self.dtype)
            return jnp.einsum("bhqk,bkhd->bqhd", probabilities, v_chunk)

        chunk = cfg.qk_head_chunk_size
        if chunk and chunk < cfg.num_heads:
            if cfg.num_heads % chunk:
                raise ValueError("qk_head_chunk_size must divide num_heads")
            chunks = cfg.num_heads // chunk
            to_chunks = lambda tensor: jnp.transpose(
                tensor.reshape(batch, length, chunks, chunk, tensor.shape[-1]), (2, 0, 1, 3, 4)
            )
            attended = jax.lax.map(
                lambda tensors: attend(*tensors),
                (to_chunks(query), to_chunks(key), to_chunks(value)),
            )
            attended = jnp.transpose(attended, (1, 2, 0, 3, 4)).reshape(
                batch, length, cfg.num_heads, cfg.v_head_dim
            )
        else:
            attended = attend(query, key, value)
        attended = attended.reshape(batch, length, cfg.num_heads * cfg.v_head_dim)
        return dense(self.hidden_size, "out_proj")(attended)
