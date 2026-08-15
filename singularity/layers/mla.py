"""Multi-head latent attention with MaxText-compatible parameter names."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from singularity.config import MLAConfig
from singularity.layers.common import RMSNorm


def apply_rope(x: jax.Array, positions: jax.Array, theta: float) -> jax.Array:
    dim = x.shape[-1]
    inv_freq = theta ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    phase = positions.astype(jnp.float32)[..., None] * inv_freq
    while phase.ndim < x.ndim:
        phase = jnp.expand_dims(phase, -2)
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = jnp.cos(phase), jnp.sin(phase)
    return jnp.stack((even * cos - odd * sin, even * sin + odd * cos), axis=-1).reshape(x.shape)


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
            q_latent = RMSNorm(cfg.q_lora_rank, param_dtype=self.param_dtype, name="q_layernorm")(q_latent)
            q = dense(cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), "query_wb_proj")(q_latent)
        else:
            q = dense(cfg.num_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), "query_proj")(inputs)

        # One shared latent and one shared RoPE key are the compressed KV cache.
        kv_and_rope = dense(cfg.kv_lora_rank + cfg.qk_rope_head_dim, "kv_wa_proj")(inputs)
        kv_latent, k_rope = jnp.split(kv_and_rope, (cfg.kv_lora_rank,), axis=-1)
        kv_latent = RMSNorm(cfg.kv_lora_rank, param_dtype=self.param_dtype, name="kv_layernorm")(kv_latent)
        kv = dense(cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), "kv_wb_proj")(kv_latent)

        q = q.reshape(batch, length, cfg.num_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        q_nope, q_rope = jnp.split(q, (cfg.qk_nope_head_dim,), axis=-1)
        kv = kv.reshape(batch, length, cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim)
        k_nope, value = jnp.split(kv, (cfg.qk_nope_head_dim,), axis=-1)
        q_rope = apply_rope(q_rope, positions, cfg.rope_theta)
        k_rope = apply_rope(k_rope[:, :, None, :], positions, cfg.rope_theta)
        key = jnp.concatenate((k_nope, jnp.broadcast_to(k_rope, q_rope.shape)), axis=-1)
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
            attended = jax.lax.map(lambda tensors: attend(*tensors), (to_chunks(query), to_chunks(key), to_chunks(value)))
            attended = jnp.transpose(attended, (1, 2, 0, 3, 4)).reshape(
                batch, length, cfg.num_heads, cfg.v_head_dim
            )
        else:
            attended = attend(query, key, value)
        attended = attended.reshape(batch, length, cfg.num_heads * cfg.v_head_dim)
        return dense(self.hidden_size, "out_proj")(attended)
