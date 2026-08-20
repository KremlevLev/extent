from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from extent.layers.common import RMSNorm
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.rorope import apply_rorope


class Qwen3RoRoPEBKVAttention(nn.Module):
    """Qwen3-faithful RoRoPE+BKV attention with an explicit compressed cache."""

    config: Qwen3TeacherConfig
    latent_rank: int = 448
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        return_cache: bool = False,
    ):
        cfg = self.config
        batch, length, _ = inputs.shape

        def dense(features: int, name: str) -> nn.Dense:
            return nn.Dense(
                features,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
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
            cfg.head_dim, cfg.rms_norm_eps, self.param_dtype, name="q_norm"
        )(query)
        key = RMSNorm(
            cfg.head_dim, cfg.rms_norm_eps, self.param_dtype, name="k_norm"
        )(key)

        rotations = self.param(
            "rorope_rotations",
            nn.initializers.orthogonal(),
            (cfg.head_dim // 2, cfg.num_key_value_heads, cfg.num_key_value_heads),
            self.param_dtype,
        )
        joint_width = (2 * cfg.num_key_value_heads - 1) * cfg.head_dim
        basis = self.param(
            "joint_basis",
            nn.initializers.orthogonal(),
            (self.latent_rank, joint_width),
            self.param_dtype,
        )
        balance = self.param(
            "bkv_balance_ratio", nn.initializers.ones, (), self.param_dtype
        ).astype(jnp.float32)
        mapping = jnp.arange(cfg.num_attention_heads, dtype=jnp.int32) // (
            cfg.num_attention_heads // cfg.num_key_value_heads
        )
        query, rotated_key = apply_rorope(
            query, key, positions, rotations, mapping, 1, cfg.rope_theta
        )
        key_nope = rotated_key[:, :, 1:].reshape(
            batch, length, (cfg.num_key_value_heads - 1) * cfg.head_dim
        )
        value_flat = value.reshape(
            batch, length, cfg.num_key_value_heads * cfg.head_dim
        )
        balanced_joint = jnp.concatenate(
            (key_nope.astype(jnp.float32) / balance, value_flat.astype(jnp.float32)),
            axis=-1,
        )
        basis_f32 = basis.astype(jnp.float32)
        latent = balanced_joint @ basis_f32.T
        reconstructed = latent @ basis_f32
        key_width = key_nope.shape[-1]
        reconstructed_key = reconstructed[:, :, :key_width] * balance
        reconstructed_value = reconstructed[:, :, key_width:]
        reconstructed_key = reconstructed_key.reshape(
            batch, length, cfg.num_key_value_heads - 1, cfg.head_dim
        )
        reconstructed_value = reconstructed_value.reshape(
            batch, length, cfg.num_key_value_heads, cfg.head_dim
        )
        reconstructed_key = jnp.concatenate(
            (rotated_key[:, :, :1], reconstructed_key), axis=2
        )

        logits = jnp.einsum(
            "bqhcd,bkcd->bhqk",
            query,
            reconstructed_key,
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
        probabilities = jax.nn.softmax(logits, axis=-1).astype(self.dtype)
        attended = jnp.einsum(
            "bhqk,bkhd->bqhd", probabilities, reconstructed_value[:, :, mapping]
        )
        attended = attended.reshape(
            batch, length, cfg.num_attention_heads * cfg.head_dim
        )
        output = dense(cfg.hidden_size, "o_proj")(attended)
        if return_cache:
            return output, {
                "kv_latent": latent.astype(self.dtype),
                "k_rope": rotated_key[:, :, 0].astype(self.dtype),
            }
        return output
