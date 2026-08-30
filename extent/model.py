from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from extent.config import HybridConfig
from extent.layers.common import RMSNorm, SwiGLU, dtype_from_name
from extent.layers.mamba3 import Mamba3MIMO
from extent.layers.mla import MultiHeadLatentAttention
from extent.layers.rorope_bkv import Qwen3RoRoPEBKVAttention
from extent.qwen3_teacher import Qwen3GQAAttention, Qwen3TeacherConfig


def _attention_source_config(config: HybridConfig) -> Qwen3TeacherConfig:
    """Construct the source-faithful attention contract without a teacher model."""
    return Qwen3TeacherConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_layers=config.num_layers,
        num_attention_heads=config.mla.num_heads,
        num_key_value_heads=config.mla.num_kv_heads,
        head_dim=config.hidden_size // config.mla.num_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.mla.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        param_dtype=config.param_dtype,
        compute_dtype=config.compute_dtype,
        logits_dtype=config.logits_dtype,
        remat_policy=config.remat_policy,
    )


class HybridDecoderLayer(nn.Module):
    config: HybridConfig
    layer_index: int

    @nn.compact
    def __call__(self, x: jax.Array, positions: jax.Array, attention_mask: jax.Array | None) -> jax.Array:
        cfg = self.config
        dtype, param_dtype = dtype_from_name(cfg.compute_dtype), dtype_from_name(cfg.param_dtype)
        residual = x
        normalized = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, param_dtype, name="input_layernorm")(x)
        if self.layer_index in cfg.attention_layer_indices:
            if cfg.mla.implementation == "qwen3_gqa":
                mixed = Qwen3GQAAttention(
                    _attention_source_config(cfg), name="self_attn"
                )(normalized, positions, attention_mask)
            elif cfg.mla.implementation == "rorope_bkv":
                mixed = Qwen3RoRoPEBKVAttention(
                    _attention_source_config(cfg),
                    latent_rank=cfg.mla.kv_lora_rank,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    name="self_attn",
                )(normalized, positions, attention_mask)
            else:
                mixed = MultiHeadLatentAttention(
                    cfg.hidden_size, cfg.mla, dtype, param_dtype, name="self_attn"
                )(normalized, positions, attention_mask)
        else:
            mixed = Mamba3MIMO(cfg.hidden_size, cfg.mamba, dtype, param_dtype, name="mamba")(normalized)
        x = residual + mixed
        residual = x
        normalized = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, param_dtype, name="post_attention_layernorm")(x)
        x = SwiGLU(cfg.hidden_size, cfg.intermediate_size, dtype, param_dtype, name="mlp")(normalized)
        return residual + x


class HybridForCausalLM(nn.Module):
    config: HybridConfig

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
        dtype, param_dtype = dtype_from_name(cfg.compute_dtype), dtype_from_name(cfg.param_dtype)
        batch, length = input_ids.shape
        positions = jnp.arange(length, dtype=jnp.int32)[None, :] if positions is None else positions
        if positions.shape[0] == 1 and batch != 1:
            positions = jnp.broadcast_to(positions, (batch, length))

        embedding = nn.Embed(
            cfg.vocab_size,
            cfg.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=nn.initializers.normal(0.02),
            name="embed_tokens",
        )
        x = embedding(input_ids)
        layer_type = HybridDecoderLayer
        if cfg.remat_policy == "full":
            layer_type = nn.remat(HybridDecoderLayer, prevent_cse=False)
        hidden_states = []
        for index in range(cfg.num_layers):
            x = layer_type(cfg, index, name=f"layers_{index}")(x, positions, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)
        x = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, param_dtype, name="norm")(x)
        if cfg.tie_word_embeddings:
            logits = embedding.attend(x.astype(param_dtype))
        else:
            logits = nn.Dense(
                cfg.vocab_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(0.02),
                name="lm_head",
            )(x)
        logits = logits.astype(dtype_from_name(cfg.logits_dtype))
        if return_hidden_states:
            return logits, tuple(hidden_states)
        return logits


def causal_lm_loss(logits: jax.Array, labels: jax.Array, loss_mask: jax.Array | None = None) -> jax.Array:
    """Next-token cross entropy with an optional [batch, length] mask."""
    targets = labels[:, 1:]
    token_loss = -jnp.take_along_axis(
        jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32)), targets[..., None], axis=-1
    ).squeeze(-1)
    if loss_mask is None:
        return token_loss.mean()
    weights = loss_mask[:, 1:].astype(jnp.float32)
    return jnp.sum(token_loss * weights) / jnp.maximum(weights.sum(), 1.0)
