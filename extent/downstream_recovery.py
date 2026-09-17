"""Cached-prefix, frozen-suffix distillation with candidate-only gradients."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from extent.layers.common import RMSNorm, dtype_from_name
from extent.model import HybridDecoderLayer
from extent.full_model_distillation import forward_kl
from extent.optimizer import cast_grads_bf16, gradient_health


class HybridDecoderSuffix(nn.Module):
    config: object
    first_layer: int

    @nn.compact
    def __call__(self, hidden):
        cfg = self.config
        positions = jnp.arange(hidden.shape[1], dtype=jnp.int32)[None]
        layer_type = nn.remat(HybridDecoderLayer, prevent_cse=False)
        for index in range(self.first_layer, cfg.num_layers):
            hidden = layer_type(cfg, index, name=f"layers_{index}")(hidden, positions, None)
        hidden = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                         dtype_from_name(cfg.param_dtype), name="norm")(hidden)
        if cfg.tie_word_embeddings:
            embedding = nn.Embed(cfg.vocab_size, cfg.hidden_size,
                                 dtype=dtype_from_name(cfg.compute_dtype),
                                 param_dtype=dtype_from_name(cfg.param_dtype),
                                 name="embed_tokens")
            logits = embedding.attend(hidden.astype(dtype_from_name(cfg.param_dtype)))
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False,
                              dtype=dtype_from_name(cfg.compute_dtype),
                              param_dtype=dtype_from_name(cfg.param_dtype),
                              name="lm_head")(hidden)
        return logits.astype(jnp.float32)


def suffix_parameters(params, first_layer):
    return {key: value for key, value in params.items()
            if not key.startswith("layers_") or int(key.split("_")[1]) >= first_layer}


def inject_candidates(frozen, candidate):
    params = dict(frozen)
    for key, mamba in candidate.items():
        params[key] = dict(params[key], mamba=mamba)
    return params


def make_downstream_recovery_step(suffix, tx, *, bf16_gradients=True):
    """No optimizer slots or parameter gradients for the frozen decoder suffix."""
    def step(candidate, opt_state, frozen, prefix_hidden, teacher_logits):
        def loss_fn(value):
            logits = suffix.apply({"params": inject_candidates(frozen, value)}, prefix_hidden)
            return forward_kl(logits, jax.lax.stop_gradient(teacher_logits), temperature=1.0)

        loss, grads = jax.value_and_grad(loss_fn)(candidate)
        health = gradient_health(grads)
        updates, opt_state = tx.update(
            cast_grads_bf16(grads) if bf16_gradients else grads, opt_state, candidate,
        )
        return optax.apply_updates(candidate, updates), opt_state, {"loss": loss, **health}

    return step
