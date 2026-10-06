"""EXP-104 FP32 master weights, fixed readout, sharded AdamW updates."""
import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze
from flax import traverse_util
from extent.full_model_distillation import causal_cross_entropy
from extent.recovery_subspace import apply_corrections, coordinates_finite
from extent.stable_gradient_clip import stable_clip_by_global_norm, scaled_norm_parts

FROZEN = frozenset(("embed_tokens", "lm_head"))


def split_decoder(parameters):
    tree = unfreeze(parameters)
    return ({k: jax.tree.map(lambda x: x.astype(jnp.float32), v)
             for k, v in tree.items() if k not in FROZEN},
            {k: v for k, v in tree.items() if k in FROZEN})


def decoder_parameters(masters, frozen):
    return dict(jax.tree.map(lambda x: x.astype(jnp.bfloat16), masters), **frozen)


def schedule(step):
    step = jnp.asarray(step, jnp.float32)
    fraction = jnp.clip((step - 64) / (2048 - 64), 0, 1)
    return jnp.where(step < 64, 1e-5 * step / 64,
                     1e-6 + 9e-6 * (1 + jnp.cos(jnp.pi * fraction)) / 2)


def optimizer(parameters):
    # Matrix-only decay includes both adapter A/B matrices. Norm/bias vectors
    # and frozen vocabulary weights are excluded, not just zeroed gradients.
    mask = traverse_util.unflatten_dict({path: x.ndim >= 2 and not path[-1].endswith("bias")
        for path, x in traverse_util.flatten_dict(parameters).items()})
    return optax.chain(stable_clip_by_global_norm(1),
        optax.scale_by_adam(b1=.9, b2=.95, eps=1e-8),
        optax.add_decayed_weights(.1, mask=mask), optax.scale_by_schedule(lambda n: -schedule(n)))


def forward_parameters(arm, trainable, fixed, columns, head_dim):
    if arm == "DECODER-CE":
        return decoder_parameters(trainable, fixed)
    if arm == "ADAPTER-CE":
        return apply_corrections(fixed, trainable, head_dim=head_dim,
                                 protected_input_columns=columns)
    raise ValueError(arm)


def make_step(model, tx, arm, columns, head_dim, layouts=None):
    def step(trainable, state, fixed, tokens):
        def loss(p):
            logits = model.apply({"params": forward_parameters(arm, p, fixed, columns, head_dim)}, tokens)
            return causal_cross_entropy(logits, tokens), jnp.all(jnp.isfinite(logits))
        (value, forward_finite), grad = jax.value_and_grad(loss, has_aux=True)(trainable)
        updates, proposed_state = tx.update(grad, state, trainable)
        proposed = optax.apply_updates(trainable, updates)
        maximum, scaled, logarithm = scaled_norm_parts(grad)
        finite = (forward_finite & jnp.isfinite(value) & coordinates_finite(grad)
                  & coordinates_finite(proposed) & coordinates_finite(proposed_state))
        health = dict(loss=value, finite=finite, grad_max_abs=maximum,
                      scaled_norm=scaled, log10_grad_norm=logarithm,
                      naive_norm_finite=jnp.isfinite(optax.global_norm(grad)),
                      update_max_abs=jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in jax.tree.leaves(updates)])))
        return proposed, proposed_state, health
    # No donation: rejected proposals must leave the previous safe state usable.
    if layouts is None:
        return jax.jit(step)
    p, s, f, b, replicated = layouts
    return jax.jit(step, in_shardings=(p, s, f, b), out_shardings=(p, s, replicated))
