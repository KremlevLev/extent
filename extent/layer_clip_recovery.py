"""EXP106/107 layer-group clipping; independent of legacy training contracts."""
import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from extent.decoder_recovery import forward_parameters
from extent.full_model_distillation import causal_cross_entropy
from extent.recovery_subspace import coordinates_finite
from extent.stable_gradient_clip import scaled_norm_parts, stable_clip_by_global_norm

TOTAL_STEPS = 32768
WARMUP = 512
AGC_RATIO = .01
MIN_WEIGHT_NORM = .001


def schedule(step):
    step = jnp.asarray(step, jnp.float32)
    fraction = jnp.clip((step - WARMUP) / (TOTAL_STEPS - WARMUP), 0, 1)
    return jnp.where(step < WARMUP, 1e-5 * step / WARMUP,
                     1e-6 + 9e-6 * (1 + jnp.cos(jnp.pi * fraction)) / 2)


def clip_gradients(gradients, parameters, adaptive):
    """A group is a complete decoder layer, or the final decoder norm.

    This is layer-group AGC, not NFNet unitwise AGC. No EMA/hidden state.
    All norm arithmetic is scaled FP32; NaN/Inf remain visible to the guard.
    """
    result, diagnostics = {}, {}
    global_max, global_scaled, global_log = scaled_norm_parts(gradients)
    global_fraction = jnp.minimum(1., jnp.power(10., -global_log))
    for name, gradient in gradients.items():
        maximum, scaled, logarithm = scaled_norm_parts(gradient)
        _, _, weight_log = scaled_norm_parts(parameters[name])
        limit_log = jnp.log10(AGC_RATIO) + jnp.maximum(weight_log, jnp.log10(MIN_WEIGHT_NORM))
        fraction = jnp.minimum(1., jnp.power(10., limit_log - logarithm)) if adaptive else global_fraction
        if adaptive:
            # Express the clipped result without forming the original norm.
            limit = jnp.power(10., limit_log)
            denominator = jnp.where(maximum > 0, maximum, 1.)
            threshold = limit / jnp.maximum(scaled, 1e-30)
            result[name] = jax.tree.map(lambda g: jnp.where(maximum > threshold,
                (g / denominator) * threshold, g), gradient)
        diagnostics[name] = dict(log10_grad_norm=logarithm, log10_weight_norm=weight_log,
                                 gradient_retained_fraction=fraction)
    if not adaptive:
        result = stable_clip_by_global_norm(1).update(gradients, optax.EmptyState())[0]
    return result, diagnostics


def optimizer(parameters, adaptive=False):
    mask = traverse_util.unflatten_dict({path: x.ndim >= 2 and not path[-1].endswith("bias")
        for path, x in traverse_util.flatten_dict(parameters).items()})
    def clip(updates, state, params):
        return clip_gradients(updates, params, adaptive)[0], state
    return optax.chain(optax.GradientTransformation(lambda _: optax.EmptyState(), clip),
        optax.scale_by_adam(b1=.9, b2=.95, eps=1e-8),
        optax.add_decayed_weights(.1, mask=mask), optax.scale_by_schedule(lambda n: -schedule(n)))


def make_step(model, tx, arm, columns, head_dim, layouts=None, *, adaptive=False):
    def step(parameters, state, fixed, tokens):
        def loss(p):
            logits = model.apply({"params": forward_parameters(arm, p, fixed, columns, head_dim)}, tokens)
            return causal_cross_entropy(logits, tokens), jnp.all(jnp.isfinite(logits))
        (value, forward_finite), gradients = jax.value_and_grad(loss, has_aux=True)(parameters)
        updates, proposed_state = tx.update(gradients, state, parameters)
        proposed = optax.apply_updates(parameters, updates)
        _, diagnostics = clip_gradients(gradients, parameters, adaptive)
        for name in diagnostics:
            _, _, update_log = scaled_norm_parts(updates[name])
            diagnostics[name]["log10_update_norm"] = update_log
            diagnostics[name]["log10_relative_update"] = update_log - jnp.maximum(
                diagnostics[name]["log10_weight_norm"], jnp.log10(MIN_WEIGHT_NORM))
        maximum, scaled, logarithm = scaled_norm_parts(gradients)
        finite = (forward_finite & jnp.isfinite(value) & coordinates_finite(gradients)
                  & coordinates_finite(proposed) & coordinates_finite(proposed_state))
        health = dict(loss=value, finite=finite, grad_max_abs=maximum,
            scaled_norm=scaled, log10_grad_norm=logarithm,
            naive_norm_finite=jnp.isfinite(optax.global_norm(gradients)), layers=diagnostics,
            update_max_abs=jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in jax.tree.leaves(updates)])))
        return proposed, proposed_state, health
    if layouts is None:
        return jax.jit(step)
    p, s, f, b, replicated = layouts
    return jax.jit(step, in_shardings=(p, s, f, b), out_shardings=(p, s, replicated))
