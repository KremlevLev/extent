"""FP32 global clipping without squaring large unscaled gradients."""
import jax
import jax.numpy as jnp
import optax

def scaled_norm_parts(tree):
    leaves = jax.tree.leaves(tree)
    maximum = jnp.max(jnp.stack([jnp.max(jnp.abs(x.astype(jnp.float32))) for x in leaves]))
    denominator = jnp.where(maximum > 0, maximum, 1.0)
    scaled = jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32) / denominator)) for x in leaves))
    log10 = jnp.where(maximum > 0, jnp.log10(denominator) + jnp.log10(jnp.maximum(scaled, 1e-30)), -30.0)
    return maximum, scaled, log10

def stable_clip_by_global_norm(limit=1.0):
    if limit <= 0:
        raise ValueError("clip limit must be positive")
    def update(updates, state, params=None):
        maximum, scaled, _ = scaled_norm_parts(updates)
        denominator = jnp.where(maximum > 0, maximum, 1.0)
        threshold = limit / jnp.maximum(scaled, 1e-30)
        # Do not form maximum*scaled or square the original gradients.
        clipped = jax.tree.map(lambda g: jnp.where(maximum > threshold,
            (g / denominator) * threshold, g), updates)
        return clipped, state
    return optax.GradientTransformation(lambda _: optax.EmptyState(), update)
