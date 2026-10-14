"""Local teacher-mixer fitting on actual patient inputs; full NLL is separate."""
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import unfreeze
from extent.layers.common import dtype_from_name
from extent.layers.mamba3 import Mamba3MIMO
from extent.plateau_forensics import norm
from extent.stable_gradient_clip import stable_clip_by_global_norm, scaled_norm_parts

RECIPES = ('ADD32', 'ADD64', 'ADD128', 'ADD256', 'FREE256', 'ALL-MAMBA')
LAYERS = (0, 12, 26)


def initialize(p, recipe, seed=119123):
    if recipe == 'ALL-MAMBA':
        return jax.tree.map(lambda x: x.astype(jnp.float32), p['mamba'])
    rank = int(recipe.replace('ADD', '').replace('FREE', ''))
    q = {}
    for i, name in enumerate(('in_proj','out_proj')):
        kernel = p['mamba'][name]['kernel']
        a = jax.random.normal(jax.random.fold_in(jax.random.key(seed), i),
            (kernel.shape[0], 256), jnp.float32) / math.sqrt(kernel.shape[0])
        q[name] = dict(a=a[:, :rank], b=jnp.zeros((rank, kernel.shape[1]), jnp.float32))
    return q


def materialize(p, q, recipe, columns):
    result = unfreeze(p)
    if recipe == 'ALL-MAMBA':
        result['mamba'] = jax.tree.map(lambda x: x.astype(jnp.bfloat16), q)
    else:
        for name, factors in q.items():
            kernel = p['mamba'][name]['kernel']
            b = factors['b']
            if name == 'in_proj' and recipe.startswith('ADD'):
                b = b * (jnp.arange(b.shape[1]) < columns)[None]
            result['mamba'][name]['kernel'] = (kernel.astype(jnp.float32) + factors['a'] @ b).astype(kernel.dtype)
    return result


def fit_step(cfg, recipe, columns):
    tx = optax.chain(stable_clip_by_global_norm(1), optax.adam(3e-4, b1=.9, b2=.999))
    module = Mamba3MIMO(cfg.hidden_size, cfg.mamba, dtype_from_name(cfg.compute_dtype), jnp.bfloat16)
    def loss(q, p, x, target):
        m = materialize(p, q, recipe, columns)['mamba']
        y = module.apply({'params': m}, x).astype(jnp.float32)
        # Scale before squaring, then relative energy; avoids overflow for
        # large teacher targets, no gradient through the fixed normalization.
        scale = jnp.maximum(jnp.max(jnp.abs(target)), 1e-6)
        y, t = y/scale, target.astype(jnp.float32)/scale
        return jnp.mean((y-t)**2)/jnp.maximum(jnp.mean(t*t), 1e-8)
    def step(q, state, p, x, target):
        value, grad = jax.value_and_grad(loss)(q, p, x, target)
        updates, after = tx.update(grad, state, q)
        proposed = optax.apply_updates(q, updates)
        finite = jnp.isfinite(value) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x))
            for x in jax.tree.leaves((grad, proposed, after))]))
        return proposed, after, dict(loss=value, finite=finite, log10_grad_norm=scaled_norm_parts(grad)[2])
    return tx, jax.jit(step), jax.jit(loss)
