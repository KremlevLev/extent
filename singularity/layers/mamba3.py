"""Readable JAX reference implementation of the Mamba-3 MIMO recurrence.

This is deliberately a correctness/bring-up path.  ``lax.scan`` compiles on TPU,
but a production run should replace only ``mamba3_reference_scan`` with a Pallas
chunked kernel while keeping the module and parameter contract unchanged.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from singularity.config import Mamba3Config
from singularity.layers.common import RMSNorm


# Formula/parameter contract checked against state-spaces/mamba at
# commit e9594ce1c732d97440f0332fdc43170a2294dbfa.
MAMBA3_REFERENCE_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


class ScanState(NamedTuple):
    ssm: jax.Array
    previous_k: jax.Array
    previous_v: jax.Array
    angle: jax.Array


def heavy_tail_activation(x: jax.Array) -> jax.Array:
    """Positive Mamba-3 decay activation, evaluated in float32."""
    x = x.astype(jnp.float32)
    return jnp.maximum(x, 0.0) + jax.lax.reciprocal(1.0 - jnp.minimum(x, 0.0))


def _rotate_mimo_halves(x: jax.Array, angle: jax.Array, rotary_pairs: int) -> jax.Array:
    """Official MIMO split-half rotation (rather than SISO interleaved pairs)."""
    half = x.shape[-1] // 2
    real, imag = x[..., :half], x[..., half:]
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    while cos.ndim < real.ndim:
        cos = jnp.expand_dims(cos, axis=1)
        sin = jnp.expand_dims(sin, axis=1)
    if rotary_pairs < half:
        padding = [(0, 0)] * cos.ndim
        padding[-1] = (0, half - rotary_pairs)
        cos = jnp.pad(cos, padding, constant_values=1.0)
        sin = jnp.pad(sin, padding, constant_values=0.0)
    return jnp.concatenate((real * cos - imag * sin, real * sin + imag * cos), axis=-1)


def mamba3_reference_scan(
    x: jax.Array,
    z: jax.Array,
    b: jax.Array,
    c: jax.Array,
    dt: jax.Array,
    decay: jax.Array,
    trap: jax.Array,
    angle_step: jax.Array,
    mimo_x: jax.Array,
    mimo_z: jax.Array,
    mimo_o: jax.Array,
    skip: jax.Array,
    rotary_pairs: int,
) -> jax.Array:
    """Scan one sequence.

    Shapes: x/z=[B,L,H,P], b/c=[B,L,R,H,N], scalar terms=[B,L,H].
    The recurrent state is float32 for stability; returned activations match x.
    """
    batch, _, heads, head_dim = x.shape
    rank, state_dim = b.shape[2], b.shape[-1]
    initial = ScanState(
        ssm=jnp.zeros((batch, heads, head_dim, state_dim), jnp.float32),
        previous_k=jnp.zeros((batch, rank, heads, state_dim), jnp.float32),
        previous_v=jnp.zeros((batch, heads, head_dim), jnp.float32),
        angle=jnp.zeros((batch, heads, rotary_pairs), jnp.float32),
    )
    mimo_x_rank_first = jnp.swapaxes(mimo_x, 0, 1).astype(jnp.float32)
    mimo_z_rank_first = jnp.swapaxes(mimo_z, 0, 1).astype(jnp.float32)
    mimo_o_rank_first = jnp.swapaxes(mimo_o, 0, 1).astype(jnp.float32)

    time_major = tuple(jnp.swapaxes(t, 0, 1) for t in (x, z, b, c, dt, decay, trap, angle_step))

    def step(state: ScanState, inputs: tuple[jax.Array, ...]):
        x_t, z_t, b_t, c_t, dt_t, decay_t, trap_t, angle_t = inputs
        dt_t = dt_t.astype(jnp.float32)
        alpha = jnp.exp(decay_t.astype(jnp.float32) * dt_t)
        gamma = jax.nn.sigmoid(trap_t.astype(jnp.float32)) * dt_t
        beta = (1.0 - jax.nn.sigmoid(trap_t.astype(jnp.float32))) * dt_t * alpha

        angle_increment = jnp.pi * angle_t[:, None, :].astype(jnp.float32)
        new_angle = state.angle + angle_increment * dt_t[..., None]
        b_rot = _rotate_mimo_halves(b_t.astype(jnp.float32), new_angle, rotary_pairs)
        c_rot = _rotate_mimo_halves(c_t.astype(jnp.float32), new_angle, rotary_pairs)

        # Rank-specific value streams; the MIMO rank is reduced only after readout.
        value_now = x_t[:, None].astype(jnp.float32) * mimo_x_rank_first[None]
        value_prev = state.previous_v[:, None] * mimo_x_rank_first[None]
        injection = jnp.einsum("brhp,brhn->bhpn", gamma[:, None, :, None] * value_now, b_rot)
        previous = jnp.einsum("brhp,brhn->bhpn", beta[:, None, :, None] * value_prev, state.previous_k)
        ssm = alpha[:, :, None, None] * state.ssm + injection + previous
        y_rank = jnp.einsum("bhpn,brhn->brhp", ssm, c_rot)
        y_rank += skip[None, None, :, None].astype(jnp.float32) * value_now
        gated = y_rank * jax.nn.silu(
            z_t[:, None].astype(jnp.float32) * mimo_z_rank_first[None]
        )
        y = jnp.sum(gated * mimo_o_rank_first[None], axis=1)
        next_state = ScanState(ssm, b_rot, x_t.astype(jnp.float32), new_angle)
        return next_state, y.astype(x_t.dtype)

    _, output = jax.lax.scan(step, initial, time_major)
    return jnp.swapaxes(output, 0, 1)


class Mamba3MIMO(nn.Module):
    hidden_size: int
    config: Mamba3Config
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self, inputs: jax.Array, *, return_features: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        cfg = self.config
        inner = int(self.hidden_size * cfg.expand)
        heads = inner // cfg.head_dim
        rank = cfg.mimo_rank
        rotary_pairs = int(cfg.d_state * cfg.rope_fraction) // 2
        projection_size = 2 * inner + 2 * rank * cfg.groups * cfg.d_state + 3 * heads + rotary_pairs

        projected = nn.Dense(
            projection_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(0.02),
            name="in_proj",
        )(inputs)
        sizes = (inner, inner, rank * cfg.groups * cfg.d_state, rank * cfg.groups * cfg.d_state,
                 heads, heads, heads, rotary_pairs)
        split_indices: list[int] = []
        offset = 0
        for size in sizes[:-1]:
            offset += size
            split_indices.append(offset)
        z, x, b, c, raw_dt, raw_a, trap, angle = jnp.split(projected, split_indices, axis=-1)
        batch, length, _ = inputs.shape
        x = x.reshape(batch, length, heads, cfg.head_dim)
        z = z.reshape(batch, length, heads, cfg.head_dim)
        b = b.reshape(batch, length, rank, cfg.groups, cfg.d_state)
        c = c.reshape(batch, length, rank, cfg.groups, cfg.d_state)
        if cfg.groups == 1:
            b = jnp.broadcast_to(b, (batch, length, rank, heads, cfg.d_state))
            c = jnp.broadcast_to(c, (batch, length, rank, heads, cfg.d_state))
        elif heads % cfg.groups == 0:
            b = jnp.repeat(b, heads // cfg.groups, axis=3)
            c = jnp.repeat(c, heads // cfg.groups, axis=3)
        else:
            raise ValueError("Mamba heads must be divisible by groups")

        b = RMSNorm(cfg.d_state, eps=1e-5, param_dtype=self.param_dtype, name="b_norm")(b)
        c = RMSNorm(cfg.d_state, eps=1e-5, param_dtype=self.param_dtype, name="c_norm")(c)
        b_bias = self.param("b_bias", nn.initializers.ones, (heads, rank, cfg.d_state), self.param_dtype)
        c_bias = self.param("c_bias", nn.initializers.ones, (heads, rank, cfg.d_state), self.param_dtype)
        b = b + jnp.transpose(b_bias, (1, 0, 2))[None, None]
        c = c + jnp.transpose(c_bias, (1, 0, 2))[None, None]

        log_dt_min, log_dt_max = math.log(cfg.dt_min), math.log(cfg.dt_max)

        def dt_bias_init(key, shape, dtype):
            log_dt = jax.random.uniform(
                key,
                shape,
                dtype=jnp.float32,
                minval=log_dt_min,
                maxval=log_dt_max,
            )
            initial_dt = jnp.maximum(jnp.exp(log_dt), cfg.dt_init_floor)
            inverse_softplus = initial_dt + jnp.log(-jnp.expm1(-initial_dt))
            return inverse_softplus.astype(dtype)

        dt_bias = self.param(
            "dt_bias", dt_bias_init, (heads,), self.param_dtype
        ).astype(jnp.float32)
        dt = jax.nn.softplus(raw_dt.astype(jnp.float32) + dt_bias)
        decay = -heavy_tail_activation(raw_a)
        decay = jnp.minimum(decay, -cfg.a_floor)

        mimo_x = self.param(
            "mimo_x",
            lambda *_: jnp.full(
                (heads, rank, cfg.head_dim), 1 / rank, self.param_dtype
            ),
        )
        mimo_z = self.param(
            "mimo_z",
            nn.initializers.ones,
            (heads, rank, cfg.head_dim),
            self.param_dtype,
        )
        mimo_o = self.param(
            "mimo_o",
            lambda *_: jnp.full(
                (heads, rank, cfg.head_dim), 1 / rank, self.param_dtype
            ),
        )
        skip = self.param("D", nn.initializers.ones, (heads,), self.param_dtype)
        y = mamba3_reference_scan(
            x,
            z,
            b,
            c,
            dt,
            decay,
            trap,
            angle,
            mimo_x,
            mimo_z,
            mimo_o,
            skip,
            rotary_pairs,
        )
        features = y.reshape(batch, length, inner)
        output = nn.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(0.02),
            name="out_proj",
        )(features)
        return (output, features) if return_features else output
