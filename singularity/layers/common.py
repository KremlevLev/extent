from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def dtype_from_name(name: str) -> jnp.dtype:
    try:
        return {"bfloat16": jnp.bfloat16, "float32": jnp.float32, "float16": jnp.float16}[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


class RMSNorm(nn.Module):
    features: int
    eps: float = 1e-6
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param("scale", nn.initializers.ones, (self.features,), self.param_dtype)
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normalized = x * jax.lax.rsqrt(variance + self.eps).astype(x.dtype)
        return normalized * scale.astype(x.dtype)


class SwiGLU(nn.Module):
    hidden_size: int
    intermediate_size: int
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dense = lambda features, name: nn.Dense(
            features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(0.02),
            name=name,
        )
        gate = dense(self.intermediate_size, "gate_proj")(x)
        up = dense(self.intermediate_size, "up_proj")(x)
        return dense(self.hidden_size, "down_proj")(jax.nn.silu(gate) * up)
