from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RidgeReadoutReport:
    samples: int
    feature_width: int
    target_width: int
    relative_ridge: float
    absolute_ridge: float
    calibration_relative_l2: float


def fit_dual_ridge_readout(
    features: jax.Array,
    targets: jax.Array,
    *,
    relative_ridge: float = 1e-3,
) -> tuple[jax.Array, RidgeReadoutReport]:
    """Fit an uncentered no-bias readout using the sample-space ridge solve."""
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be rank-two matrices")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same sample count")
    if relative_ridge <= 0:
        raise ValueError("relative_ridge must be positive")
    x = features.astype(jnp.float32)
    y = targets.astype(jnp.float32)
    gram = x @ x.T
    mean_diagonal = jnp.trace(gram) / gram.shape[0]
    ridge = relative_ridge * jnp.maximum(mean_diagonal, jnp.finfo(jnp.float32).tiny)
    dual = jnp.linalg.solve(gram + ridge * jnp.eye(gram.shape[0]), y)
    kernel = x.T @ dual
    reconstruction = x @ kernel
    relative_l2 = jnp.linalg.norm(reconstruction - y) / jnp.maximum(
        jnp.linalg.norm(y), jnp.finfo(jnp.float32).tiny
    )
    jax.block_until_ready((kernel, relative_l2, ridge))
    report = RidgeReadoutReport(
        samples=features.shape[0],
        feature_width=features.shape[1],
        target_width=targets.shape[1],
        relative_ridge=relative_ridge,
        absolute_ridge=float(ridge),
        calibration_relative_l2=float(relative_l2),
    )
    return kernel, report
