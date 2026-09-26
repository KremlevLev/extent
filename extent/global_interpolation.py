"""Whole-model BF16 parameter interpolation for post-recovery geometry probes."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def interpolate_parameters(start, trained, alpha: float):
    """Interpolate all trainable leaves in FP32, then restore their storage dtype."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0.0:
        return start
    if alpha == 1.0:
        return trained

    def mix(original, candidate):
        if original.shape != candidate.shape or original.dtype != candidate.dtype:
            raise ValueError("interpolation trees must have matching leaves")
        value = original.astype(jnp.float32) + jnp.asarray(alpha, jnp.float32) * (
            candidate.astype(jnp.float32) - original.astype(jnp.float32)
        )
        return value.astype(original.dtype)

    return jax.tree.map(mix, start, trained)


def select_alpha(calibration_nll: dict[float, float], *, min_gain: float) -> float:
    """Choose on calibration only; keep the untouched model for tiny gains."""
    if min_gain < 0 or 0.0 not in calibration_nll:
        raise ValueError("positive gain threshold and alpha zero are required")
    best = min(calibration_nll, key=lambda alpha: (calibration_nll[alpha], alpha))
    if calibration_nll[best] > calibration_nll[0.0] - min_gain:
        return 0.0
    return float(best)
