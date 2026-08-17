from __future__ import annotations

import numpy as np

from singularity.weight_mapping import truncated_svd


def bkv_balance_ratio(key: np.ndarray, value: np.ndarray) -> float:
    """TransMLA BKV ratio E||K|| / E||V|| over token activations."""
    key = np.asarray(key, dtype=np.float32).reshape(-1, key.shape[-1])
    value = np.asarray(value, dtype=np.float32).reshape(-1, value.shape[-1])
    key_norm = float(np.mean(np.linalg.norm(key, axis=-1)))
    value_norm = float(np.mean(np.linalg.norm(value, axis=-1)))
    return key_norm / max(value_norm, np.finfo(np.float32).tiny)


def fit_activation_pca(
    samples: np.ndarray, rank: int, *, seed: int = 0
) -> np.ndarray:
    """Fit an uncentered activation-PCA reconstruction basis."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or not 0 < rank <= min(samples.shape):
        raise ValueError("rank must fit the two-dimensional sample matrix")
    _, up = truncated_svd(
        samples,
        rank,
        exact_threshold=512,
        oversample=16,
        power_iterations=2,
        seed=seed,
    )
    norms = np.linalg.norm(up, axis=1, keepdims=True)
    return (up / np.maximum(norms, np.finfo(np.float32).tiny)).astype(np.float32)


def project_activations(samples: np.ndarray, basis: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    return (samples @ basis.T) @ basis
