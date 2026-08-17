import numpy as np

from singularity.activation_compression import (
    bkv_balance_ratio,
    fit_activation_pca,
    fit_activation_pca_jax,
    project_activations,
)


def test_full_rank_activation_pca_reconstructs_and_bkv_balances_norms():
    rng = np.random.default_rng(10)
    samples = rng.normal(size=(12, 6)).astype(np.float32)
    basis = fit_activation_pca(samples, 6, seed=2)
    reconstructed = project_activations(samples, basis)
    np.testing.assert_allclose(reconstructed, samples, atol=2e-5, rtol=2e-5)

    key = 4.0 * rng.normal(size=(3, 4, 5)).astype(np.float32)
    value = rng.normal(size=(3, 4, 7)).astype(np.float32)
    ratio = bkv_balance_ratio(key, value)
    balanced_key_norm = np.mean(
        np.linalg.norm((key / ratio).reshape(-1, key.shape[-1]), axis=-1)
    )
    value_norm = np.mean(
        np.linalg.norm(value.reshape(-1, value.shape[-1]), axis=-1)
    )
    np.testing.assert_allclose(balanced_key_norm, value_norm, rtol=1e-6)

    jax_basis = fit_activation_pca_jax(samples, 6, seed=2)
    jax_reconstructed = np.asarray(samples @ np.asarray(jax_basis).T @ np.asarray(jax_basis))
    np.testing.assert_allclose(jax_reconstructed, samples, atol=2e-5, rtol=2e-5)
