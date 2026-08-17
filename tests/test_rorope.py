import jax.numpy as jnp
import numpy as np

from singularity.qwen3_teacher import apply_qwen3_rope
from singularity.rorope import (
    fit_freqfold_rotations,
    fit_rorope_rotations,
    freqfold_attend,
    rorope_attend,
)


def test_all_rorope_components_preserve_gqa_attention():
    rng = np.random.default_rng(8)
    query = jnp.asarray(rng.normal(size=(1, 5, 4, 8)).astype(np.float32))
    key = jnp.asarray(rng.normal(size=(1, 5, 2, 8)).astype(np.float32))
    value = jnp.asarray(rng.normal(size=(1, 5, 2, 8)).astype(np.float32))
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]
    mapping = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    rotations, eigenvalues = fit_rorope_rotations(np.asarray(key))

    candidate = rorope_attend(
        query, key, value, positions, jnp.asarray(rotations), mapping, 2, 10_000.0
    )
    source_query = apply_qwen3_rope(query, positions, 10_000.0)
    source_key = apply_qwen3_rope(key, positions, 10_000.0)[:, :, mapping]
    logits = jnp.einsum("bqhd,bkhd->bhqk", source_query, source_key) * (8**-0.5)
    causal = positions[:, None, :, None] >= positions[:, None, None, :]
    probabilities = jnp.asarray(
        jnp.where(causal, logits, jnp.finfo(jnp.float32).min)
    )
    probabilities = jnp.exp(probabilities - probabilities.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    reference = jnp.einsum(
        "bhqk,bkhd->bqhd", probabilities, value[:, :, mapping]
    )

    np.testing.assert_allclose(candidate, reference, atol=2e-5, rtol=2e-5)
    assert rotations.shape == (4, 2, 2)
    assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])


def test_freqfold_one_matches_standard_one_component_rorope():
    rng = np.random.default_rng(9)
    query = jnp.asarray(rng.normal(size=(1, 4, 4, 8)).astype(np.float32))
    key = jnp.asarray(rng.normal(size=(1, 4, 2, 8)).astype(np.float32))
    value = jnp.asarray(rng.normal(size=(1, 4, 2, 8)).astype(np.float32))
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mapping = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    standard, _ = fit_rorope_rotations(np.asarray(key))
    folded, eigenvalues = fit_freqfold_rotations(np.asarray(key), 1)

    expected = rorope_attend(
        query, key, value, positions, jnp.asarray(standard), mapping, 1, 10_000.0
    )
    candidate = freqfold_attend(
        query,
        key,
        value,
        positions,
        jnp.asarray(folded),
        mapping,
        1,
        10_000.0,
    )

    np.testing.assert_allclose(candidate, expected, atol=2e-5, rtol=2e-5)
    assert folded.shape == (4, 2, 2)
    assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])
