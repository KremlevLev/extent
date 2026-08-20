from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def fit_rorope_rotations(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit standard RoRoPE head rotations from unrotated normalized keys.

    ``keys`` has shape ``[..., kv_heads, head_dim]`` and uses Qwen's split-half
    real/imaginary layout. One orthogonal rotation is fitted per RoPE frequency.
    """
    keys = np.asarray(keys, dtype=np.float64)
    if keys.ndim < 3 or keys.shape[-1] % 2:
        raise ValueError("keys must end in [kv_heads, even_head_dim]")
    samples = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
    real, imaginary = np.split(samples, 2, axis=-1)
    covariance = np.einsum("ngp,nhp->pgh", real, real)
    covariance += np.einsum("ngp,nhp->pgh", imaginary, imaginary)

    rotations = []
    eigenvalues = []
    for matrix in covariance:
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        # Eigenvector signs are arbitrary. Canonical signs make artifacts stable.
        anchors = np.argmax(np.abs(vectors), axis=0)
        signs = np.sign(vectors[anchors, np.arange(vectors.shape[1])])
        vectors *= np.where(signs == 0, 1.0, signs)
        rotations.append(vectors)
        eigenvalues.append(values)
    return np.asarray(rotations, dtype=np.float32), np.asarray(
        eigenvalues, dtype=np.float64
    )


def fit_freqfold_rotations(
    keys: np.ndarray, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit FreqFold PCA over adjacent RoPE frequencies and KV heads."""
    keys = np.asarray(keys, dtype=np.float64)
    head_dim = keys.shape[-1]
    if keys.ndim < 3 or head_dim % 2 or (head_dim // 2) % fold:
        raise ValueError("fold must divide the number of RoPE frequency pairs")
    samples = keys.reshape(-1, keys.shape[-2], head_dim)
    real, imaginary = np.split(samples, 2, axis=-1)
    groups = head_dim // 2 // fold
    real = real.reshape(samples.shape[0], samples.shape[1], groups, fold)
    imaginary = imaginary.reshape(samples.shape[0], samples.shape[1], groups, fold)
    real = real.transpose(0, 2, 1, 3).reshape(samples.shape[0], groups, -1)
    imaginary = imaginary.transpose(0, 2, 1, 3).reshape(
        samples.shape[0], groups, -1
    )
    covariance = np.einsum("ngd,nge->gde", real, real)
    covariance += np.einsum("ngd,nge->gde", imaginary, imaginary)
    rotations = []
    eigenvalues = []
    for matrix in covariance:
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        anchors = np.argmax(np.abs(vectors), axis=0)
        signs = np.sign(vectors[anchors, np.arange(vectors.shape[1])])
        vectors *= np.where(signs == 0, 1.0, signs)
        rotations.append(vectors)
        eigenvalues.append(values)
    return np.asarray(rotations, dtype=np.float32), np.asarray(
        eigenvalues, dtype=np.float64
    )


def rotate_rorope_key(key: jax.Array, rotations: jax.Array) -> jax.Array:
    """Apply per-frequency KV-head PCA without positional rotations."""
    kv_heads, head_dim = key.shape[-2:]
    pairs = head_dim // 2
    if rotations.shape != (pairs, kv_heads, kv_heads):
        raise ValueError("rotations have an incompatible shape")
    real, imaginary = jnp.split(key.astype(jnp.float32), 2, axis=-1)
    real = jnp.einsum("blgp,pgc->blcp", real, rotations)
    imaginary = jnp.einsum("blgp,pgc->blcp", imaginary, rotations)
    return jnp.concatenate((real, imaginary), axis=-1)


def apply_rorope(
    query: jax.Array,
    key: jax.Array,
    positions: jax.Array,
    rotations: jax.Array,
    query_to_kv: jax.Array,
    rope_components: int,
    theta: float,
) -> tuple[jax.Array, jax.Array]:
    """Rotate KV-head components and retain RoPE on leading PCA components."""
    kv_heads, head_dim = key.shape[-2:]
    pairs = head_dim // 2
    if not 0 < rope_components <= kv_heads:
        raise ValueError("rope_components must be within the KV-head count")
    if rotations.shape != (pairs, kv_heads, kv_heads):
        raise ValueError("rotations have an incompatible shape")

    q_real, q_imaginary = jnp.split(query.astype(jnp.float32), 2, axis=-1)
    key = rotate_rorope_key(key, rotations)
    k_real, k_imaginary = jnp.split(key, 2, axis=-1)
    q_rotation = rotations[:, query_to_kv, :].transpose(1, 0, 2)
    q_rotation = q_rotation.transpose(0, 2, 1)
    q_real = q_real[..., None, :] * q_rotation[None, None, :, :, :]
    q_imaginary = (
        q_imaginary[..., None, :] * q_rotation[None, None, :, :, :]
    )

    frequencies = jnp.arange(pairs, dtype=jnp.float32)
    inverse = theta ** (-(2.0 * frequencies) / head_dim)
    phase = positions.astype(jnp.float32)[..., None] * inverse
    cosine, sine = jnp.cos(phase), jnp.sin(phase)
    q_cos = cosine[:, :, None, None, :]
    q_sin = sine[:, :, None, None, :]
    k_cos = cosine[:, :, None, :]
    k_sin = sine[:, :, None, :]
    q_rotated_real = q_real * q_cos - q_imaginary * q_sin
    q_rotated_imaginary = q_imaginary * q_cos + q_real * q_sin
    k_rotated_real = k_real * k_cos - k_imaginary * k_sin
    k_rotated_imaginary = k_imaginary * k_cos + k_real * k_sin
    component_mask = jnp.arange(kv_heads) < rope_components
    q_mask = component_mask[None, None, None, :, None]
    k_mask = component_mask[None, None, :, None]
    q_real = jnp.where(q_mask, q_rotated_real, q_real)
    q_imaginary = jnp.where(q_mask, q_rotated_imaginary, q_imaginary)
    k_real = jnp.where(k_mask, k_rotated_real, k_real)
    k_imaginary = jnp.where(k_mask, k_rotated_imaginary, k_imaginary)
    return (
        jnp.concatenate((q_real, q_imaginary), axis=-1),
        jnp.concatenate((k_real, k_imaginary), axis=-1),
    )


def rorope_attend(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    positions: jax.Array,
    rotations: jax.Array,
    query_to_kv: jax.Array,
    rope_components: int,
    theta: float,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
    """Reference TransMLA RoRoPE attention before joint KV compression."""
    query, key = apply_rorope(
        query,
        key,
        positions,
        rotations,
        query_to_kv,
        rope_components,
        theta,
    )
    logits = jnp.einsum(
        "bqhcd,bkcd->bhqk", query, key, preferred_element_type=jnp.float32
    ) * (value.shape[-1] ** -0.5)
    causal = positions[:, None, :, None] >= positions[:, None, None, :]
    mask = causal
    if attention_mask is not None:
        supplied = attention_mask.astype(jnp.bool_)
        if supplied.ndim == 2:
            supplied = supplied[:, None, None, :]
        mask = mask & supplied
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    probabilities = jax.nn.softmax(logits, axis=-1)
    expanded_value = value[:, :, query_to_kv, :]
    return jnp.einsum("bhqk,bkhd->bqhd", probabilities, expanded_value)


def freqfold_attend(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    positions: jax.Array,
    rotations: jax.Array,
    query_to_kv: jax.Array,
    fold: int,
    theta: float,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
    """Activation-level FreqFold diagnostic retaining one head of RoPE cache.

    Adjacent frequencies are jointly rotated with KV heads. The leading ``fold``
    components in each group retain the group's original frequency slots, so the
    total RoPE cache remains one source head (``head_dim`` elements).
    """
    kv_heads, head_dim = key.shape[-2:]
    pairs = head_dim // 2
    groups = pairs // fold
    width = kv_heads * fold
    if rotations.shape != (groups, width, width):
        raise ValueError("FreqFold rotations have an incompatible shape")

    q_real, q_imaginary = jnp.split(query.astype(jnp.float32), 2, axis=-1)
    k_real, k_imaginary = jnp.split(key.astype(jnp.float32), 2, axis=-1)
    q_real = q_real.reshape(*q_real.shape[:-1], groups, fold)
    q_imaginary = q_imaginary.reshape(*q_imaginary.shape[:-1], groups, fold)
    k_real = k_real.reshape(*k_real.shape[:-1], groups, fold)
    k_imaginary = k_imaginary.reshape(*k_imaginary.shape[:-1], groups, fold)
    # Each query head initially selects one KV head in the merged representation.
    row_offsets = query_to_kv[:, None] * fold + jnp.arange(fold)[None, :]
    q_rows = jnp.stack(
        [
            rotations[:, row_offsets[:, index], :].transpose(1, 0, 2)
            for index in range(fold)
        ],
        axis=2,
    )
    q_real = jnp.einsum("blhgf,hgfc->blhgc", q_real, q_rows)
    q_imaginary = jnp.einsum("blhgf,hgfc->blhgc", q_imaginary, q_rows)
    k_real = k_real.transpose(0, 1, 3, 2, 4).reshape(
        key.shape[0], key.shape[1], groups, width
    )
    k_imaginary = k_imaginary.transpose(0, 1, 3, 2, 4).reshape(
        key.shape[0], key.shape[1], groups, width
    )
    k_real = jnp.einsum("blgd,gdc->blgc", k_real, rotations)
    k_imaginary = jnp.einsum("blgd,gdc->blgc", k_imaginary, rotations)

    frequency_ids = jnp.arange(pairs, dtype=jnp.float32).reshape(groups, fold)
    inverse = theta ** (-(2.0 * frequency_ids) / head_dim)
    phase = positions.astype(jnp.float32)[..., None, None] * inverse
    cosine, sine = jnp.cos(phase), jnp.sin(phase)
    q_cos = cosine[:, :, None, :, :]
    q_sin = sine[:, :, None, :, :]
    k_cos, k_sin = cosine, sine
    q_leading_real = q_real[..., :fold]
    q_leading_imaginary = q_imaginary[..., :fold]
    k_leading_real = k_real[..., :fold]
    k_leading_imaginary = k_imaginary[..., :fold]
    q_real = q_real.at[..., :fold].set(
        q_leading_real * q_cos - q_leading_imaginary * q_sin
    )
    q_imaginary = q_imaginary.at[..., :fold].set(
        q_leading_imaginary * q_cos + q_leading_real * q_sin
    )
    k_real = k_real.at[..., :fold].set(
        k_leading_real * k_cos - k_leading_imaginary * k_sin
    )
    k_imaginary = k_imaginary.at[..., :fold].set(
        k_leading_imaginary * k_cos + k_leading_real * k_sin
    )

    logits = (
        jnp.einsum("bqhgc,bkgc->bhqk", q_real, k_real)
        + jnp.einsum("bqhgc,bkgc->bhqk", q_imaginary, k_imaginary)
    ) * (head_dim**-0.5)
    causal = positions[:, None, :, None] >= positions[:, None, None, :]
    mask = causal
    if attention_mask is not None:
        supplied = attention_mask.astype(jnp.bool_)
        if supplied.ndim == 2:
            supplied = supplied[:, None, None, :]
        mask = mask & supplied
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    probabilities = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum(
        "bhqk,bkhd->bqhd", probabilities, value[:, :, query_to_kv, :]
    )
