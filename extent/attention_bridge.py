"""Attention-to-Mamba-3 bridge primitives for controlled transplantation.

This module deliberately separates three claims that are easy to conflate:

* ``HedgehogBridge`` is an Apple-style learned linear-attention intermediary.
* ``mamba3_orientation_matrix`` is a MOHAWK-inspired, head-averaged proxy for
  the input-dependent Mamba-3 MIMO mixing matrix.
* ``build_bridge_mamba3_initialization`` folds the *linear part* of a learned
  bridge into the canonical Mamba-3 projections.  It is an initialization, not
  exact functional equivalence, because canonical Mamba-3 has BC RMSNorm,
  MIMO gates, and exponential-trapezoidal dynamics rather than Hedgehog's
  feature softmax and explicit normalization state.

Keeping those boundaries explicit lets the experiments measure which part of
the recipe helps without silently changing the production mixer contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable
from typing import Mapping, NamedTuple

from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
import optax

from extent.config import Mamba3Config
from extent.layers.common import dtype_from_name
from extent.layers.mamba3 import heavy_tail_activation
from extent.mamba3_transplant import mamba3_projection_slices
from extent.optimizer import cast_grads_bf16, gradient_health
from extent.qwen3_teacher import (
    Qwen3TeacherConfig,
    apply_qwen3_rope,
)
from extent.weight_mapping import fit_matrix


APPLE_ATTENTION_TO_MAMBA_ARXIV = "2604.14191"
MOHAWK_ARXIV = "2408.10189"


@dataclass(frozen=True)
class AttentionBridgeConfig:
    """Configuration for the learned positive Q/K feature map."""

    feature_dim: int = 64
    epsilon: float = 1e-6
    rope_theta: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.feature_dim < 2 or self.feature_dim % 2:
            raise ValueError("bridge feature_dim must be a positive even number")
        if self.epsilon <= 0:
            raise ValueError("bridge epsilon must be positive")
        if self.rope_theta <= 0:
            raise ValueError("bridge rope_theta must be positive")


class QwenAttentionComponents(NamedTuple):
    query: jax.Array
    key: jax.Array
    value: jax.Array
    matrix: jax.Array
    output: jax.Array


class BridgeOutputs(NamedTuple):
    matrix: jax.Array
    output: jax.Array


@dataclass(frozen=True)
class BridgeInitializationReport:
    variant: str
    description: str
    feature_dim: int
    copied_output_projection: bool
    recurrence_rule: str
    exact_functional_equivalence: bool
    source_arxiv: str


def _rms_norm(x: jax.Array, scale: jax.Array, epsilon: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(variance + epsilon) * scale.astype(jnp.float32))


def qwen3_attention_components(
    params: Mapping,
    inputs: jax.Array,
    positions: jax.Array,
    config: Qwen3TeacherConfig,
    attention_mask: jax.Array | None = None,
) -> QwenAttentionComponents:
    """Project one frozen Qwen3 attention layer and expose its mixing matrix."""
    batch, length, _ = inputs.shape
    dtype = dtype_from_name(config.compute_dtype)
    query = jnp.einsum(
        "bld,df->blf", inputs, params["q_proj"]["kernel"]
    ).astype(dtype).reshape(batch, length, config.num_attention_heads, config.head_dim)
    key = jnp.einsum(
        "bld,df->blf", inputs, params["k_proj"]["kernel"]
    ).astype(dtype).reshape(batch, length, config.num_key_value_heads, config.head_dim)
    value = jnp.einsum(
        "bld,df->blf", inputs, params["v_proj"]["kernel"]
    ).astype(dtype).reshape(batch, length, config.num_key_value_heads, config.head_dim)
    query = _rms_norm(
        query, params["q_norm"]["scale"], config.rms_norm_eps
    ).astype(dtype)
    key = _rms_norm(
        key, params["k_norm"]["scale"], config.rms_norm_eps
    ).astype(dtype)
    groups = config.num_attention_heads // config.num_key_value_heads
    key = jnp.repeat(key, groups, axis=2)
    value = jnp.repeat(value, groups, axis=2)
    rotated_query = apply_qwen3_rope(query, positions, config.rope_theta)
    rotated_key = apply_qwen3_rope(key, positions, config.rope_theta)
    logits = jnp.einsum("bqhd,bkhd->bhqk", rotated_query, rotated_key) * (
        config.head_dim**-0.5
    )
    causal = positions[:, None, :, None] >= positions[:, None, None, :]
    mask = causal
    if attention_mask is not None:
        supplied = attention_mask.astype(jnp.bool_)
        if supplied.ndim == 2:
            supplied = supplied[:, None, None, :]
        mask = mask & supplied
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    matrix = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(dtype)
    attended = jnp.einsum("bhqk,bkhd->bqhd", matrix, value)
    attended = attended.reshape(batch, length, config.hidden_size)
    output = jnp.einsum(
        "bld,df->blf", attended, params["o_proj"]["kernel"]
    ).astype(dtype)
    return QwenAttentionComponents(query, key, value, matrix, output)


def initialize_bridge_params(
    key: jax.Array,
    source_head_dim: int,
    config: AttentionBridgeConfig,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> FrozenDict:
    """Initialize independent Q/K Hedgehog maps with paired +/- features."""
    q_key, k_key = jax.random.split(key)
    half = config.feature_dim // 2
    scale = 1.0 / math.sqrt(source_head_dim)
    return freeze(
        {
            "query": {
                "kernel": jax.random.normal(
                    q_key, (source_head_dim, half), dtype=jnp.float32
                ).astype(dtype)
                * scale,
                "bias": jnp.zeros((half,), dtype=dtype),
            },
            "key": {
                "kernel": jax.random.normal(
                    k_key, (source_head_dim, half), dtype=jnp.float32
                ).astype(dtype)
                * scale,
                "bias": jnp.zeros((half,), dtype=dtype),
            },
        }
    )


def hedgehog_features(x: jax.Array, params: Mapping) -> jax.Array:
    """Apple's stable Hedgehog map: Dense, concatenate +/- and feature softmax."""
    projected = jnp.einsum(
        "blhd,df->blhf", x.astype(jnp.float32), params["kernel"].astype(jnp.float32)
    ) + params["bias"].astype(jnp.float32)
    paired = jnp.concatenate((projected, -projected), axis=-1)
    return jax.nn.softmax(paired, axis=-1)


def _causal_mask(
    positions: jax.Array, attention_mask: jax.Array | None
) -> jax.Array:
    mask = positions[:, :, None] >= positions[:, None, :]
    if attention_mask is not None:
        supplied = attention_mask.astype(jnp.bool_)
        mask = mask & supplied[:, None, :]
    return mask


def explicit_bridge_matrix(
    query_features: jax.Array,
    key_features: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array | None,
    epsilon: float,
) -> jax.Array:
    scores = jnp.einsum("bqhf,bkhf->bhqk", query_features, key_features)
    mask = _causal_mask(positions, attention_mask)[:, None]
    scores = jnp.where(mask, scores, 0.0)
    denominator = jnp.sum(scores, axis=-1, keepdims=True)
    safe_denominator = jnp.where(
        jnp.abs(denominator) >= epsilon,
        denominator,
        jnp.where(denominator >= 0, epsilon, -epsilon),
    )
    return scores / safe_denominator


def recurrent_bridge_attention(
    query_features: jax.Array,
    key_features: jax.Array,
    values: jax.Array,
    attention_mask: jax.Array | None,
    epsilon: float,
) -> jax.Array:
    """Linear-time normalized causal attention, exactly matching the explicit form."""
    batch, _, heads, feature_dim = query_features.shape
    value_dim = values.shape[-1]
    initial = (
        jnp.zeros((batch, heads, feature_dim, value_dim), jnp.float32),
        jnp.zeros((batch, heads, feature_dim), jnp.float32),
    )
    if attention_mask is None:
        valid = jnp.ones(query_features.shape[:2], dtype=jnp.bool_)
    else:
        valid = attention_mask.astype(jnp.bool_)
    time_major = tuple(
        jnp.swapaxes(value, 0, 1)
        for value in (query_features, key_features, values, valid)
    )

    def step(state, values_at_t):
        kv_state, k_state = state
        query_t, key_t, value_t, valid_t = values_at_t
        key_t = jnp.where(valid_t[:, None, None], key_t, 0.0)
        value_t = jnp.where(valid_t[:, None, None], value_t, 0.0)
        kv_state = kv_state + jnp.einsum("bhf,bhd->bhfd", key_t, value_t)
        k_state = k_state + key_t
        numerator = jnp.einsum("bhf,bhfd->bhd", query_t, kv_state)
        denominator = jnp.einsum("bhf,bhf->bh", query_t, k_state)
        safe_denominator = jnp.where(
            jnp.abs(denominator) >= epsilon,
            denominator,
            jnp.where(denominator >= 0, epsilon, -epsilon),
        )
        output = numerator / safe_denominator[..., None]
        output = jnp.where(valid_t[:, None, None], output, 0.0)
        return (kv_state, k_state), output

    _, output = jax.lax.scan(step, initial, time_major)
    return jnp.swapaxes(output, 0, 1)


def apply_attention_bridge(
    bridge_params: Mapping,
    teacher: QwenAttentionComponents,
    positions: jax.Array,
    attention_mask: jax.Array | None,
    output_kernel: jax.Array,
    config: AttentionBridgeConfig,
) -> BridgeOutputs:
    """Apply the learned bridge with a linear-time value recurrence."""
    query_features = hedgehog_features(teacher.query, bridge_params["query"])
    key_features = hedgehog_features(teacher.key, bridge_params["key"])
    query_features = apply_qwen3_rope(
        query_features, positions, config.rope_theta
    )
    key_features = apply_qwen3_rope(
        key_features, positions, config.rope_theta
    )
    matrix = explicit_bridge_matrix(
        query_features,
        key_features,
        positions,
        attention_mask,
        config.epsilon,
    )
    attended = recurrent_bridge_attention(
        query_features,
        key_features,
        teacher.value,
        attention_mask,
        config.epsilon,
    )
    batch, length, _, _ = attended.shape
    attended = attended.reshape(batch, length, -1)
    output = jnp.einsum("bld,df->blf", attended, output_kernel)
    return BridgeOutputs(matrix, output)


def cosine_distance(prediction: jax.Array, target: jax.Array) -> jax.Array:
    prediction = prediction.astype(jnp.float32).reshape(-1)
    target = target.astype(jnp.float32).reshape(-1)
    denominator = jnp.maximum(
        jnp.linalg.norm(prediction) * jnp.linalg.norm(target),
        jnp.finfo(jnp.float32).tiny,
    )
    return 1.0 - jnp.vdot(prediction, target).real / denominator


def relative_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    error = prediction.astype(jnp.float32) - target.astype(jnp.float32)
    denominator = jnp.maximum(
        jnp.sum(jnp.square(target.astype(jnp.float32))),
        jnp.finfo(jnp.float32).tiny,
    )
    return jnp.sum(jnp.square(error)) / denominator


def create_bridge_train_step(
    tx: optax.GradientTransformation,
    config: AttentionBridgeConfig,
    *,
    matrix_loss_weight: float,
) -> Callable:
    """Train only the Hedgehog Q/K feature maps against frozen Qwen attention."""
    if matrix_loss_weight < 0:
        raise ValueError("matrix_loss_weight must be non-negative")
    matrix_weight = jnp.asarray(matrix_loss_weight, jnp.float32)

    @jax.jit
    def train_step(
        params,
        opt_state,
        teacher,
        positions,
        attention_mask,
        output_kernel,
    ):
        def loss_fn(candidate):
            bridge = apply_attention_bridge(
                candidate,
                teacher,
                positions,
                attention_mask,
                output_kernel,
                config,
            )
            output_loss = cosine_distance(bridge.output, teacher.output)
            matrix_loss = relative_mse(bridge.matrix, teacher.matrix)
            return output_loss + matrix_weight * matrix_loss, (
                output_loss,
                matrix_loss,
            )

        (loss, (output_loss, matrix_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        health = gradient_health(grads)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {
            "loss": loss,
            "output_cosine_loss": output_loss,
            "matrix_relative_mse": matrix_loss,
            **health,
        }

    return train_step


def _rotate_split_halves(
    value: jax.Array, angle: jax.Array, rotary_pairs: int
) -> jax.Array:
    half = value.shape[-1] // 2
    real, imag = value[..., :half], value[..., half:]
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    if rotary_pairs < half:
        cos = jnp.pad(cos, ((0, 0), (0, 0), (0, 0), (0, half - rotary_pairs)), constant_values=1.0)
        sin = jnp.pad(sin, ((0, 0), (0, 0), (0, 0), (0, half - rotary_pairs)), constant_values=0.0)
    cos = cos[:, :, None]
    sin = sin[:, :, None]
    return jnp.concatenate((real * cos - imag * sin, real * sin + imag * cos), axis=-1)


def _trapezoidal_mask(
    alpha: jax.Array, gamma: jax.Array, beta: jax.Array
) -> jax.Array:
    """Materialize the official Mamba-3 1-semiseparable x 2-band mask."""
    batch, length, heads = alpha.shape
    initial = jnp.zeros((batch, heads, length), jnp.float32)
    indices = jnp.arange(length, dtype=jnp.int32)
    values = tuple(jnp.swapaxes(value, 0, 1) for value in (alpha, gamma, beta))

    def step(previous, values_at_t):
        index, alpha_t, gamma_t, beta_t = values_at_t
        current = alpha_t[..., None] * previous
        previous_index = jnp.maximum(index - 1, 0)
        previous_weight = jnp.where(index > 0, beta_t, 0.0)
        current = current + previous_weight[..., None] * jax.nn.one_hot(
            previous_index, length, dtype=jnp.float32
        )
        current = current + gamma_t[..., None] * jax.nn.one_hot(
            index, length, dtype=jnp.float32
        )
        return current, current

    _, rows = jax.lax.scan(step, initial, (indices, *values))
    return jnp.transpose(rows, (1, 2, 0, 3))


def mamba3_orientation_matrix(
    params: Mapping,
    inputs: jax.Array,
    config: Mamba3Config,
) -> jax.Array:
    """Return a causal, head-averaged orientation proxy for Mamba-3 MIMO.

    The returned rows are normalized distributions.  This intentionally omits
    the value-dependent gate, output projection, and D skip, matching the
    sequence-mixer scope of MOHAWK Stage 1 while accommodating unequal Qwen and
    Mamba head counts.
    """
    batch, length, hidden_size = inputs.shape
    inner = int(hidden_size * config.expand)
    heads = inner // config.head_dim
    rank = config.mimo_rank
    slices = mamba3_projection_slices(hidden_size, config)
    projected = jnp.einsum(
        "bld,df->blf",
        inputs.astype(jnp.float32),
        params["in_proj"]["kernel"].astype(jnp.float32),
    )
    b = projected[..., slices["b"]].reshape(
        batch, length, rank, config.groups, config.d_state
    )
    c = projected[..., slices["c"]].reshape(
        batch, length, rank, config.groups, config.d_state
    )
    if config.groups == 1:
        b = jnp.broadcast_to(b, (batch, length, rank, heads, config.d_state))
        c = jnp.broadcast_to(c, (batch, length, rank, heads, config.d_state))
    elif heads % config.groups == 0:
        repeats = heads // config.groups
        b = jnp.repeat(b, repeats, axis=3)
        c = jnp.repeat(c, repeats, axis=3)
    else:
        raise ValueError("Mamba heads must be divisible by groups")
    b = _rms_norm(b, params["b_norm"]["scale"], 1e-5)
    c = _rms_norm(c, params["c_norm"]["scale"], 1e-5)
    b = b + jnp.transpose(params["b_bias"], (1, 0, 2))[None, None]
    c = c + jnp.transpose(params["c_bias"], (1, 0, 2))[None, None]

    raw_dt = projected[..., slices["dt"]]
    raw_a = projected[..., slices["a"]]
    raw_trap = projected[..., slices["trap"]]
    raw_angle = projected[..., slices["angle"]]
    dt = jax.nn.softplus(raw_dt + params["dt_bias"].astype(jnp.float32))
    decay = -heavy_tail_activation(raw_a)
    decay = jnp.minimum(decay, -config.a_floor)
    alpha = jnp.exp(decay * dt)
    gate = jax.nn.sigmoid(raw_trap)
    gamma = gate * dt
    beta = (1.0 - gate) * dt * alpha
    mask = _trapezoidal_mask(alpha, gamma, beta)

    rotary_pairs = int(config.d_state * config.rope_fraction) // 2
    increments = (
        jnp.pi * raw_angle[:, :, None, :].astype(jnp.float32) * dt[..., None]
    )
    angle = jnp.cumsum(increments, axis=1)
    b = _rotate_split_halves(b, angle, rotary_pairs)
    c = _rotate_split_halves(c, angle, rotary_pairs)
    affinity = jnp.einsum("btrhn,bsrhn->bhtsr", c, b)
    rank_weight = jnp.mean(
        params["mimo_x"].astype(jnp.float32)
        * params["mimo_o"].astype(jnp.float32),
        axis=-1,
    )
    mixed = jnp.sum(
        affinity * rank_weight[None, :, None, None, :],
        axis=-1,
    )
    mixed = mixed * mask
    causal = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
    logits = mixed / math.sqrt(config.d_state)
    logits = jnp.where(causal[None, None], logits, jnp.finfo(jnp.float32).min)
    return jax.nn.softmax(logits, axis=-1).mean(axis=1)


def create_orientation_train_step(
    tx: optax.GradientTransformation,
    config: Mamba3Config,
    *,
    bf16_gradients: bool,
) -> Callable:
    """Optimize the head-averaged Mamba matrix before block-output recovery."""

    @jax.jit
    def train_step(params, opt_state, inputs, target_matrix):
        target = target_matrix.astype(jnp.float32).mean(axis=1)

        def loss_fn(candidate):
            return relative_mse(
                mamba3_orientation_matrix(candidate, inputs, config), target
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        health = gradient_health(grads)
        optimizer_grads = cast_grads_bf16(grads) if bf16_gradients else grads
        updates, opt_state = tx.update(optimizer_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"loss": loss, **health}

    return train_step


def _composed_head_features(
    projection: np.ndarray,
    source_heads: int,
    source_head_dim: int,
    feature_kernel: np.ndarray,
    target_rank: int,
) -> np.ndarray:
    hidden_size = projection.shape[0]
    heads = projection.reshape(hidden_size, source_heads, source_head_dim)
    half = feature_kernel.shape[1]
    composed = np.einsum("hqd,df->hqf", heads, feature_kernel)
    composed = np.concatenate((composed, -composed), axis=-1)
    groups = np.array_split(np.arange(source_heads), target_rank)
    return np.concatenate(
        [composed[:, group].mean(axis=1) for group in groups], axis=-1
    ).astype(np.float32)


def build_bridge_mamba3_initialization(
    base_params: Mapping,
    arrays: Mapping[str, np.ndarray],
    bridge_params: Mapping,
    source: Qwen3TeacherConfig,
    config: Mamba3Config,
    layer_index: int,
) -> tuple[Mapping, BridgeInitializationReport]:
    """Fold a learned Hedgehog pre-feature map into canonical Mamba-3 MIMO."""
    if config.groups != 1:
        raise ValueError("bridge initialization currently requires one B/C group")
    if config.d_state != 2 * int(bridge_params["query"]["kernel"].shape[1]):
        raise ValueError("bridge feature dimension must equal Mamba d_state")
    prefix = f"model.layers.{layer_index}.self_attn"
    q = np.asarray(arrays[f"{prefix}.q_proj.weight"], np.float32).T
    k = np.asarray(arrays[f"{prefix}.k_proj.weight"], np.float32).T
    v = np.asarray(arrays[f"{prefix}.v_proj.weight"], np.float32).T
    o = np.asarray(arrays[f"{prefix}.o_proj.weight"], np.float32).T
    q_feature = np.asarray(bridge_params["query"]["kernel"], np.float32)
    k_feature = np.asarray(bridge_params["key"]["kernel"], np.float32)
    q_bias = np.asarray(bridge_params["query"]["bias"], np.float32)
    k_bias = np.asarray(bridge_params["key"]["bias"], np.float32)
    q_state = _composed_head_features(
        q,
        source.num_attention_heads,
        source.head_dim,
        q_feature,
        config.mimo_rank,
    )
    k_state = _composed_head_features(
        k,
        source.num_key_value_heads,
        source.head_dim,
        k_feature,
        config.mimo_rank,
    )
    was_frozen = isinstance(base_params, FrozenDict)
    mutable = (
        unfreeze(base_params)
        if was_frozen
        else unfreeze(freeze(base_params))
    )
    slices = mamba3_projection_slices(source.hidden_size, config)
    in_kernel = np.asarray(mutable["in_proj"]["kernel"], np.float32).copy()
    inner = int(source.hidden_size * config.expand)
    in_kernel[:, slices["x"]] = fit_matrix(v, (source.hidden_size, inner))
    in_kernel[:, slices["b"]] = fit_matrix(
        k_state, (source.hidden_size, config.mimo_rank * config.d_state)
    )
    in_kernel[:, slices["c"]] = fit_matrix(
        q_state, (source.hidden_size, config.mimo_rank * config.d_state)
    )
    for name in ("dt", "a", "trap", "angle"):
        in_kernel[:, slices[name]] = 0.0
    parameter_dtype = mutable["in_proj"]["kernel"].dtype
    mutable["in_proj"]["kernel"] = jnp.asarray(in_kernel, dtype=parameter_dtype)
    mutable["out_proj"]["kernel"] = jnp.asarray(
        fit_matrix(o, tuple(mutable["out_proj"]["kernel"].shape)),
        dtype=mutable["out_proj"]["kernel"].dtype,
    )
    norm_indices = np.floor(
        np.arange(config.d_state) * source.head_dim / config.d_state
    ).astype(np.int32)
    mutable["b_norm"]["scale"] = jnp.asarray(
        np.asarray(arrays[f"{prefix}.k_norm.weight"], np.float32)[norm_indices],
        dtype=mutable["b_norm"]["scale"].dtype,
    )
    mutable["c_norm"]["scale"] = jnp.asarray(
        np.asarray(arrays[f"{prefix}.q_norm.weight"], np.float32)[norm_indices],
        dtype=mutable["c_norm"]["scale"].dtype,
    )
    heads = inner // config.head_dim
    paired_k_bias = np.concatenate((k_bias, -k_bias))
    paired_q_bias = np.concatenate((q_bias, -q_bias))
    mutable["b_bias"] = jnp.asarray(
        np.broadcast_to(
            paired_k_bias, (heads, config.mimo_rank, config.d_state)
        ).copy(),
        dtype=mutable["b_bias"].dtype,
    )
    mutable["c_bias"] = jnp.asarray(
        np.broadcast_to(
            paired_q_bias, (heads, config.mimo_rank, config.d_state)
        ).copy(),
        dtype=mutable["c_bias"].dtype,
    )
    initial_dt = max(config.dt_min, config.dt_init_floor)
    inverse_softplus = initial_dt + math.log(-math.expm1(-initial_dt))
    mutable["dt_bias"] = jnp.full_like(mutable["dt_bias"], inverse_softplus)
    mutable["D"] = jnp.zeros_like(mutable["D"])
    result = freeze(mutable) if was_frozen else mutable
    if not all(
        np.all(np.isfinite(np.asarray(value))) for value in jax.tree.leaves(result)
    ):
        raise FloatingPointError("bridge initialization produced non-finite parameters")
    report = BridgeInitializationReport(
        variant="INIT-J-apple-linear-bridge",
        description=(
            "Learned Hedgehog Q/K pre-features are composed with Qwen Q/K, "
            "partitioned across MIMO rank, and copied with V/O into canonical Mamba-3."
        ),
        feature_dim=config.d_state,
        copied_output_projection=True,
        recurrence_rule=(
            "dt initialized at dt_min; dt/a/trap/angle input projections zeroed; "
            "D skip zeroed; canonical Mamba-3 gate and BC RMSNorm retained"
        ),
        exact_functional_equivalence=False,
        source_arxiv=APPLE_ATTENTION_TO_MAMBA_ARXIV,
    )
    return result, report
