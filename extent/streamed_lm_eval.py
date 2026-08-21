from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def full_model_shard_last_use(
    weight_map: Mapping[str, str], num_layers: int
) -> dict[str, int]:
    """Last streamed stage that needs each checkpoint shard.

    Decoder layers use stages 0..num_layers-1 and final norm/lm_head use
    stage num_layers. Embeddings are stage -1 unless they share a later shard.
    """
    result: dict[str, int] = {}
    for name, shard in weight_map.items():
        stage = None
        if name == "model.embed_tokens.weight":
            stage = -1
        elif name in {"model.norm.weight", "lm_head.weight"}:
            stage = num_layers
        elif name.startswith("model.layers."):
            try:
                stage = int(name.split(".")[2])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid Qwen layer tensor name: {name}") from exc
        if stage is not None:
            result[shard] = max(result.get(shard, -1), stage)
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    missing = required.difference(weight_map)
    if missing:
        raise KeyError(f"checkpoint index is missing final-model tensors: {sorted(missing)}")
    return result


def hidden_relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    denominator = max(
        float(np.linalg.norm(reference.reshape(-1))),
        float(np.finfo(np.float32).tiny),
    )
    return float(np.linalg.norm((candidate - reference).reshape(-1)) / denominator)


def next_token_statistics(
    logits: jax.Array,
    token_ids: jax.Array,
    valid_windows: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return summed NLL, token count, and top-1 hits for within-window labels."""
    if logits.ndim != 3 or token_ids.ndim != 2:
        raise ValueError("logits and token IDs must have batch/sequence axes")
    if logits.shape[:2] != token_ids.shape or token_ids.shape[1] < 2:
        raise ValueError("logit/token shapes must agree and sequence length must exceed one")
    prediction_logits = logits[:, :-1].astype(jnp.float32)
    labels = token_ids[:, 1:].astype(jnp.int32)
    log_normalizer = jax.nn.logsumexp(prediction_logits, axis=-1)
    selected = jnp.take_along_axis(
        prediction_logits, labels[..., None], axis=-1
    )[..., 0]
    losses = log_normalizer - selected
    correct = jnp.argmax(prediction_logits, axis=-1) == labels
    if valid_windows is None:
        valid_windows = jnp.ones((token_ids.shape[0],), dtype=jnp.bool_)
    mask = jnp.broadcast_to(valid_windows[:, None], losses.shape)
    return (
        jnp.sum(jnp.where(mask, losses, 0.0), dtype=jnp.float32),
        jnp.sum(mask, dtype=jnp.int32),
        jnp.sum(jnp.where(mask, correct, False), dtype=jnp.int32),
    )


def create_lm_metrics_runner(norm_module, *, data_parallel_devices=None) -> Callable:
    """Compile final RMSNorm + lm_head NLL without returning full logits to host."""

    def apply(norm_params, lm_head_kernel, hidden, token_ids, valid_windows):
        normalized = norm_module.apply({"params": norm_params}, hidden)
        logits = jnp.einsum(
            "bsh,hv->bsv",
            normalized,
            lm_head_kernel,
            preferred_element_type=jnp.bfloat16,
        )
        return next_token_statistics(logits, token_ids, valid_windows)

    if data_parallel_devices:
        return jax.pmap(
            apply,
            in_axes=(None, None, 0, 0, 0),
            devices=data_parallel_devices,
        )
    return jax.jit(apply)


def end_to_end_loss_comparison(
    *,
    original_nll: float,
    calibrated_nll: float,
    mixer_only_nll: float,
    joint_nll: float,
    all_finite: bool,
    required_recovery_fraction: float = 0.10,
) -> dict:
    """Apply the frozen EXP-040 excess-NLL recovery gate."""
    mixer_excess = mixer_only_nll - original_nll
    joint_excess = joint_nll - original_nll
    recovered = (
        (mixer_excess - joint_excess) / mixer_excess
        if mixer_excess > 0
        else None
    )
    passed = bool(
        all_finite
        and mixer_excess > 0
        and joint_nll < mixer_only_nll
        and joint_nll < calibrated_nll
        and recovered is not None
        and recovered >= required_recovery_fraction
    )
    return {
        "original_mean_nll": original_nll,
        "calibrated_mean_nll": calibrated_nll,
        "mixer_only_mean_nll": mixer_only_nll,
        "joint_mean_nll": joint_nll,
        "mixer_only_excess_nll": mixer_excess,
        "joint_excess_nll": joint_excess,
        "joint_recovered_mixer_excess_fraction": recovered,
        "required_recovered_excess_fraction": required_recovery_fraction,
        "scientific_gate_passed": passed,
    }
