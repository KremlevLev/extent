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


def next_token_window_statistics(
    logits: jax.Array,
    token_ids: jax.Array,
    valid_windows: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-window NLL sums, label counts, and top-1 hits."""
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
        jnp.sum(jnp.where(mask, losses, 0.0), axis=1, dtype=jnp.float32),
        jnp.sum(mask, axis=1, dtype=jnp.int32),
        jnp.sum(jnp.where(mask, correct, False), axis=1, dtype=jnp.int32),
    )


def next_token_statistics(
    logits: jax.Array,
    token_ids: jax.Array,
    valid_windows: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return summed NLL, token count, and top-1 hits for within-window labels."""
    losses, counts, correct = next_token_window_statistics(
        logits, token_ids, valid_windows
    )
    return jnp.sum(losses), jnp.sum(counts), jnp.sum(correct)


def create_lm_metrics_runner(
    norm_module,
    *,
    data_parallel_devices=None,
    per_window: bool = False,
) -> Callable:
    """Compile final RMSNorm + lm_head NLL without returning full logits to host."""

    def apply(norm_params, lm_head_kernel, hidden, token_ids, valid_windows):
        normalized = norm_module.apply({"params": norm_params}, hidden)
        logits = jnp.einsum(
            "bsh,hv->bsv",
            normalized,
            lm_head_kernel,
            preferred_element_type=jnp.bfloat16,
        )
        statistics = (
            next_token_window_statistics
            if per_window
            else next_token_statistics
        )
        return statistics(logits, token_ids, valid_windows)

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


def aggregate_multiseed_end_to_end(
    *,
    original_nll: float,
    seed_metrics: Mapping[str, Mapping[str, Mapping[str, float | bool]]],
    reference_reproduced: bool,
    reference_required: bool = True,
    required_recovery_fraction: float = 0.10,
    required_wins: int = 2,
) -> dict:
    """Aggregate the frozen three-seed EXP-041 end-to-end gate."""
    if len(seed_metrics) != 3:
        raise ValueError("EXP-041 requires exactly three seed metric groups")
    paired = []
    for seed, metrics in seed_metrics.items():
        calibrated = metrics["calibrated"]
        mixer = metrics["mixer_only"]
        joint = metrics["joint"]
        calibrated_nll = float(calibrated["mean_nll"])
        mixer_nll = float(mixer["mean_nll"])
        joint_nll = float(joint["mean_nll"])
        mixer_excess = mixer_nll - original_nll
        joint_excess = joint_nll - original_nll
        recovery = (
            (mixer_excess - joint_excess) / mixer_excess
            if mixer_excess > 0
            else None
        )
        finite = bool(
            calibrated["finite"] and mixer["finite"] and joint["finite"]
        )
        paired.append(
            {
                "seed": int(seed),
                "calibrated_mean_nll": calibrated_nll,
                "mixer_only_mean_nll": mixer_nll,
                "joint_mean_nll": joint_nll,
                "mixer_only_excess_nll": mixer_excess,
                "joint_excess_nll": joint_excess,
                "joint_recovered_mixer_excess_fraction": recovery,
                "joint_wins_mixer_only": joint_nll < mixer_nll,
                "joint_wins_calibrated": joint_nll < calibrated_nll,
                "finite": finite,
            }
        )
    recoveries = [
        record["joint_recovered_mixer_excess_fraction"] for record in paired
    ]
    calibrated_nlls = np.asarray(
        [record["calibrated_mean_nll"] for record in paired], dtype=np.float64
    )
    mixer_nlls = np.asarray(
        [record["mixer_only_mean_nll"] for record in paired], dtype=np.float64
    )
    joint_nlls = np.asarray(
        [record["joint_mean_nll"] for record in paired], dtype=np.float64
    )
    valid_recoveries = [value for value in recoveries if value is not None]
    all_finite = all(record["finite"] for record in paired)
    all_mixer_excess_positive = len(valid_recoveries) == len(paired)
    mixer_wins = sum(record["joint_wins_mixer_only"] for record in paired)
    calibrated_wins = sum(record["joint_wins_calibrated"] for record in paired)
    mean_recovery = (
        float(np.mean(np.asarray(valid_recoveries, dtype=np.float64)))
        if valid_recoveries
        else None
    )
    passed = bool(
        all_finite
        and all_mixer_excess_positive
        and mixer_wins >= required_wins
        and calibrated_wins >= required_wins
        and mean_recovery is not None
        and mean_recovery >= required_recovery_fraction
        and (reference_reproduced or not reference_required)
    )
    return {
        "original_mean_nll": original_nll,
        "paired_endpoints": paired,
        "calibrated_mean_nll_mean": float(np.mean(calibrated_nlls)),
        "calibrated_mean_nll_std": float(np.std(calibrated_nlls)),
        "mixer_only_mean_nll_mean": float(np.mean(mixer_nlls)),
        "mixer_only_mean_nll_std": float(np.std(mixer_nlls)),
        "joint_mean_nll_mean": float(np.mean(joint_nlls)),
        "joint_mean_nll_std": float(np.std(joint_nlls)),
        "joint_recovered_mixer_excess_fraction_mean": mean_recovery,
        "joint_recovered_mixer_excess_fraction_std": (
            float(np.std(np.asarray(valid_recoveries, dtype=np.float64)))
            if valid_recoveries
            else None
        ),
        "joint_mixer_only_wins": mixer_wins,
        "joint_calibrated_wins": calibrated_wins,
        "required_joint_wins": required_wins,
        "required_recovered_excess_fraction_mean": required_recovery_fraction,
        "all_mixer_only_excess_nll_positive": all_mixer_excess_positive,
        "all_finite": all_finite,
        "reference_exp040_reproduced": reference_reproduced,
        "reference_exp040_required": reference_required,
        "scientific_gate_passed": passed,
    }


def bootstrap_excess_nll_recovery(
    *,
    original_window_nll: list[float],
    seed_window_nll: Mapping[str, Mapping[str, list[float]]],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260823,
    confidence: float = 0.95,
) -> dict:
    """Paired window bootstrap of mean three-seed excess-NLL recovery."""
    original = np.asarray(original_window_nll, dtype=np.float64)
    if original.ndim != 1 or len(original) < 2:
        raise ValueError("bootstrap requires at least two original windows")
    if len(seed_window_nll) != 3:
        raise ValueError("bootstrap requires exactly three seed groups")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    paired = {}
    for seed, metrics in seed_window_nll.items():
        mixer = np.asarray(metrics["mixer_only"], dtype=np.float64)
        joint = np.asarray(metrics["joint"], dtype=np.float64)
        if mixer.shape != original.shape or joint.shape != original.shape:
            raise ValueError("all bootstrap branches must share the window shape")
        paired[str(seed)] = (mixer, joint)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, len(original), size=(bootstrap_samples, len(original))
    )
    original_means = np.mean(original[indices], axis=1)
    seed_recoveries = []
    for mixer, joint in paired.values():
        mixer_means = np.mean(mixer[indices], axis=1)
        joint_means = np.mean(joint[indices], axis=1)
        excess = mixer_means - original_means
        recovery = np.where(
            excess > 0,
            (mixer_means - joint_means) / excess,
            np.nan,
        )
        seed_recoveries.append(recovery)
    mean_recovery = np.nanmean(np.stack(seed_recoveries, axis=1), axis=1)
    valid = mean_recovery[np.isfinite(mean_recovery)]
    if len(valid) != bootstrap_samples:
        raise ValueError("bootstrap produced non-positive mixer excess NLL")
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "paired_window_percentile_bootstrap",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "confidence": confidence,
        "evaluation_windows": len(original),
        "mean_recovery_bootstrap_mean": float(np.mean(valid)),
        "mean_recovery_confidence_interval": [
            float(np.quantile(valid, alpha)),
            float(np.quantile(valid, 1.0 - alpha)),
        ],
        "finite_samples": int(len(valid)),
    }


def aggregate_two_layer_composition(
    *,
    original_nll: float,
    seed_metrics: Mapping[str, Mapping[str, Mapping[str, float | bool]]],
    maximum_mean_inflation: float = 1.25,
    maximum_seed_inflation: float = 1.50,
    required_seed_passes: int = 2,
) -> dict:
    """Aggregate the frozen EXP-044 non-additive composition endpoint."""
    if len(seed_metrics) != 3:
        raise ValueError("EXP-044 requires exactly three seed metric groups")
    if maximum_mean_inflation <= 0 or maximum_seed_inflation <= 0:
        raise ValueError("composition inflation limits must be positive")
    if not 1 <= required_seed_passes <= len(seed_metrics):
        raise ValueError("required seed passes must fit the seed count")

    paired = []
    for seed, metrics in seed_metrics.items():
        layer0 = metrics["layer0_only"]
        layer18 = metrics["layer18_only"]
        both = metrics["layer0_layer18"]
        layer0_nll = float(layer0["mean_nll"])
        layer18_nll = float(layer18["mean_nll"])
        both_nll = float(both["mean_nll"])
        layer0_excess = layer0_nll - original_nll
        layer18_excess = layer18_nll - original_nll
        additive_excess = layer0_excess + layer18_excess
        both_excess = both_nll - original_nll
        interaction = both_excess - additive_excess
        inflation = (
            both_excess / additive_excess if additive_excess > 0 else None
        )
        marginal_layer18 = both_nll - layer0_nll
        marginal_amplification = (
            marginal_layer18 / layer18_excess if layer18_excess > 0 else None
        )
        finite = bool(
            layer0["finite"]
            and layer18["finite"]
            and both["finite"]
            and np.isfinite(
                [layer0_nll, layer18_nll, both_nll, original_nll]
            ).all()
        )
        paired.append(
            {
                "seed": int(seed),
                "layer0_only_mean_nll": layer0_nll,
                "layer18_only_mean_nll": layer18_nll,
                "layer0_layer18_mean_nll": both_nll,
                "layer0_excess_nll": layer0_excess,
                "layer18_excess_nll": layer18_excess,
                "additive_expected_excess_nll": additive_excess,
                "observed_composed_excess_nll": both_excess,
                "interaction_nll": interaction,
                "composition_inflation_ratio": inflation,
                "layer18_marginal_excess_after_layer0": marginal_layer18,
                "layer18_marginal_amplification": marginal_amplification,
                "single_layer_excess_positive": (
                    layer0_excess > 0 and layer18_excess > 0
                ),
                "seed_inflation_within_limit": (
                    inflation is not None
                    and inflation <= maximum_seed_inflation
                ),
                "finite": finite,
            }
        )

    valid_inflations = [
        record["composition_inflation_ratio"]
        for record in paired
        if record["composition_inflation_ratio"] is not None
    ]
    all_finite = all(record["finite"] for record in paired)
    all_single_excess_positive = all(
        record["single_layer_excess_positive"] for record in paired
    )
    seed_passes = sum(
        record["seed_inflation_within_limit"] for record in paired
    )
    mean_inflation = (
        float(np.mean(np.asarray(valid_inflations, dtype=np.float64)))
        if valid_inflations
        else None
    )
    passed = bool(
        all_finite
        and all_single_excess_positive
        and len(valid_inflations) == len(paired)
        and mean_inflation is not None
        and mean_inflation <= maximum_mean_inflation
        and seed_passes >= required_seed_passes
    )
    return {
        "original_mean_nll": float(original_nll),
        "paired_endpoints": paired,
        "composition_inflation_ratio_mean": mean_inflation,
        "composition_inflation_ratio_std": (
            float(np.std(np.asarray(valid_inflations, dtype=np.float64)))
            if valid_inflations
            else None
        ),
        "maximum_allowed_mean_inflation": maximum_mean_inflation,
        "maximum_allowed_seed_inflation": maximum_seed_inflation,
        "seed_inflation_passes": seed_passes,
        "required_seed_inflation_passes": required_seed_passes,
        "all_single_layer_excess_nll_positive": all_single_excess_positive,
        "all_finite": all_finite,
        "scientific_gate_passed": passed,
    }


def bootstrap_two_layer_composition_inflation(
    *,
    original_window_nll: list[float],
    seed_window_nll: Mapping[str, Mapping[str, list[float]]],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260824,
    confidence: float = 0.95,
) -> dict:
    """Paired window bootstrap of mean EXP-044 composition inflation."""
    original = np.asarray(original_window_nll, dtype=np.float64)
    if original.ndim != 1 or len(original) < 2:
        raise ValueError("bootstrap requires at least two original windows")
    if len(seed_window_nll) != 3:
        raise ValueError("bootstrap requires exactly three seed groups")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    paired = {}
    for seed, metrics in seed_window_nll.items():
        arrays = {
            name: np.asarray(metrics[name], dtype=np.float64)
            for name in ("layer0_only", "layer18_only", "layer0_layer18")
        }
        if any(value.shape != original.shape for value in arrays.values()):
            raise ValueError("all bootstrap branches must share the window shape")
        paired[str(seed)] = arrays

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, len(original), size=(bootstrap_samples, len(original))
    )
    original_means = np.mean(original[indices], axis=1)
    seed_inflations = []
    for metrics in paired.values():
        layer0 = np.mean(metrics["layer0_only"][indices], axis=1)
        layer18 = np.mean(metrics["layer18_only"][indices], axis=1)
        both = np.mean(metrics["layer0_layer18"][indices], axis=1)
        additive = (layer0 - original_means) + (layer18 - original_means)
        inflation = np.where(
            additive > 0,
            (both - original_means) / additive,
            np.nan,
        )
        seed_inflations.append(inflation)
    mean_inflation = np.nanmean(np.stack(seed_inflations, axis=1), axis=1)
    valid = mean_inflation[np.isfinite(mean_inflation)]
    if len(valid) != bootstrap_samples:
        raise ValueError("bootstrap produced non-positive additive excess NLL")
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "paired_window_percentile_bootstrap",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "confidence": confidence,
        "evaluation_windows": len(original),
        "mean_inflation_bootstrap_mean": float(np.mean(valid)),
        "mean_inflation_confidence_interval": [
            float(np.quantile(valid, alpha)),
            float(np.quantile(valid, 1.0 - alpha)),
        ],
        "finite_samples": int(len(valid)),
    }
