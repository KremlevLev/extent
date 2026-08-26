from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from extent.progressive_composition import analyze_progressive_composition


def boundary_branch_name(seed: int, label: str) -> str:
    return f"SEED-{int(seed)}-{label}"


def _windows(metric: Mapping, expected: int | None = None) -> np.ndarray:
    values = np.asarray(metric["window_mean_nll"], dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("boundary analysis requires at least two paired windows")
    if expected is not None and len(values) != expected:
        raise ValueError("all boundary branches must use paired windows")
    if not np.all(np.isfinite(values)):
        raise ValueError("boundary analysis contains non-finite window NLL")
    return values


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _boundary_stage(
    *,
    count: int,
    seeds: tuple[int, ...],
    original_metric: Mapping,
    composed_metrics: Mapping[str, Mapping],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    original = _windows(original_metric)
    windows = len(original)
    records = []
    for model_seed in seeds:
        layer0 = composed_metrics[
            boundary_branch_name(model_seed, "LAYER0-ONLY")
        ]
        internal = composed_metrics[
            boundary_branch_name(model_seed, f"INTERNAL-{count - 1}")
        ]
        full = composed_metrics[
            boundary_branch_name(model_seed, f"COMPOSED-{count}")
        ]
        layer0_excess = float(layer0["mean_nll"]) - float(
            original_metric["mean_nll"]
        )
        internal_excess = float(internal["mean_nll"]) - float(
            original_metric["mean_nll"]
        )
        full_excess = float(full["mean_nll"]) - float(
            original_metric["mean_nll"]
        )
        additive = layer0_excess + internal_excess
        interaction = full_excess - additive
        records.append(
            {
                "seed": model_seed,
                "layer0_only_excess_nll": layer0_excess,
                "internal_only_excess_nll": internal_excess,
                "full_composition_excess_nll": full_excess,
                "internal_minus_layer0_excess_nll": (
                    internal_excess - layer0_excess
                ),
                "boundary_interaction_nll": interaction,
                "boundary_inflation_ratio": (
                    full_excess / additive if additive > 0 else None
                ),
                "internal_smaller_than_layer0": internal_excess < layer0_excess,
                "subadditive_boundary_interaction": interaction < 0,
            }
        )

    rng = np.random.default_rng(bootstrap_seed + count)
    indices = rng.integers(0, windows, size=(bootstrap_samples, windows))
    dominance_draws = []
    interaction_draws = []
    inflation_draws = []
    for model_seed in seeds:
        layer0 = _windows(
            composed_metrics[
                boundary_branch_name(model_seed, "LAYER0-ONLY")
            ],
            windows,
        )
        internal = _windows(
            composed_metrics[
                boundary_branch_name(model_seed, f"INTERNAL-{count - 1}")
            ],
            windows,
        )
        full = _windows(
            composed_metrics[
                boundary_branch_name(model_seed, f"COMPOSED-{count}")
            ],
            windows,
        )
        original_draw = np.mean(original[indices], axis=1)
        layer0_excess = np.mean(layer0[indices], axis=1) - original_draw
        internal_excess = np.mean(internal[indices], axis=1) - original_draw
        full_excess = np.mean(full[indices], axis=1) - original_draw
        additive = layer0_excess + internal_excess
        dominance_draws.append(internal_excess - layer0_excess)
        interaction_draws.append(full_excess - additive)
        inflation_draws.append(
            np.where(additive > 0, full_excess / additive, np.nan)
        )
    dominance = np.mean(np.stack(dominance_draws, axis=1), axis=1)
    interaction = np.mean(np.stack(interaction_draws, axis=1), axis=1)
    inflation = np.mean(np.stack(inflation_draws, axis=1), axis=1)
    valid_inflation = inflation[np.isfinite(inflation)]
    dominance_ci = _interval(dominance)
    interaction_ci = _interval(interaction)
    inflation_ci = _interval(valid_inflation) if len(valid_inflation) else None
    dominance_wins = sum(
        record["internal_smaller_than_layer0"] for record in records
    )
    subadditive_wins = sum(
        record["subadditive_boundary_interaction"] for record in records
    )
    dominance_gate = bool(dominance_wins >= 2 and dominance_ci[1] < 0)
    subadditive_gate = bool(
        subadditive_wins >= 2 and interaction_ci[1] < 0
    )
    return {
        "replacement_count": count,
        "internal_replacement_count": count - 1,
        "per_seed": records,
        "layer0_only_excess_nll_mean": float(
            np.mean([record["layer0_only_excess_nll"] for record in records])
        ),
        "internal_only_excess_nll_mean": float(
            np.mean([record["internal_only_excess_nll"] for record in records])
        ),
        "full_composition_excess_nll_mean": float(
            np.mean(
                [record["full_composition_excess_nll"] for record in records]
            )
        ),
        "internal_minus_layer0_excess_nll_mean": float(
            np.mean(
                [
                    record["internal_minus_layer0_excess_nll"]
                    for record in records
                ]
            )
        ),
        "boundary_interaction_nll_mean": float(
            np.mean([record["boundary_interaction_nll"] for record in records])
        ),
        "boundary_inflation_ratio_mean": (
            float(
                np.mean(
                    [
                        record["boundary_inflation_ratio"]
                        for record in records
                        if record["boundary_inflation_ratio"] is not None
                    ]
                )
            )
            if all(
                record["boundary_inflation_ratio"] is not None
                for record in records
            )
            else None
        ),
        "internal_smaller_than_layer0_wins": dominance_wins,
        "subadditive_boundary_interaction_wins": subadditive_wins,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed + count,
            "internal_minus_layer0_95ci": dominance_ci,
            "boundary_interaction_nll_95ci": interaction_ci,
            "boundary_inflation_ratio_95ci": inflation_ci,
            "all_inflation_samples_finite": bool(
                len(valid_inflation) == bootstrap_samples
            ),
        },
        "layer0_dominance_gate_passed": dominance_gate,
        "boundary_subadditivity_gate_passed": subadditive_gate,
        "mechanism_gate_passed": bool(dominance_gate and subadditive_gate),
    }


def _incremental_stage(
    *,
    seeds: tuple[int, ...],
    lower_count: int,
    upper_count: int,
    layer_sets: Mapping[int, Sequence[int]],
    original_metric: Mapping,
    standalone_metrics: Mapping[str, Mapping],
    composed_metrics: Mapping[str, Mapping],
    bootstrap_samples: int,
    bootstrap_seed: int,
    maximum_mean_inflation: float,
    maximum_seed_inflation: float,
    maximum_bootstrap_upper: float,
) -> dict:
    lower = tuple(layer_sets[lower_count])
    upper = tuple(layer_sets[upper_count])
    added_layers = tuple(layer for layer in upper if layer not in set(lower))
    original = _windows(original_metric)
    windows = len(original)
    records = []
    for model_seed in seeds:
        expected = sum(
            float(standalone_metrics[str(layer)]["seeds"][str(model_seed)]["mean_nll"])
            - float(standalone_metrics[str(layer)]["original"]["mean_nll"])
            for layer in added_layers
        )
        lower_nll = float(
            composed_metrics[
                boundary_branch_name(model_seed, f"COMPOSED-{lower_count}")
            ]["mean_nll"]
        )
        upper_nll = float(
            composed_metrics[
                boundary_branch_name(model_seed, f"COMPOSED-{upper_count}")
            ]["mean_nll"]
        )
        observed = upper_nll - lower_nll
        inflation = observed / expected if expected > 0 else None
        records.append(
            {
                "seed": model_seed,
                "expected_added_excess_nll": expected,
                "observed_added_excess_nll": observed,
                "incremental_interaction_nll": observed - expected,
                "incremental_inflation_ratio": inflation,
                "expected_added_excess_positive": expected > 0,
                "seed_inflation_within_limit": bool(
                    inflation is not None
                    and inflation <= maximum_seed_inflation
                ),
            }
        )

    rng = np.random.default_rng(bootstrap_seed + upper_count)
    indices = rng.integers(0, windows, size=(bootstrap_samples, windows))
    ratio_draws = []
    interaction_draws = []
    for model_seed in seeds:
        expected = np.zeros(bootstrap_samples, dtype=np.float64)
        for layer in added_layers:
            baseline = _windows(
                standalone_metrics[str(layer)]["original"], windows
            )
            branch = _windows(
                standalone_metrics[str(layer)]["seeds"][str(model_seed)],
                windows,
            )
            expected += np.mean((branch - baseline)[indices], axis=1)
        lower_values = _windows(
            composed_metrics[
                boundary_branch_name(model_seed, f"COMPOSED-{lower_count}")
            ],
            windows,
        )
        upper_values = _windows(
            composed_metrics[
                boundary_branch_name(model_seed, f"COMPOSED-{upper_count}")
            ],
            windows,
        )
        observed = np.mean(
            (upper_values - lower_values)[indices], axis=1
        )
        ratio_draws.append(np.where(expected > 0, observed / expected, np.nan))
        interaction_draws.append(observed - expected)
    ratio = np.mean(np.stack(ratio_draws, axis=1), axis=1)
    interaction = np.mean(np.stack(interaction_draws, axis=1), axis=1)
    valid = ratio[np.isfinite(ratio)]
    ratio_ci = _interval(valid) if len(valid) else None
    valid_ratios = [
        record["incremental_inflation_ratio"]
        for record in records
        if record["incremental_inflation_ratio"] is not None
    ]
    mean_ratio = (
        float(np.mean(valid_ratios)) if len(valid_ratios) == len(records) else None
    )
    seed_passes = sum(record["seed_inflation_within_limit"] for record in records)
    gate = bool(
        all(record["expected_added_excess_positive"] for record in records)
        and mean_ratio is not None
        and mean_ratio <= maximum_mean_inflation
        and seed_passes >= 2
        and len(valid) == bootstrap_samples
        and ratio_ci is not None
        and ratio_ci[1] <= maximum_bootstrap_upper
    )
    return {
        "lower_replacement_count": lower_count,
        "upper_replacement_count": upper_count,
        "added_layers": list(added_layers),
        "per_seed": records,
        "expected_added_excess_nll_mean": float(
            np.mean([record["expected_added_excess_nll"] for record in records])
        ),
        "observed_added_excess_nll_mean": float(
            np.mean([record["observed_added_excess_nll"] for record in records])
        ),
        "incremental_interaction_nll_mean": float(
            np.mean([record["incremental_interaction_nll"] for record in records])
        ),
        "incremental_inflation_ratio_mean": mean_ratio,
        "seed_inflation_passes": seed_passes,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed + upper_count,
            "incremental_inflation_ratio_95ci": ratio_ci,
            "incremental_interaction_nll_95ci": _interval(interaction),
            "all_inflation_samples_finite": bool(len(valid) == bootstrap_samples),
        },
        "scientific_gate_passed": gate,
    }


def analyze_boundary_scaling(
    *,
    seeds: Sequence[int],
    layer_sets: Mapping[int, Sequence[int]],
    original_metric: Mapping,
    standalone_metrics: Mapping[str, Mapping],
    composed_metrics: Mapping[str, Mapping],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260831,
    maximum_original_nll_difference: float = 0.01,
    maximum_mean_inflation: float = 1.25,
    maximum_seed_inflation: float = 1.50,
    maximum_bootstrap_upper: float = 1.50,
) -> dict:
    counts = sorted(int(count) for count in layer_sets)
    if len(counts) != 2:
        raise ValueError("boundary scaling requires exactly two nested full sets")
    seeds = tuple(int(seed) for seed in seeds)
    progressive = analyze_progressive_composition(
        seeds=seeds,
        layer_sets=layer_sets,
        original_metric=original_metric,
        standalone_metrics=standalone_metrics,
        composed_metrics=composed_metrics,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        maximum_original_nll_difference=maximum_original_nll_difference,
        maximum_mean_inflation=maximum_mean_inflation,
        maximum_seed_inflation=maximum_seed_inflation,
        maximum_bootstrap_upper=maximum_bootstrap_upper,
    )
    boundary = [
        _boundary_stage(
            count=count,
            seeds=seeds,
            original_metric=original_metric,
            composed_metrics=composed_metrics,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100,
        )
        for count in counts
    ]
    incremental = _incremental_stage(
        seeds=seeds,
        lower_count=counts[0],
        upper_count=counts[1],
        layer_sets=layer_sets,
        original_metric=original_metric,
        standalone_metrics=standalone_metrics,
        composed_metrics=composed_metrics,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 200,
        maximum_mean_inflation=maximum_mean_inflation,
        maximum_seed_inflation=maximum_seed_inflation,
        maximum_bootstrap_upper=maximum_bootstrap_upper,
    )
    primary_boundary = boundary[-1]
    progressive["boundary_analysis"] = {
        "stages": boundary,
        "primary_replacement_count": counts[-1],
        "scientific_gate_passed": primary_boundary["mechanism_gate_passed"],
    }
    progressive["incremental_scaling"] = incremental
    progressive["composition_scaling_gate_passed"] = bool(
        progressive["scientific_gate_passed"]
        and incremental["scientific_gate_passed"]
    )
    progressive["boundary_mechanism_gate_passed"] = primary_boundary[
        "mechanism_gate_passed"
    ]
    progressive["scientific_gate_passed"] = progressive[
        "composition_scaling_gate_passed"
    ]
    return progressive
