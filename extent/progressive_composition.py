from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def validate_progressive_layer_sets(
    layer_sets: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    """Validate strictly growing nested replacement sets."""
    if not layer_sets:
        raise ValueError("at least one progressive layer set is required")
    normalized = {
        int(count): tuple(int(layer) for layer in layers)
        for count, layers in layer_sets.items()
    }
    counts = sorted(normalized)
    previous: set[int] = set()
    for count in counts:
        layers = normalized[count]
        if count < 1 or len(layers) != count:
            raise ValueError("each layer-set key must equal its layer count")
        if len(set(layers)) != len(layers):
            raise ValueError("progressive layer sets cannot contain duplicates")
        if tuple(sorted(layers)) != layers:
            raise ValueError("progressive layer sets must be sorted")
        if min(layers) < 0 or max(layers) >= 40:
            raise ValueError("replacement layers must be in [0, 40)")
        current = set(layers)
        if previous and not previous < current:
            raise ValueError("progressive layer sets must be strictly nested")
        previous = current
    return normalized


def composition_branch_name(seed: int, replacement_count: int) -> str:
    return f"SEED-{int(seed)}-COMPOSED-{int(replacement_count)}"


def progressive_branch_names(
    seeds: Sequence[int], layer_sets: Mapping[int, Sequence[int]]
) -> tuple[str, ...]:
    sets = validate_progressive_layer_sets(layer_sets)
    names = ["ORIGINAL-CACHED-QWEN"]
    for seed in seeds:
        names.extend(
            composition_branch_name(seed, count) for count in sorted(sets)
        )
    return tuple(names)


def _metric_finite(metric: Mapping) -> bool:
    return bool(metric.get("finite")) and np.isfinite(
        float(metric["mean_nll"])
    )


def _window_values(metric: Mapping, expected: int | None = None) -> np.ndarray:
    values = np.asarray(metric["window_mean_nll"], dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("composition bootstrap requires at least two windows")
    if expected is not None and len(values) != expected:
        raise ValueError("all composition metrics must use paired windows")
    if not np.all(np.isfinite(values)):
        raise ValueError("composition bootstrap contains non-finite window NLL")
    return values


def _bootstrap_stage(
    *,
    count: int,
    layers: tuple[int, ...],
    seeds: tuple[int, ...],
    original_metric: Mapping,
    standalone_metrics: Mapping[str, Mapping],
    composed_metrics: Mapping[str, Mapping],
    samples: int,
    seed: int,
) -> dict:
    original = _window_values(original_metric)
    windows = len(original)
    rng = np.random.default_rng(seed + count)
    indices = rng.integers(0, windows, size=(samples, windows))
    original_draws = np.mean(original[indices], axis=1)
    ratios = []
    interactions = []
    for model_seed in seeds:
        additive = np.zeros(samples, dtype=np.float64)
        for layer in layers:
            layer_metrics = standalone_metrics[str(layer)]
            standalone_original = _window_values(
                layer_metrics["original"], windows
            )
            standalone_branch = _window_values(
                layer_metrics["seeds"][str(model_seed)], windows
            )
            additive += np.mean(
                (standalone_branch - standalone_original)[indices], axis=1
            )
        composed = _window_values(
            composed_metrics[
                composition_branch_name(model_seed, count)
            ],
            windows,
        )
        composed_excess = np.mean(composed[indices], axis=1) - original_draws
        ratios.append(
            np.where(additive > 0, composed_excess / additive, np.nan)
        )
        interactions.append(composed_excess - additive)
    ratio_draws = np.mean(np.stack(ratios, axis=1), axis=1)
    interaction_draws = np.mean(np.stack(interactions, axis=1), axis=1)
    valid = np.isfinite(ratio_draws)
    valid_ratios = ratio_draws[valid]
    ratio_interval = (
        [float(value) for value in np.percentile(valid_ratios, [2.5, 97.5])]
        if len(valid_ratios)
        else None
    )
    return {
        "method": "paired_window_percentile_bootstrap",
        "samples": samples,
        "seed": seed + count,
        "evaluation_windows": windows,
        "finite_ratio_samples": int(np.sum(valid)),
        "all_ratio_samples_finite": bool(np.all(valid)),
        "mean_inflation_ratio_95ci": ratio_interval,
        "mean_interaction_nll_95ci": [
            float(value)
            for value in np.percentile(interaction_draws, [2.5, 97.5])
        ],
    }


def analyze_progressive_composition(
    *,
    seeds: Sequence[int],
    layer_sets: Mapping[int, Sequence[int]],
    original_metric: Mapping,
    standalone_metrics: Mapping[str, Mapping],
    composed_metrics: Mapping[str, Mapping],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260830,
    maximum_original_nll_difference: float = 0.01,
    maximum_mean_inflation: float = 1.25,
    maximum_seed_inflation: float = 1.50,
    maximum_bootstrap_upper: float = 1.50,
    required_seed_passes: int = 2,
) -> dict:
    """Measure nonlinear shock as progressively more Mamba layers compose."""
    sets = validate_progressive_layer_sets(layer_sets)
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("progressive composition requires three distinct seeds")
    if bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if min(
        maximum_original_nll_difference,
        maximum_mean_inflation,
        maximum_seed_inflation,
        maximum_bootstrap_upper,
    ) <= 0:
        raise ValueError("composition thresholds must be positive")
    if not 1 <= required_seed_passes <= len(seeds):
        raise ValueError("required seed passes must fit the seed count")

    required_layers = {layer for layers in sets.values() for layer in layers}
    if required_layers != {int(layer) for layer in standalone_metrics}:
        raise ValueError("standalone metrics do not match the replacement atlas")
    original_nll = float(original_metric["mean_nll"])
    baseline_differences = {
        str(layer): abs(
            float(standalone_metrics[str(layer)]["original"]["mean_nll"])
            - original_nll
        )
        for layer in sorted(required_layers)
    }
    baseline_passed = bool(
        _metric_finite(original_metric)
        and all(
            difference <= maximum_original_nll_difference
            for difference in baseline_differences.values()
        )
    )

    stages = []
    for count in sorted(sets):
        layers = sets[count]
        records = []
        for model_seed in seeds:
            additive_excess = 0.0
            standalone_finite = True
            for layer in layers:
                metrics = standalone_metrics[str(layer)]
                standalone = metrics["seeds"][str(model_seed)]
                baseline = metrics["original"]
                additive_excess += float(standalone["mean_nll"]) - float(
                    baseline["mean_nll"]
                )
                standalone_finite = (
                    standalone_finite
                    and _metric_finite(standalone)
                    and _metric_finite(baseline)
                )
            branch = composition_branch_name(model_seed, count)
            composed = composed_metrics[branch]
            composed_excess = float(composed["mean_nll"]) - original_nll
            interaction = composed_excess - additive_excess
            inflation = (
                composed_excess / additive_excess
                if additive_excess > 0
                else None
            )
            finite = bool(
                standalone_finite
                and _metric_finite(composed)
                and np.isfinite(
                    [additive_excess, composed_excess, interaction]
                ).all()
            )
            records.append(
                {
                    "seed": model_seed,
                    "additive_expected_excess_nll": additive_excess,
                    "observed_composed_excess_nll": composed_excess,
                    "interaction_nll": interaction,
                    "composition_inflation_ratio": inflation,
                    "additive_excess_positive": additive_excess > 0,
                    "seed_inflation_within_limit": bool(
                        inflation is not None
                        and inflation <= maximum_seed_inflation
                    ),
                    "finite": finite,
                }
            )
        valid_ratios = [
            record["composition_inflation_ratio"]
            for record in records
            if record["composition_inflation_ratio"] is not None
        ]
        bootstrap = _bootstrap_stage(
            count=count,
            layers=layers,
            seeds=seeds,
            original_metric=original_metric,
            standalone_metrics=standalone_metrics,
            composed_metrics=composed_metrics,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        mean_inflation = (
            float(np.mean(np.asarray(valid_ratios, dtype=np.float64)))
            if valid_ratios
            else None
        )
        seed_passes = sum(
            record["seed_inflation_within_limit"] for record in records
        )
        interval = bootstrap["mean_inflation_ratio_95ci"]
        stage_passed = bool(
            baseline_passed
            and all(record["finite"] for record in records)
            and all(record["additive_excess_positive"] for record in records)
            and len(valid_ratios) == len(records)
            and mean_inflation is not None
            and mean_inflation <= maximum_mean_inflation
            and seed_passes >= required_seed_passes
            and bootstrap["all_ratio_samples_finite"]
            and interval is not None
            and interval[1] <= maximum_bootstrap_upper
        )
        stages.append(
            {
                "replacement_count": count,
                "layers": list(layers),
                "per_seed": records,
                "additive_expected_excess_nll_mean": float(
                    np.mean(
                        [
                            record["additive_expected_excess_nll"]
                            for record in records
                        ]
                    )
                ),
                "observed_composed_excess_nll_mean": float(
                    np.mean(
                        [
                            record["observed_composed_excess_nll"]
                            for record in records
                        ]
                    )
                ),
                "interaction_nll_mean": float(
                    np.mean([record["interaction_nll"] for record in records])
                ),
                "composition_inflation_ratio_mean": mean_inflation,
                "composition_inflation_ratio_std": (
                    float(np.std(np.asarray(valid_ratios, dtype=np.float64)))
                    if valid_ratios
                    else None
                ),
                "seed_inflation_passes": seed_passes,
                "required_seed_inflation_passes": required_seed_passes,
                "bootstrap": bootstrap,
                "scientific_gate_passed": stage_passed,
            }
        )

    all_finite = bool(
        _metric_finite(original_metric)
        and all(
            record["finite"]
            for stage in stages
            for record in stage["per_seed"]
        )
    )
    primary = stages[-1]
    return {
        "original_mean_nll": original_nll,
        "baseline_reproduction": {
            "absolute_mean_nll_difference_by_layer": baseline_differences,
            "maximum_absolute_mean_nll_difference": max(
                baseline_differences.values()
            ),
            "allowed_absolute_mean_nll_difference": (
                maximum_original_nll_difference
            ),
            "passed": baseline_passed,
        },
        "maximum_allowed_mean_inflation": maximum_mean_inflation,
        "maximum_allowed_seed_inflation": maximum_seed_inflation,
        "maximum_allowed_bootstrap_upper": maximum_bootstrap_upper,
        "stages": stages,
        "primary_replacement_count": primary["replacement_count"],
        "all_finite": all_finite,
        "scientific_gate_passed": bool(
            all_finite and primary["scientific_gate_passed"]
        ),
    }
