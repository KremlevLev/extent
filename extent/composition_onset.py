from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from extent.progressive_composition import (
    analyze_progressive_composition,
    composition_branch_name,
    validate_progressive_layer_sets,
)


def single_addition_branch_name(seed: int, layer: int) -> str:
    return f"SEED-{int(seed)}-BASE8-PLUS-L{int(layer)}"


def layout_branch_name(seed: int, layout: str) -> str:
    return f"SEED-{int(seed)}-{layout}-12"


def _windows(metric: Mapping, expected: int | None = None) -> np.ndarray:
    values = np.asarray(metric["window_mean_nll"], dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("onset analysis requires at least two paired windows")
    if expected is not None and len(values) != expected:
        raise ValueError("all onset branches must use paired windows")
    if not np.all(np.isfinite(values)):
        raise ValueError("onset analysis contains non-finite window NLL")
    return values


def _interval(values: np.ndarray, alpha: float = 0.05) -> list[float]:
    return [
        float(value)
        for value in np.percentile(
            values, [100 * alpha / 2, 100 * (1 - alpha / 2)]
        )
    ]


def _bootstrap_effects(
    *,
    seeds: tuple[int, ...],
    expected_pairs: Mapping[int, tuple[np.ndarray, np.ndarray]],
    observed_pairs: Mapping[int, tuple[np.ndarray, np.ndarray]],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    interaction_draws = []
    ratio_draws = []
    for model_seed in seeds:
        expected_branch, expected_baseline = expected_pairs[model_seed]
        observed_branch, observed_baseline = observed_pairs[model_seed]
        expected = np.mean(
            (expected_branch - expected_baseline)[indices], axis=1
        )
        observed = np.mean(
            (observed_branch - observed_baseline)[indices], axis=1
        )
        interaction_draws.append(observed - expected)
        ratio = np.full_like(expected, np.nan)
        np.divide(observed, expected, out=ratio, where=expected > 0)
        ratio_draws.append(ratio)
    return (
        np.mean(np.stack(interaction_draws, axis=1), axis=1),
        np.mean(np.stack(ratio_draws, axis=1), axis=1),
    )


def _incremental_onset_stages(
    *,
    seeds: tuple[int, ...],
    layer_sets: Mapping[int, tuple[int, ...]],
    standalone_metrics: Mapping[str, Mapping],
    composed_metrics: Mapping[str, Mapping],
    bootstrap_samples: int,
    bootstrap_seed: int,
    familywise_alpha: float,
) -> list[dict]:
    counts = sorted(layer_sets)
    base = _windows(
        composed_metrics[composition_branch_name(seeds[0], counts[0])]
    )
    windows = len(base)
    rng = np.random.default_rng(bootstrap_seed + 100)
    indices = rng.integers(0, windows, size=(bootstrap_samples, windows))
    comparisons = len(counts) - 1
    adjusted_alpha = familywise_alpha / comparisons
    stages = []
    for lower_count, upper_count in zip(counts, counts[1:]):
        lower_layers = set(layer_sets[lower_count])
        added_layers = tuple(
            layer for layer in layer_sets[upper_count] if layer not in lower_layers
        )
        records = []
        expected_pairs = {}
        observed_pairs = {}
        for model_seed in seeds:
            expected = sum(
                float(
                    standalone_metrics[str(layer)]["seeds"][str(model_seed)][
                        "mean_nll"
                    ]
                )
                - float(standalone_metrics[str(layer)]["original"]["mean_nll"])
                for layer in added_layers
            )
            lower_metric = composed_metrics[
                composition_branch_name(model_seed, lower_count)
            ]
            upper_metric = composed_metrics[
                composition_branch_name(model_seed, upper_count)
            ]
            observed = float(upper_metric["mean_nll"]) - float(
                lower_metric["mean_nll"]
            )
            interaction = observed - expected
            records.append(
                {
                    "seed": model_seed,
                    "expected_added_excess_nll": expected,
                    "observed_added_excess_nll": observed,
                    "incremental_interaction_nll": interaction,
                    "incremental_inflation_ratio": (
                        observed / expected if expected > 0 else None
                    ),
                    "positive_interaction": interaction > 0,
                }
            )
            expected_branch = np.zeros(windows, dtype=np.float64)
            expected_baseline = np.zeros(windows, dtype=np.float64)
            for layer in added_layers:
                expected_branch += _windows(
                    standalone_metrics[str(layer)]["seeds"][str(model_seed)],
                    windows,
                )
                expected_baseline += _windows(
                    standalone_metrics[str(layer)]["original"], windows
                )
            expected_pairs[model_seed] = (
                expected_branch,
                expected_baseline,
            )
            observed_pairs[model_seed] = (
                _windows(upper_metric, windows),
                _windows(lower_metric, windows),
            )
        interaction_draws, ratio_draws = _bootstrap_effects(
            seeds=seeds,
            expected_pairs=expected_pairs,
            observed_pairs=observed_pairs,
            indices=indices,
        )
        valid_ratios = ratio_draws[np.isfinite(ratio_draws)]
        interaction_ci = _interval(interaction_draws, adjusted_alpha)
        ratio_ci = (
            _interval(valid_ratios, adjusted_alpha)
            if len(valid_ratios) == bootstrap_samples
            else None
        )
        positive_wins = sum(record["positive_interaction"] for record in records)
        detected = bool(
            positive_wins >= 2
            and interaction_ci[0] > 0
        )
        point_ratios = [
            record["incremental_inflation_ratio"]
            for record in records
            if record["incremental_inflation_ratio"] is not None
        ]
        stages.append(
            {
                "lower_replacement_count": lower_count,
                "upper_replacement_count": upper_count,
                "added_layers": list(added_layers),
                "per_seed": records,
                "expected_added_excess_nll_mean": float(
                    np.mean(
                        [record["expected_added_excess_nll"] for record in records]
                    )
                ),
                "observed_added_excess_nll_mean": float(
                    np.mean(
                        [record["observed_added_excess_nll"] for record in records]
                    )
                ),
                "incremental_interaction_nll_mean": float(
                    np.mean(
                        [record["incremental_interaction_nll"] for record in records]
                    )
                ),
                "incremental_inflation_ratio_mean": (
                    float(np.mean(point_ratios))
                    if len(point_ratios) == len(records)
                    else None
                ),
                "positive_interaction_wins": positive_wins,
                "bootstrap": {
                    "samples": bootstrap_samples,
                    "seed": bootstrap_seed + 100,
                    "familywise_alpha": familywise_alpha,
                    "comparisons": comparisons,
                    "adjusted_confidence_level": 1 - adjusted_alpha,
                    "incremental_interaction_nll_adjusted_ci": interaction_ci,
                    "incremental_inflation_ratio_adjusted_ci": ratio_ci,
                },
                "superadditive_onset_detected": detected,
            }
        )
    return stages


def _single_addition_analysis(
    *,
    seeds: tuple[int, ...],
    base_count: int,
    added_layers: tuple[int, ...],
    standalone_metrics: Mapping[str, Mapping],
    composed_metrics: Mapping[str, Mapping],
    bootstrap_samples: int,
    bootstrap_seed: int,
    familywise_alpha: float,
) -> list[dict]:
    base = _windows(
        composed_metrics[composition_branch_name(seeds[0], base_count)]
    )
    windows = len(base)
    rng = np.random.default_rng(bootstrap_seed + 200)
    indices = rng.integers(0, windows, size=(bootstrap_samples, windows))
    adjusted_alpha = familywise_alpha / len(added_layers)
    layers = []
    for layer in added_layers:
        records = []
        expected_pairs = {}
        observed_pairs = {}
        for model_seed in seeds:
            standalone = standalone_metrics[str(layer)]["seeds"][str(model_seed)]
            standalone_original = standalone_metrics[str(layer)]["original"]
            expected = float(standalone["mean_nll"]) - float(
                standalone_original["mean_nll"]
            )
            base_metric = composed_metrics[
                composition_branch_name(model_seed, base_count)
            ]
            conditional = composed_metrics[
                single_addition_branch_name(model_seed, layer)
            ]
            observed = float(conditional["mean_nll"]) - float(
                base_metric["mean_nll"]
            )
            interaction = observed - expected
            records.append(
                {
                    "seed": model_seed,
                    "standalone_expected_excess_nll": expected,
                    "conditional_added_excess_nll": observed,
                    "conditional_interaction_nll": interaction,
                    "conditional_amplification_ratio": (
                        observed / expected if expected > 0 else None
                    ),
                    "positive_interaction": interaction > 0,
                }
            )
            expected_pairs[model_seed] = (
                _windows(standalone, windows),
                _windows(standalone_original, windows),
            )
            observed_pairs[model_seed] = (
                _windows(conditional, windows),
                _windows(base_metric, windows),
            )
        interaction_draws, ratio_draws = _bootstrap_effects(
            seeds=seeds,
            expected_pairs=expected_pairs,
            observed_pairs=observed_pairs,
            indices=indices,
        )
        valid_ratios = ratio_draws[np.isfinite(ratio_draws)]
        interaction_ci = _interval(interaction_draws, adjusted_alpha)
        ratio_ci = (
            _interval(valid_ratios, adjusted_alpha)
            if len(valid_ratios) == bootstrap_samples
            else None
        )
        positive_wins = sum(record["positive_interaction"] for record in records)
        context_sensitive = bool(
            positive_wins >= 2
            and interaction_ci[0] > 0
        )
        point_ratios = [
            record["conditional_amplification_ratio"]
            for record in records
            if record["conditional_amplification_ratio"] is not None
        ]
        layers.append(
            {
                "layer": layer,
                "per_seed": records,
                "standalone_expected_excess_nll_mean": float(
                    np.mean(
                        [
                            record["standalone_expected_excess_nll"]
                            for record in records
                        ]
                    )
                ),
                "conditional_added_excess_nll_mean": float(
                    np.mean(
                        [record["conditional_added_excess_nll"] for record in records]
                    )
                ),
                "conditional_interaction_nll_mean": float(
                    np.mean(
                        [record["conditional_interaction_nll"] for record in records]
                    )
                ),
                "conditional_amplification_ratio_mean": (
                    float(np.mean(point_ratios))
                    if len(point_ratios) == len(records)
                    else None
                ),
                "positive_interaction_wins": positive_wins,
                "bootstrap": {
                    "samples": bootstrap_samples,
                    "seed": bootstrap_seed + 200,
                    "familywise_alpha": familywise_alpha,
                    "comparisons": len(added_layers),
                    "adjusted_confidence_level": 1 - adjusted_alpha,
                    "conditional_interaction_nll_adjusted_ci": interaction_ci,
                    "conditional_amplification_ratio_adjusted_ci": ratio_ci,
                },
                "context_sensitive_layer_detected": context_sensitive,
            }
        )
    return layers


def _layout_analysis(
    *,
    seeds: tuple[int, ...],
    original_metric: Mapping,
    composed_metrics: Mapping[str, Mapping],
    bootstrap_samples: int,
    bootstrap_seed: int,
    familywise_alpha: float,
) -> dict:
    labels = ("EARLY", "BALANCED", "LATE")
    original = _windows(original_metric)
    windows = len(original)
    metrics = {
        model_seed: {
            "EARLY": composed_metrics[layout_branch_name(model_seed, "EARLY")],
            "BALANCED": composed_metrics[
                composition_branch_name(model_seed, 12)
            ],
            "LATE": composed_metrics[layout_branch_name(model_seed, "LATE")],
        }
        for model_seed in seeds
    }
    layout_records = []
    for label in labels:
        per_seed = [
            {
                "seed": model_seed,
                "excess_nll": float(metrics[model_seed][label]["mean_nll"])
                - float(original_metric["mean_nll"]),
            }
            for model_seed in seeds
        ]
        layout_records.append(
            {
                "layout": label,
                "per_seed": per_seed,
                "excess_nll_mean": float(
                    np.mean([record["excess_nll"] for record in per_seed])
                ),
            }
        )
    rng = np.random.default_rng(bootstrap_seed + 300)
    indices = rng.integers(0, windows, size=(bootstrap_samples, windows))
    comparisons = []
    pairs = (("EARLY", "BALANCED"), ("EARLY", "LATE"), ("BALANCED", "LATE"))
    adjusted_alpha = familywise_alpha / len(pairs)
    for left, right in pairs:
        seed_draws = []
        per_seed = []
        for model_seed in seeds:
            left_values = _windows(metrics[model_seed][left], windows)
            right_values = _windows(metrics[model_seed][right], windows)
            difference = float(metrics[model_seed][left]["mean_nll"]) - float(
                metrics[model_seed][right]["mean_nll"]
            )
            per_seed.append({"seed": model_seed, "nll_difference": difference})
            seed_draws.append(
                np.mean((left_values - right_values)[indices], axis=1)
            )
        draws = np.mean(np.stack(seed_draws, axis=1), axis=1)
        interval = _interval(draws, adjusted_alpha)
        comparisons.append(
            {
                "left": left,
                "right": right,
                "per_seed": per_seed,
                "mean_nll_difference": float(
                    np.mean([record["nll_difference"] for record in per_seed])
                ),
                "adjusted_ci": interval,
                "resolved": bool(interval[0] > 0 or interval[1] < 0),
            }
        )
    best = min(layout_records, key=lambda record: record["excess_nll_mean"])
    worst = max(layout_records, key=lambda record: record["excess_nll_mean"])
    return {
        "replacement_count": 12,
        "layouts": layout_records,
        "pairwise_comparisons": comparisons,
        "best_layout": best["layout"],
        "worst_layout": worst["layout"],
        "layout_spread_nll": worst["excess_nll_mean"] - best["excess_nll_mean"],
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed + 300,
            "familywise_alpha": familywise_alpha,
            "comparisons": len(pairs),
            "adjusted_confidence_level": 1 - adjusted_alpha,
        },
    }


def analyze_composition_onset(
    *,
    seeds: Sequence[int],
    layer_sets: Mapping[int, Sequence[int]],
    original_metric: Mapping,
    standalone_metrics: Mapping[str, Mapping],
    composed_metrics: Mapping[str, Mapping],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260903,
    maximum_original_nll_difference: float = 0.01,
    maximum_mean_inflation: float = 1.25,
    maximum_seed_inflation: float = 1.50,
    maximum_bootstrap_upper: float = 1.50,
    familywise_alpha: float = 0.05,
) -> dict:
    sets = validate_progressive_layer_sets(layer_sets)
    counts = sorted(sets)
    if counts != [8, 10, 12, 14, 16]:
        raise ValueError("onset localization requires nested 8/10/12/14/16 sets")
    seeds = tuple(int(seed) for seed in seeds)
    if not 0 < familywise_alpha < 1:
        raise ValueError("familywise alpha must be in (0, 1)")
    added_layers = tuple(layer for layer in sets[16] if layer not in set(sets[8]))
    if len(added_layers) != 8:
        raise ValueError("onset localization requires eight added layers")
    progressive = analyze_progressive_composition(
        seeds=seeds,
        layer_sets=sets,
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
    incremental = _incremental_onset_stages(
        seeds=seeds,
        layer_sets=sets,
        standalone_metrics=standalone_metrics,
        composed_metrics=composed_metrics,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        familywise_alpha=familywise_alpha,
    )
    singles = _single_addition_analysis(
        seeds=seeds,
        base_count=8,
        added_layers=added_layers,
        standalone_metrics=standalone_metrics,
        composed_metrics=composed_metrics,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        familywise_alpha=familywise_alpha,
    )
    layouts = _layout_analysis(
        seeds=seeds,
        original_metric=original_metric,
        composed_metrics=composed_metrics,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        familywise_alpha=familywise_alpha,
    )
    detected_stages = [
        stage for stage in incremental if stage["superadditive_onset_detected"]
    ]
    context_sensitive_layers = [
        layer["layer"]
        for layer in singles
        if layer["context_sensitive_layer_detected"]
    ]
    progressive["global_16_layer_composition_gate_passed"] = progressive[
        "scientific_gate_passed"
    ]
    progressive["incremental_onset"] = {
        "stages": incremental,
        "earliest_detected_upper_count": (
            detected_stages[0]["upper_replacement_count"]
            if detected_stages
            else None
        ),
        "onset_localization_gate_passed": bool(detected_stages),
    }
    progressive["single_addition_attribution"] = {
        "base_replacement_count": 8,
        "layers": singles,
        "context_sensitive_layers": context_sensitive_layers,
        "first_order_attribution_gate_passed": bool(context_sensitive_layers),
    }
    progressive["matched_layouts"] = layouts
    progressive["onset_localization_gate_passed"] = bool(detected_stages)
    progressive["first_order_attribution_gate_passed"] = bool(
        context_sensitive_layers
    )
    progressive["scientific_gate_passed"] = bool(
        progressive["all_finite"]
        and detected_stages
        and context_sensitive_layers
    )
    return progressive
