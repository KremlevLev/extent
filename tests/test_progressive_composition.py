from __future__ import annotations

import copy

import numpy as np
import pytest

from extent.progressive_composition import (
    analyze_progressive_composition,
    progressive_branch_names,
    validate_progressive_layer_sets,
)
from scripts.qwen_progressive_composition_campaign import (
    LAYER_BUDGETS,
    LAYER_SETS,
    TARGET_LAYERS,
    extract_standalone_joint_metrics,
)


SEEDS = (123, 456, 789)


def _metric(values: np.ndarray) -> dict:
    return {
        "mean_nll": float(np.mean(values)),
        "window_mean_nll": values.tolist(),
        "finite": True,
    }


def _synthetic_payload(inflation: float = 1.1):
    original_values = np.asarray([4.0, 4.1, 3.9, 4.05], dtype=np.float64)
    original = _metric(original_values)
    standalone = {}
    layer_excess = {}
    for index, layer in enumerate(TARGET_LAYERS):
        excess = 0.01 + index * 0.001
        layer_excess[layer] = excess
        standalone[str(layer)] = {
            "original": _metric(original_values),
            "seeds": {
                str(seed): _metric(original_values + excess)
                for seed in SEEDS
            },
        }
    composed = {"ORIGINAL-CACHED-QWEN": original}
    for seed in SEEDS:
        for count, layers in LAYER_SETS.items():
            additive = sum(layer_excess[layer] for layer in layers)
            composed[f"SEED-{seed}-COMPOSED-{count}"] = _metric(
                original_values + inflation * additive
            )
    return original, standalone, composed


def test_progressive_sets_and_branch_order_are_frozen():
    assert validate_progressive_layer_sets(LAYER_SETS) == LAYER_SETS
    assert LAYER_BUDGETS[0] == 8192
    assert all(
        LAYER_BUDGETS[layer] == 2048 for layer in TARGET_LAYERS if layer != 0
    )
    assert progressive_branch_names(SEEDS, LAYER_SETS)[:5] == (
        "ORIGINAL-CACHED-QWEN",
        "SEED-123-COMPOSED-2",
        "SEED-123-COMPOSED-4",
        "SEED-123-COMPOSED-8",
        "SEED-456-COMPOSED-2",
    )
    with pytest.raises(ValueError, match="strictly nested"):
        validate_progressive_layer_sets({2: (0, 18), 4: (0, 6, 12, 29)})


def test_progressive_analysis_accepts_controlled_subcritical_inflation():
    original, standalone, composed = _synthetic_payload(1.1)
    result = analyze_progressive_composition(
        seeds=SEEDS,
        layer_sets=LAYER_SETS,
        original_metric=original,
        standalone_metrics=standalone,
        composed_metrics=composed,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert result["baseline_reproduction"]["passed"]
    assert result["primary_replacement_count"] == 8
    assert result["scientific_gate_passed"]
    assert result["stages"][-1]["composition_inflation_ratio_mean"] == pytest.approx(
        1.1
    )


def test_progressive_primary_gate_rejects_superlinear_eight_layer_shock():
    original, standalone, composed = _synthetic_payload(1.1)
    bad = copy.deepcopy(composed)
    original_values = np.asarray(original["window_mean_nll"])
    additive = sum(
        standalone[str(layer)]["seeds"]["123"]["mean_nll"]
        - standalone[str(layer)]["original"]["mean_nll"]
        for layer in LAYER_SETS[8]
    )
    for seed in SEEDS:
        bad[f"SEED-{seed}-COMPOSED-8"] = _metric(
            original_values + 2.0 * additive
        )
    result = analyze_progressive_composition(
        seeds=SEEDS,
        layer_sets=LAYER_SETS,
        original_metric=original,
        standalone_metrics=standalone,
        composed_metrics=bad,
        bootstrap_samples=100,
    )
    assert not result["scientific_gate_passed"]
    assert not result["stages"][-1]["scientific_gate_passed"]


def test_baseline_parity_failure_is_scientific_not_numerical_failure():
    original, standalone, composed = _synthetic_payload(1.1)
    standalone["6"]["original"]["mean_nll"] += 0.1
    result = analyze_progressive_composition(
        seeds=SEEDS,
        layer_sets=LAYER_SETS,
        original_metric=original,
        standalone_metrics=standalone,
        composed_metrics=composed,
        bootstrap_samples=100,
    )
    assert result["all_finite"]
    assert not result["baseline_reproduction"]["passed"]
    assert not result["scientific_gate_passed"]


def test_standalone_metric_extraction_keeps_only_paired_joint_surface():
    metrics = {
        "ORIGINAL-CACHED-QWEN": {
            "mean_nll": 4.0,
            "window_mean_nll": [3.9, 4.1],
            "finite": True,
            "large_unused": [0] * 100,
        }
    }
    for seed in SEEDS:
        metrics[f"SEED-{seed}-JOINT-STEP2048"] = {
            "mean_nll": 4.1,
            "window_mean_nll": [4.0, 4.2],
            "finite": True,
            "large_unused": [0] * 100,
        }
    cell = {"layer_results": {"6": {"lm_metrics": metrics}}}
    compact = extract_standalone_joint_metrics(cell, layer=6, steps=2048)
    assert set(compact) == {"original", "seeds"}
    assert "large_unused" not in compact["original"]
    assert set(compact["seeds"]) == {"123", "456", "789"}
