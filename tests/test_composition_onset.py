from __future__ import annotations

import copy

import numpy as np
import pytest

from extent.composition_onset import analyze_composition_onset
from scripts.qwen_composition_onset_campaign import (
    ADDED_LAYERS,
    BRANCH_SETS,
    EARLY_12,
    LATE_12,
    LAYER_BUDGETS,
    LAYER_SETS,
    SEEDS,
    TARGET_LAYERS,
)
from scripts.qwen_progressive_composition_eval import _branch_specs


def _metric(values: np.ndarray) -> dict:
    return {
        "mean_nll": float(np.mean(values)),
        "window_mean_nll": values.tolist(),
        "finite": True,
    }


def _payload() -> tuple[dict, dict, dict]:
    original_values = np.asarray([4.0, 4.1, 3.9, 4.05], dtype=np.float64)
    original = _metric(original_values)
    target_layers = LAYER_SETS[16]
    layer_excess = {
        layer: (0.70 if layer == 0 else 0.01) for layer in target_layers
    }
    standalone = {
        str(layer): {
            "original": _metric(original_values),
            "seeds": {
                str(seed): _metric(original_values + layer_excess[layer])
                for seed in SEEDS
            },
        }
        for layer in target_layers
    }
    composed = {"ORIGINAL-CACHED-QWEN": original}
    nested_excess = {8: 0.75, 10: 0.85, 12: 0.87, 14: 0.89, 16: 0.91}
    for seed in SEEDS:
        for count, excess in nested_excess.items():
            composed[f"SEED-{seed}-COMPOSED-{count}"] = _metric(
                original_values + excess
            )
        for layer in ADDED_LAYERS:
            conditional = 0.06 if layer == 1 else 0.01
            composed[f"SEED-{seed}-BASE8-PLUS-L{layer}"] = _metric(
                original_values + nested_excess[8] + conditional
            )
        composed[f"SEED-{seed}-EARLY-12"] = _metric(
            original_values + 0.95
        )
        composed[f"SEED-{seed}-LATE-12"] = _metric(
            original_values + 0.82
        )
    return original, standalone, composed


def test_exp053_design_is_frozen_and_balanced():
    assert tuple(LAYER_SETS) == (8, 10, 12, 14, 16)
    assert TARGET_LAYERS == LAYER_SETS[16]
    assert ADDED_LAYERS == (1, 3, 9, 15, 21, 26, 32, 36)
    assert LAYER_BUDGETS[0] == 8192
    assert all(
        LAYER_BUDGETS[layer] == 2048 for layer in TARGET_LAYERS if layer != 0
    )
    assert len(BRANCH_SETS) == 15
    assert len(EARLY_12) == len(LATE_12) == 12
    assert BRANCH_SETS["BASE8-PLUS-L21"] == tuple(
        sorted(set(LAYER_SETS[8]) | {21})
    )
    specs = _branch_specs(SEEDS, LAYER_SETS, BRANCH_SETS)
    assert len(specs) == 46
    assert specs[0]["name"] == "ORIGINAL-CACHED-QWEN"
    assert specs[-1]["name"] == "SEED-789-LATE-12"


def test_onset_analysis_localizes_first_pair_and_single_layer_culprit():
    original, standalone, composed = _payload()
    result = analyze_composition_onset(
        seeds=SEEDS,
        layer_sets=LAYER_SETS,
        original_metric=original,
        standalone_metrics=standalone,
        composed_metrics=composed,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert result["all_finite"]
    assert result["scientific_gate_passed"]
    assert result["onset_localization_gate_passed"]
    assert result["first_order_attribution_gate_passed"]
    assert result["incremental_onset"]["earliest_detected_upper_count"] == 10
    assert result["single_addition_attribution"][
        "context_sensitive_layers"
    ] == [1]
    transition = result["incremental_onset"]["stages"][0]
    assert transition["bootstrap"]["adjusted_confidence_level"] == pytest.approx(
        0.9875
    )
    layer = result["single_addition_attribution"]["layers"][0]
    assert layer["bootstrap"]["adjusted_confidence_level"] == pytest.approx(
        0.99375
    )
    assert result["matched_layouts"]["best_layout"] == "LATE"
    assert result["matched_layouts"]["worst_layout"] == "EARLY"


def test_onset_without_single_layer_culprit_fails_attribution_gate():
    original, standalone, composed = _payload()
    no_culprit = copy.deepcopy(composed)
    original_values = np.asarray(original["window_mean_nll"])
    for seed in SEEDS:
        for layer in ADDED_LAYERS:
            no_culprit[f"SEED-{seed}-BASE8-PLUS-L{layer}"] = _metric(
                original_values + 0.76
            )
    result = analyze_composition_onset(
        seeds=SEEDS,
        layer_sets=LAYER_SETS,
        original_metric=original,
        standalone_metrics=standalone,
        composed_metrics=no_culprit,
        bootstrap_samples=100,
    )
    assert result["onset_localization_gate_passed"]
    assert not result["first_order_attribution_gate_passed"]
    assert not result["scientific_gate_passed"]


def test_negative_standalone_effect_does_not_hide_context_sensitive_harm():
    original, standalone, composed = _payload()
    original_values = np.asarray(original["window_mean_nll"])
    for seed in SEEDS:
        standalone["1"]["seeds"][str(seed)] = _metric(original_values - 0.01)
    result = analyze_composition_onset(
        seeds=SEEDS,
        layer_sets=LAYER_SETS,
        original_metric=original,
        standalone_metrics=standalone,
        composed_metrics=composed,
        bootstrap_samples=100,
    )
    layer1 = result["single_addition_attribution"]["layers"][0]
    assert layer1["context_sensitive_layer_detected"]
    assert layer1["conditional_amplification_ratio_mean"] is None
    assert layer1["bootstrap"][
        "conditional_amplification_ratio_adjusted_ci"
    ] is None
