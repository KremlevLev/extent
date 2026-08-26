from __future__ import annotations

import copy

import numpy as np

from extent.boundary_composition import analyze_boundary_scaling
from scripts.qwen_boundary_scaling_campaign import (
    BRANCH_SETS,
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


def _synthetic_boundary_payload() -> tuple[dict, dict, dict]:
    original_values = np.asarray([4.0, 4.1, 3.9, 4.05], dtype=np.float64)
    original = _metric(original_values)
    layer_excess = {
        layer: (0.80 if layer == 0 else 0.01) for layer in TARGET_LAYERS
    }
    standalone = {
        str(layer): {
            "original": _metric(original_values),
            "seeds": {
                str(seed): _metric(original_values + layer_excess[layer])
                for seed in SEEDS
            },
        }
        for layer in TARGET_LAYERS
    }
    branch_excess = {
        "LAYER0-ONLY": 0.80,
        "INTERNAL-7": 0.07,
        "COMPOSED-8": 0.70,
        "INTERNAL-15": 0.15,
        "COMPOSED-16": 0.72,
    }
    composed = {"ORIGINAL-CACHED-QWEN": original}
    for seed in SEEDS:
        for label, excess in branch_excess.items():
            composed[f"SEED-{seed}-{label}"] = _metric(
                original_values + excess
            )
    return original, standalone, composed


def test_exp052_design_is_nested_and_has_boundary_counterfactuals():
    assert set(LAYER_SETS[8]) < set(LAYER_SETS[16])
    assert TARGET_LAYERS == LAYER_SETS[16]
    assert LAYER_BUDGETS[0] == 8192
    assert all(
        LAYER_BUDGETS[layer] == 2048 for layer in TARGET_LAYERS if layer != 0
    )
    assert BRANCH_SETS["INTERNAL-7"] == tuple(
        layer for layer in LAYER_SETS[8] if layer != 0
    )
    assert BRANCH_SETS["INTERNAL-15"] == tuple(
        layer for layer in LAYER_SETS[16] if layer != 0
    )
    specs = _branch_specs(SEEDS, LAYER_SETS, BRANCH_SETS)
    assert len(specs) == 16
    assert specs[0]["name"] == "ORIGINAL-CACHED-QWEN"
    assert specs[1]["name"] == "SEED-123-LAYER0-ONLY"
    assert specs[-1]["name"] == "SEED-789-COMPOSED-16"


def test_boundary_analysis_accepts_subadditive_16_layer_scaling():
    original, standalone, composed = _synthetic_boundary_payload()
    result = analyze_boundary_scaling(
        seeds=SEEDS,
        layer_sets=LAYER_SETS,
        original_metric=original,
        standalone_metrics=standalone,
        composed_metrics=composed,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert result["scientific_gate_passed"]
    assert result["composition_scaling_gate_passed"]
    assert result["boundary_mechanism_gate_passed"]
    assert result["primary_replacement_count"] == 16
    assert result["incremental_scaling"]["added_layers"] == [
        1,
        3,
        9,
        15,
        21,
        26,
        32,
        36,
    ]
    assert (
        result["incremental_scaling"]["incremental_inflation_ratio_mean"]
        < 1.0
    )
    primary_boundary = result["boundary_analysis"]["stages"][-1]
    assert primary_boundary["internal_only_excess_nll_mean"] < (
        primary_boundary["layer0_only_excess_nll_mean"]
    )
    assert primary_boundary["boundary_interaction_nll_mean"] < 0


def test_boundary_analysis_rejects_superlinear_16_layer_scaling():
    original, standalone, composed = _synthetic_boundary_payload()
    bad = copy.deepcopy(composed)
    original_values = np.asarray(original["window_mean_nll"])
    for seed in SEEDS:
        bad[f"SEED-{seed}-COMPOSED-16"] = _metric(original_values + 1.90)
    result = analyze_boundary_scaling(
        seeds=SEEDS,
        layer_sets=LAYER_SETS,
        original_metric=original,
        standalone_metrics=standalone,
        composed_metrics=bad,
        bootstrap_samples=100,
    )
    assert not result["scientific_gate_passed"]
    assert not result["composition_scaling_gate_passed"]
    assert not result["boundary_mechanism_gate_passed"]
