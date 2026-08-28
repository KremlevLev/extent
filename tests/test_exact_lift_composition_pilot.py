import pytest

from scripts.qwen_exact_lift_composition_pilot import LAYERS, PSEUDO_SEEDS, STEPS
from scripts.qwen_progressive_composition_eval import (
    analyze_exact_lift_multiseed,
    analyze_exact_lift_pilot,
)


def _metric(value):
    return {"mean_nll": value, "finite": True, "window_mean_nll": [value]}


def test_pilot_protocol_is_minimal_and_paired():
    assert LAYERS == (0, 18)
    assert PSEUDO_SEEDS == (123, 456)
    assert STEPS == 1024


def test_pilot_gate_compares_exact_and_random_compositions():
    aggregate = analyze_exact_lift_pilot(
        original_metric=_metric(3.0),
        composed_metrics={
            "SEED-123-COMPOSED-2": _metric(3.4),
            "SEED-456-COMPOSED-2": _metric(3.2),
        },
    )
    assert aggregate["pilot_gate_passed"]
    assert aggregate["scientific_gate_passed"]
    assert aggregate["exact_minus_random_nll"] == pytest.approx(-0.2)


def test_multiseed_gate_uses_paired_windows_and_uncertainty():
    metrics = {}
    for seed in (123, 456, 789):
        metrics[f"SEED-{seed}-COMPOSED-2"] = {
            "mean_nll": 4.0, "window_mean_nll": [4.0] * 32,
        }
        metrics[f"SEED-{seed + 1000}-COMPOSED-2"] = {
            "mean_nll": 3.8, "window_mean_nll": [3.8] * 32,
        }
    result = analyze_exact_lift_multiseed(
        original_metric=_metric(3.0), composed_metrics=metrics,
        bootstrap_samples=200, bootstrap_seed=7,
    )
    assert result["scientific_gate_passed"]
    assert result["wins"] == 3
    assert result["bootstrap_95_ci"][1] < 0
