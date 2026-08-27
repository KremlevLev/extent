import pytest

from scripts.qwen_exact_lift_composition_pilot import LAYERS, PSEUDO_SEEDS, STEPS
from scripts.qwen_progressive_composition_eval import analyze_exact_lift_pilot


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
