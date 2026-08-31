from __future__ import annotations

from extent.qwen_source import QWEN_SOURCES
from scripts.m3q_complex_bridge_campaign import ARMS, SOURCE_MODEL, aggregate_results


def _recovery(final: float, middle: float | None = None):
    middle = final * 1.3 if middle is None else middle
    return {
        "complete": True,
        "completed_steps": 12288,
        "evaluations": {
            "0": {"decoder_output": {"relative_l2": final * 2.0}},
            "6144": {"decoder_output": {"relative_l2": middle}},
            "12288": {"decoder_output": {"relative_l2": final}},
        },
    }


def test_complex_bridge_aggregate_requires_depth_and_mechanism_wins():
    layers = {}
    for layer in (0, 6, 13, 20, 27):
        seeds = {}
        for seed in (123, 456, 789):
            finals = {
                "CONTROL-RANDOM": 1.0,
                "BALANCED-RANK-LIFT": 0.9,
                "M3Q-DUAL-RANDOM": 0.82,
                "M3Q-DUAL-EXACT-NO-COMPLEX": 0.78,
                "M3Q-DUAL-EXACT": 0.70,
            }
            seeds[str(seed)] = {
                "arms": {arm: {"recovery": _recovery(finals[arm])} for arm in ARMS}
            }
        layers[str(layer)] = {"seeds": seeds}
    aggregate = aggregate_results(layers)
    assert aggregate["completed_layers"] == 5
    assert aggregate["primary_screen_passed"]
    assert aggregate["mechanism_supported"]
    assert aggregate["scientific_gate_passed"]
    assert aggregate["layer_gate_counts"]["complex_phase_beats_no_complex"] == 5


def test_campaign_uses_a_registered_qwen_source_name():
    assert SOURCE_MODEL in QWEN_SOURCES
