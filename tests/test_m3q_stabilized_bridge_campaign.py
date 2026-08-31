from __future__ import annotations

from extent.qwen_source import QWEN_SOURCES
from scripts.m3q_stabilized_bridge_campaign import (
    ARMS,
    CACHE_START_RESERVE_SECONDS,
    SOURCE_MODEL,
    aggregate_results,
)


def _recovery(final: float):
    return {
        "complete": True,
        "completed_steps": 8192,
        "evaluations": {
            "0": {"decoder_output": {"relative_l2": 2.0 * final}},
            "4096": {"decoder_output": {"relative_l2": 1.3 * final}},
            "8192": {"decoder_output": {"relative_l2": final}},
        },
    }


def test_stabilized_bridge_gate_requires_primary_and_complex_wins():
    layers = {}
    for layer in (0, 6, 13, 20, 27):
        seeds = {}
        for seed in (123, 456, 789):
            finals = {
                "CONTROL-RANDOM": 1.0,
                "BALANCED-RANK-LIFT": 0.95,
                "M3Q-DUAL-RANDOM": 0.90,
                "M3Q-DUAL-RANDOM-PRECAL-WHITENED-NO-COMPLEX": 0.85,
                "M3Q-DUAL-RANDOM-PRECAL-WHITENED": 0.70,
            }
            seeds[str(seed)] = {
                "arms": {arm: {"recovery": _recovery(finals[arm])} for arm in ARMS}
            }
        layers[str(layer)] = {"seeds": seeds}
    aggregate = aggregate_results(layers)
    assert aggregate["completed_layers"] == 5
    assert aggregate["scientific_gate_passed"]
    assert aggregate["layer_gate_counts"]["primary_beats_legacy_dual"] == 5
    assert aggregate["layer_gate_counts"]["complex_beats_no_complex"] == 5


def test_stabilized_campaign_source_and_runtime_reserve_are_valid():
    assert SOURCE_MODEL in QWEN_SOURCES
    assert CACHE_START_RESERVE_SECONDS >= 30 * 60
