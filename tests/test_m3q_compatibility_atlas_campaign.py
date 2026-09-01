from __future__ import annotations

import pytest

from extent.qwen_source import QWEN_SOURCES
from scripts.m3q_compatibility_atlas_campaign import (
    ARMS,
    CHECKPOINTS,
    LAYER_ORDER,
    SEEDS,
    SOURCE_MODEL,
    aggregate_atlas,
)


def _recovery(final: float) -> dict:
    evaluations = {
        str(step): {
            "decoder_output": {
                "relative_l2": final + (2048 - step) / 2048,
            }
        }
        for step in CHECKPOINTS
    }
    return {
        "complete": True,
        "completed_steps": 2048,
        "evaluations": evaluations,
    }


def _layer_result(layer: int) -> dict:
    # The final difficulty ordering is intentionally identical at steps 256/1024.
    dual_final = 0.01 * (layer + 1)
    seeds = {}
    for seed in SEEDS:
        seeds[str(seed)] = {
            "arms": {
                "CONTROL-RANDOM": {"recovery": _recovery(dual_final + 0.1)},
                "M3Q-DUAL-RANDOM": {"recovery": _recovery(dual_final)},
            }
        }
    return {
        "protocol": "exp067-m3q-full-depth-compatibility-atlas",
        "target_layer": layer,
        "complete": True,
        "passed": True,
        "seeds": seeds,
    }


def test_full_atlas_selects_four_hardest_layers_and_validates_early_ranking():
    aggregate = aggregate_atlas(
        {str(layer): _layer_result(layer) for layer in range(28)}
    )

    assert aggregate["completed_layers"] == 28
    assert aggregate["provisional_retained_attention_layers"] == [27, 26, 25, 24]
    assert aggregate["spearman_step256_to_final"] == pytest.approx(1.0)
    assert aggregate["spearman_step1024_to_final"] == pytest.approx(1.0)
    assert aggregate["scientific_gate_passed"]


def test_atlas_protocol_covers_every_qwen17_layer_once():
    assert len(LAYER_ORDER) == len(set(LAYER_ORDER)) == 28
    assert set(LAYER_ORDER) == set(range(28))
    assert SOURCE_MODEL in QWEN_SOURCES
    assert ARMS == ("CONTROL-RANDOM", "M3Q-DUAL-RANDOM")


def test_atlas_aggregation_supports_single_seed_confirmation():
    layers = {str(layer): _layer_result(layer) for layer in range(28)}
    for result in layers.values():
        result["seeds"] = {"789": result["seeds"]["123"]}
    aggregate = aggregate_atlas(layers, seeds=(789,))
    assert aggregate["completed_layers"] == 28
    assert aggregate["provisional_retained_attention_layers"] == [27, 26, 25, 24]
    assert aggregate["scientific_gate_passed"]
