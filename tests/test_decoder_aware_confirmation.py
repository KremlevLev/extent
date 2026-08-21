from __future__ import annotations

import pytest

from scripts.qwen_decoder_aware_confirmation import (
    aggregate_confirmation,
    parse_seeds,
)


def _seed_result(improvement: float, degradation: float) -> dict:
    mixer_decoder = 0.6
    joint_decoder = mixer_decoder * (1.0 - improvement)
    mixer_mixer = 0.5
    joint_mixer = mixer_mixer * (1.0 + degradation)
    step0 = {
        "mixer_output": {"relative_l2": 0.8},
        "decoder_output": {"relative_l2": 0.7},
        "finite": True,
    }
    return {
        "passed": True,
        "arms": {
            "MIXER-ONLY": {
                "finite": True,
                "evaluations": {
                    "0": step0,
                    "1024": {
                        "mixer_output": {"relative_l2": mixer_mixer},
                        "decoder_output": {"relative_l2": mixer_decoder},
                    },
                },
            },
            "JOINT-MIXER-DECODER": {
                "finite": True,
                "evaluations": {
                    "0": step0.copy(),
                    "1024": {
                        "mixer_output": {"relative_l2": joint_mixer},
                        "decoder_output": {"relative_l2": joint_decoder},
                    },
                },
            },
        },
    }


def test_confirmation_gate_passes_only_complete_paired_contract():
    results = {
        "123": _seed_result(0.11, 0.02),
        "456": _seed_result(0.10, 0.03),
        "789": _seed_result(0.12, 0.01),
    }
    aggregate = aggregate_confirmation(results, 1024)
    assert aggregate["scientific_gate_passed"] is True
    assert aggregate["joint_decoder_wins"] == 3
    assert aggregate["step0_identical_all_seeds"] is True


def test_confirmation_gate_rejects_mean_or_per_seed_regression():
    weak = {
        "123": _seed_result(0.09, 0.02),
        "456": _seed_result(0.08, 0.03),
        "789": _seed_result(0.07, 0.01),
    }
    assert aggregate_confirmation(weak, 1024)["scientific_gate_passed"] is False
    regression = {
        "123": _seed_result(0.12, 0.02),
        "456": _seed_result(0.12, 0.11),
        "789": _seed_result(0.12, 0.01),
    }
    assert (
        aggregate_confirmation(regression, 1024)["scientific_gate_passed"]
        is False
    )


def test_seed_parser_requires_three_distinct_values():
    assert parse_seeds("123,456,789") == (123, 456, 789)
    with pytest.raises(ValueError, match="exactly three"):
        parse_seeds("123,123,789")
