from scripts.qwen17_full_model_distill_campaign import (
    CHECKPOINTS,
    aggregate_results,
)


def _arm(scale: float):
    return {
        "complete": True,
        "evaluations": {
            str(step): {
                "excess_nll": scale * (1.0 - step / 10_000),
                "prediction_kl": scale * (1.0 - step / 12_000),
            }
            for step in CHECKPOINTS
        },
    }


def test_full_model_gate_requires_staged_final_and_curve_wins():
    aggregate = aggregate_results(
        {
            "EXACT-KL": _arm(1.0),
            "M3Q-HIDDEN-BRIDGE-KL": _arm(0.8),
            "RANDOM-KL": _arm(1.2),
        },
        CHECKPOINTS,
    )
    assert aggregate["complete_primary_pair"] is True
    assert aggregate["final_staged_relative_gain"] > 0
    assert aggregate["staged_auc_relative_gain"] > 0
    assert aggregate["scientific_gate_passed"] is True


def test_full_model_gate_never_passes_a_partial_primary_pair():
    aggregate = aggregate_results({"EXACT-KL": _arm(1.0)}, CHECKPOINTS)
    assert aggregate["complete_primary_pair"] is False
    assert aggregate["scientific_gate_passed"] is False
