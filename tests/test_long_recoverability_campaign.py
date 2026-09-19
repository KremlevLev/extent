from dataclasses import replace

from extent.config import tiny_config
from scripts.m3q_long_recoverability_campaign import (
    CHECKPOINTS,
    PROTOCOL,
    TOTAL_STEPS,
    aggregate,
    experiment_contract,
)


def _row(excess, kl, *, complete=True):
    return {
        "complete": complete,
        "evaluations": {
            str(step): {"excess_nll": value, "prediction_kl": kl + value}
            for step, value in excess.items()
        },
    }


def test_contract_is_long_horizon_and_pins_model_shape():
    config = replace(tiny_config(), attention_layer_indices=(0,))
    contract = experiment_contract(config)
    assert contract["protocol"] == PROTOCOL
    assert contract["total_steps"] == TOTAL_STEPS
    assert contract["tokens_per_trajectory"] == TOTAL_STEPS * 256
    assert contract["checkpoints"] == list(CHECKPOINTS)
    assert contract["model"]["attention_layer_indices"] == (0,)


def test_aggregate_requires_exact_to_win_both_metrics_at_both_seeds():
    result = {"trajectories": {}}
    for seed in (123, 456):
        result["trajectories"][str(seed)] = {
            "RANDOM": _row({TOTAL_STEPS: 2.0}, 3.0),
            "EXACT-LIFT": _row({TOTAL_STEPS: 1.0}, 2.0),
        }
    assert aggregate(result)["scientific_gate_passed"]
    result["trajectories"]["456"]["EXACT-LIFT"] = _row(
        {TOTAL_STEPS: 2.1}, 2.0
    )
    summary = aggregate(result)
    assert not summary["scientific_gate_passed"]
    assert summary["exact_endpoint_wins"] == 1


def test_partial_resume_curve_is_reported_but_cannot_pass():
    result = {
        "trajectories": {
            "123": {
                "RANDOM": _row({8192: 5.0}, 4.0, complete=False),
                "EXACT-LIFT": _row({8192: 4.0}, 4.0, complete=False),
            }
        }
    }
    summary = aggregate(result)
    assert summary["paired"]["123"]["shared_checkpoints"] == [8192]
    assert not summary["complete"]
    assert not summary["scientific_gate_passed"]
