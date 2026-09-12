import copy

from scripts.m3q_dual_domain_trust_campaign import (
    PRIMARY_ARM,
    REPLICATIONS,
    aggregate,
    experiment_contract,
    replication_overrides,
)


def completed_replication(delta=-0.3, versus_control=-0.2):
    branches = {}
    for seed in (123, 456):
        start, final = 10.0, 10.0 + delta
        control_final = final - versus_control
        branches[str(seed)] = {
            PRIMARY_ARM: {
                "complete": True,
                "evaluations": {
                    str(step): {
                        "student_nll": start if step == 0 else final,
                        "window_nll": [start, start + 0.1]
                        if step == 0 else [final, final + 0.1],
                    }
                    for step in (0, 6, 12, 18, 24)
                },
                "decisions": {"1": {"layer": 0, "selected_alpha": 0.25}},
            },
            "WIKI-ONLY": {
                "complete": True,
                "evaluations": {"24": {"student_nll": control_final}},
                "decisions": {},
            },
        }
    return {"status": "completed", "branches": branches}


def test_dual_domain_contract_keeps_pg19_test_locked():
    contract = experiment_contract()
    assert "test manifest" in contract["locked_evaluation"]["dataset"]
    assert contract["secondary_calibration"]["windows"] == 16
    overrides = replication_overrides(3)
    assert overrides["ARM_SELECTION_MODE"][PRIMARY_ARM] == "dual_consensus"
    assert overrides["EVAL_SPLIT"] == "test"
    assert overrides["SECONDARY_CALIBRATION"]["split"] == "validation"


def test_dual_domain_gate_uses_all_repetitions_and_both_seeds():
    result = {
        "replications": {
            str(rep): completed_replication() for rep in REPLICATIONS
        }
    }
    assert aggregate(result)["scientific_gate_passed"]
    missing = copy.deepcopy(result)
    del missing["replications"]["7"]
    assert not aggregate(missing)["scientific_gate_passed"]


def test_dual_domain_gate_rejects_unstable_seed():
    result = {
        "replications": {
            str(rep): completed_replication(0.2 if rep >= 5 else -0.1)
            for rep in REPLICATIONS
        }
    }
    assert not aggregate(result)["scientific_gate_passed"]
