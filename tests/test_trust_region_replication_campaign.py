import copy

from scripts.m3q_trust_region_replication_campaign import (
    BASE_TRAIN_OFFSET,
    EVAL_STRIDE,
    REPLICATIONS,
    TRAIN_STRIDE,
    aggregate,
    experiment_contract,
    replication_overrides,
)


def completed_replication(trust_delta=-0.2, trust_minus_hard=-0.4):
    branches = {}
    for seed in (123, 456):
        start = 10.0
        final = start + trust_delta
        hard_final = final - trust_minus_hard
        branches[str(seed)] = {
            "TRUST-LINE": {
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
            "HARD-ACCEPT": {
                "complete": True,
                "evaluations": {
                    "24": {"student_nll": hard_final, "window_nll": [hard_final] * 2}
                },
                "decisions": {},
            },
        }
    return {"status": "completed", "branches": branches}


def test_replication_offsets_are_disjoint_and_validation_is_locked():
    contract = experiment_contract()
    assert len(set(contract["proposal_offsets"])) == len(REPLICATIONS)
    assert contract["proposal_offsets"][1] - contract["proposal_offsets"][0] == TRAIN_STRIDE
    assert contract["locked_validation_offsets"][1] == EVAL_STRIDE
    assert "deepmind/pg19@4d28bd7" in contract["locked_evaluation_dataset"]
    overrides = replication_overrides(7)
    assert overrides["TRAIN_OFFSET"] == BASE_TRAIN_OFFSET + 7 * TRAIN_STRIDE
    assert overrides["EVAL_SPLIT"] == "validation"
    assert overrides["EVAL_DATASET_CONFIG"] == "pg19-pinned-manifest"


def test_aggregate_gate_uses_all_eight_replications_for_each_seed():
    result = {
        "replications": {
            str(replication): completed_replication() for replication in REPLICATIONS
        }
    }
    summary = aggregate(result)
    assert summary["scientific_gate_passed"]
    assert summary["per_seed"]["123"]["negative_replications"] == 8
    assert summary["per_seed"]["456"]["normal_95_upper"] < 0

    missing = copy.deepcopy(result)
    del missing["replications"]["7"]
    assert not aggregate(missing)["scientific_gate_passed"]


def test_aggregate_rejects_nonnegative_confidence_bound():
    result = {
        "replications": {
            str(replication): completed_replication(-0.2 if replication < 7 else 0.6)
            for replication in REPLICATIONS
        }
    }
    summary = aggregate(result)
    assert summary["per_seed"]["123"]["mean_delta"] < 0
    assert summary["per_seed"]["123"]["normal_95_upper"] > 0
    assert not summary["scientific_gate_passed"]
