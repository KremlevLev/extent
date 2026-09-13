from scripts.m3q_paired_lookahead_campaign import (
    OVERRIDES,
    configured_aggregate,
    configured_contract,
)


def completed_replication():
    branches = {}
    for seed in (123, 456):
        branches[str(seed)] = {
            "PAIR-LOOKAHEAD": {
                "complete": True,
                "evaluations": {
                    str(step): {
                        "student_nll": 10.0 if step == 0 else 9.5,
                        "window_nll": [10.0, 10.1]
                        if step == 0 else [9.5, 9.6],
                    }
                    for step in (0, 6, 12, 18, 24)
                },
                "decisions": {
                    "2": {
                        "layers": [0, 1],
                        "selected_alphas": [0.25, 0.5],
                        "joint_only_rescue": True,
                    }
                },
            },
            "GREEDY-PAIR-GRID": {
                "complete": True,
                "evaluations": {"24": {"student_nll": 9.8}},
                "decisions": {},
            },
        }
    return {"status": "completed", "branches": branches}


def test_exp085_contract_freezes_matched_pair_grid_and_fresh_ranges():
    contract = configured_contract()
    extension = contract["interaction_aware_extension"]
    assert contract["replications"] == [0, 1, 2, 3]
    assert contract["primary_arm"] == "PAIR-LOOKAHEAD"
    assert extension["decision_group_size"] == 2
    assert extension["primary_group_selection_mode"] == "pair_consensus"
    assert extension["control_group_selection_mode"] == "greedy_pair_grid"
    assert contract["proposal_and_primary_calibration"]["proposal_offsets"][0] == 1_114_112
    assert contract["secondary_calibration"]["offsets"][0] == 98_304
    assert contract["locked_evaluation"]["offsets"][0] == 131_072
    assert OVERRIDES["INNER_MAX_WALL_HOURS"] == 2.0


def test_exp085_gate_requires_replication_and_joint_rescue():
    result = {
        "replications": {
            str(rep): completed_replication() for rep in range(4)
        }
    }
    assert configured_aggregate(result)["scientific_gate_passed"]
    for replication in result["replications"].values():
        for seed in ("123", "456"):
            replication["branches"][seed]["PAIR-LOOKAHEAD"]["decisions"]["2"][
                "joint_only_rescue"
            ] = False
    assert not configured_aggregate(result)["scientific_gate_passed"]
