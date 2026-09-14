from scripts.m3q_pair_threshold_campaign import OVERRIDES, configured_contract


def test_exp086_contract_is_disjoint_and_changes_only_pair_acceptance_threshold():
    contract = configured_contract()
    extension = contract["interaction_aware_extension"]

    assert contract["replications"] == [0, 1, 2, 3]
    assert contract["primary_arm"] == "STRICT-PAIR-0.5PCT"
    assert extension["decision_group_size"] == 2
    assert extension["primary_group_selection_mode"] == "pair_consensus"
    assert extension["control_group_selection_mode"] == "pair_consensus"
    assert extension["arm_minimum_relative_calibration_kl_gain"] == {
        "STANDARD-PAIR-0.1PCT": 0.001,
        "STRICT-PAIR-0.5PCT": 0.005,
    }
    assert contract["proposal_and_primary_calibration"]["proposal_offsets"][0] == 1_392_640
    assert contract["secondary_calibration"]["offsets"][0] == 114_688
    assert contract["locked_evaluation"]["offsets"][0] == 163_840
    assert OVERRIDES["HF_PREFIX"] == "experiments/exp086-pair-threshold"
