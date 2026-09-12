from scripts.m3q_robust_consensus_campaign import OVERRIDES, configured_contract


def test_robust_campaign_uses_fresh_ranges_and_matched_control():
    contract = configured_contract()
    assert contract["primary_arm"] == "ROBUST-CONSENSUS"
    assert contract["arms"] == ["MEAN-CONSENSUS", "ROBUST-CONSENSUS"]
    assert contract["proposal_and_primary_calibration"]["proposal_offsets"][0] == 557_056
    assert contract["secondary_calibration"]["offsets"][0] == 65_536
    assert contract["locked_evaluation"]["offsets"][0] == 65_536
    assert contract["robust_selection_extension"][
        "minimum_window_improvement_fraction_per_domain"
    ] == 0.60
    assert OVERRIDES["CONTROL_SELECTION_MODE"] == "dual_consensus"
