from scripts.m3q_trust_region_sweep_campaign import (
    selected_coordinate_count,
    selected_layers,
)


def test_pair_decision_numeric_dictionary_values_not_keys_are_counted():
    decision = {
        "layers": {"1": 12, "0": 11},
        "selected_alphas": {"1": 0.0, "0": 0.5},
    }
    assert selected_layers(decision) == [11]
    assert selected_coordinate_count({"2": decision}) == 1
