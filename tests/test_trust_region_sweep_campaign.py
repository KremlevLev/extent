import jax.numpy as jnp
import numpy as np

from scripts.m3q_trust_region_sweep_campaign import (
    aggregate,
    blend_parameters,
    choose_consensus_alpha,
    choose_robust_consensus_alpha,
    choose_trust_alpha,
    experiment_contract,
)
from extent.config import load_config
from pathlib import Path


def test_trust_alpha_requires_registered_relative_gain():
    alpha, gain = choose_trust_alpha({"0": 10.0, "0.5": 9.98, "1": 10.5})
    assert alpha == 0.5
    assert np.isclose(gain, 0.002)
    alpha, gain = choose_trust_alpha({"0": 10.0, "0.5": 9.995, "1": 9.999})
    assert alpha == 0.0
    assert np.isclose(gain, 0.0005)
    assert choose_trust_alpha({"0": 10.0, "1": 10.1}) == (0.0, 0.0)


def test_consensus_alpha_must_improve_both_domains():
    assert choose_consensus_alpha(
        {"0": 10.0, "0.5": 9.0, "1": 8.0},
        {"0": 5.0, "0.5": 5.1, "1": 5.2},
    ) == (0.0, 0.0, 0.0)
    alpha, primary_gain, secondary_gain = choose_consensus_alpha(
        {"0": 10.0, "0.5": 9.5, "1": 9.0},
        {"0": 5.0, "0.5": 4.5, "1": 4.9},
    )
    assert alpha == 0.5
    assert np.isclose(primary_gain, 0.05)
    assert np.isclose(secondary_gain, 0.1)


def test_robust_consensus_rejects_mean_gain_without_window_majority():
    def metrics(mean, windows):
        return {"prediction_kl": mean, "window_prediction_kl": windows}

    primary = {
        "0": metrics(10.0, [1.0, 1.0, 1.0, 1.0]),
        "0.5": metrics(9.0, [0.5, 0.5, 1.1, 1.1]),
    }
    secondary = {
        "0": metrics(5.0, [1.0, 1.0, 1.0, 1.0]),
        "0.5": metrics(4.0, [0.5, 0.5, 0.5, 1.1]),
    }
    assert choose_robust_consensus_alpha(primary, secondary) == (
        0.0, 0.0, 0.0, 0.0, 0.0
    )
    primary["0.5"] = metrics(9.0, [0.5, 0.5, 0.5, 1.1])
    selected = choose_robust_consensus_alpha(primary, secondary)
    assert selected[0] == 0.5
    assert selected[3:] == (0.75, 0.75)


def test_blend_parameters_preserves_dtype_and_endpoints():
    current = {"w": jnp.asarray([0.0, 2.0], jnp.bfloat16)}
    proposal = {"w": jnp.asarray([2.0, 4.0], jnp.bfloat16)}
    zero = blend_parameters(current, proposal, 0.0)
    half = blend_parameters(current, proposal, 0.5)
    one = blend_parameters(current, proposal, 1.0)
    assert half["w"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(zero["w"], np.float32), [0.0, 2.0])
    np.testing.assert_array_equal(np.asarray(half["w"], np.float32), [1.0, 3.0])
    np.testing.assert_array_equal(np.asarray(one["w"], np.float32), [2.0, 4.0])


def test_registered_gate_requires_both_seeds_and_matched_control():
    result = {"branches": {}}
    for seed in (123, 456):
        result["branches"][str(seed)] = {}
        for arm, final in (("HARD-ACCEPT", 9.5), ("TRUST-LINE", 9.0)):
            result["branches"][str(seed)][arm] = {
                "complete": True,
                "evaluations": {
                    str(step): {"student_nll": 10.0 if step == 0 else final}
                    for step in (0, 6, 12, 18, 24)
                },
                "decisions": {"1": {"selected_alpha": 0.25}},
            }
    assert aggregate(result)["scientific_gate_passed"]
    result["branches"]["456"]["TRUST-LINE"]["evaluations"]["24"]["student_nll"] = 10.1
    assert not aggregate(result)["scientific_gate_passed"]


def test_contract_keeps_calibration_and_locked_test_distinct():
    config, _ = load_config(
        Path(__file__).resolve().parents[1] / "config/hybrid_1_7b_gqa_v5e8.yaml"
    )
    contract = experiment_contract(config)
    assert contract["calibration"][0] == "train"
    assert contract["locked_evaluation"][0] == "test"
    assert 0.0 in contract["alphas"]["TRUST-LINE"]
