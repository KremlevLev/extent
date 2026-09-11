import jax.numpy as jnp
import numpy as np

from scripts.m3q_trust_region_sweep_campaign import (
    aggregate,
    blend_parameters,
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
