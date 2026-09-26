from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from extent.config import tiny_config
from extent.global_interpolation import interpolate_parameters, select_alpha
from scripts import m3q_global_path_probe as exp092


def test_parameter_interpolation_keeps_bf16_dtype_and_endpoints():
    start = {"layers_0": {"mamba": jnp.asarray([0.0, 2.0], jnp.bfloat16)}}
    trained = {"layers_0": {"mamba": jnp.asarray([2.0, 4.0], jnp.bfloat16)}}
    assert interpolate_parameters(start, trained, 0.0) is start
    assert interpolate_parameters(start, trained, 1.0) is trained
    midpoint = interpolate_parameters(start, trained, 0.5)
    assert midpoint["layers_0"]["mamba"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(midpoint["layers_0"]["mamba"], [1.0, 3.0])
    with pytest.raises(ValueError):
        interpolate_parameters(start, trained, 1.1)


def test_calibration_selection_never_uses_locked_results():
    assert select_alpha({0.0: 10.0, 0.25: 9.8, 1.0: 11.0}, min_gain=0.01) == 0.25
    assert select_alpha({0.0: 10.0, 0.25: 9.995, 1.0: 11.0}, min_gain=0.01) == 0.0
    assert select_alpha({0.0: 10.0, 0.25: 9.8, 0.5: 9.8}, min_gain=0.01) == 0.25


def test_registered_gate_requires_locked_gain_at_both_anchor_seeds():
    def row(alpha, baseline, selected):
        return {"selected_alpha": alpha,
                "start_locked": {"student_nll": baseline},
                "selected_locked": {"student_nll": selected}}

    result = {"branches": {
        "123": {"ONPOLICY-PLAIN": row(0.0, 11.0, 11.0),
                "ONPOLICY-ANCHOR": row(0.25, 11.0, 10.8)},
        "456": {"ONPOLICY-ANCHOR": row(0.125, 12.0, 11.8)},
    }}
    assert exp092.aggregate(result)["scientific_gate_passed"]
    result["branches"]["456"]["ONPOLICY-ANCHOR"] = row(0.0, 12.0, 12.0)
    assert not exp092.aggregate(result)["scientific_gate_passed"]


def test_contract_pins_fresh_split_and_alpha_grid():
    config = replace(tiny_config(), attention_layer_indices=(0,))
    source = {"contract": {"source": "Qwen/test@rev", "protocol": "exp091",
                           "model": {"num_layers": config.num_layers}},
              "data_sha256": {"train": "abc", "validation": "def"}}
    contract = exp092.contract_for(config, source)
    assert contract["calibration"]["split"] == "validation"
    assert contract["locked_test"]["split"] == "test"
    assert contract["alphas"][0] == 0.0
    assert contract["alphas"][-1] == 1.0
