from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from extent.endpoint_checkpoint import (
    compact_endpoint_checkpoint,
    endpoint_checkpoint_supports_arms,
    restore_endpoint_checkpoint,
    save_endpoint_checkpoint,
)


def test_endpoint_checkpoint_round_trip_and_contract(tmp_path):
    endpoints = {
        "123": {
            "MIXER-ONLY": {"kernel": jnp.arange(6).reshape(2, 3)},
            "JOINT-MIXER-DECODER": {"kernel": jnp.ones((2, 3))},
        }
    }
    compatibility = {
        "source": "teacher@revision",
        "target_layer": 18,
        "seeds": [123],
    }
    metadata = save_endpoint_checkpoint(
        tmp_path, endpoints, compatibility=compatibility
    )
    restored, restored_metadata = restore_endpoint_checkpoint(
        tmp_path, expected_compatibility=compatibility
    )
    np.testing.assert_array_equal(
        restored["123"]["MIXER-ONLY"]["kernel"],
        endpoints["123"]["MIXER-ONLY"]["kernel"],
    )
    assert restored_metadata["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert metadata["checkpoint_bytes"] > 0
    assert endpoint_checkpoint_supports_arms(
        tmp_path, {"MIXER-ONLY", "JOINT-MIXER-DECODER"}
    )
    with pytest.raises(ValueError, match="compatibility"):
        restore_endpoint_checkpoint(
            tmp_path,
            expected_compatibility={**compatibility, "target_layer": 0},
        )


def test_endpoint_checkpoint_rejects_corruption(tmp_path):
    compatibility = {"source": "teacher@revision"}
    save_endpoint_checkpoint(
        tmp_path,
        {"123": {"JOINT": {"weight": jnp.ones(2)}}},
        compatibility=compatibility,
    )
    metadata = json.loads((tmp_path / "checkpoint.json").read_text())
    payload = tmp_path / metadata["checkpoint_file"]
    payload.write_bytes(payload.read_bytes() + b"broken")
    with pytest.raises(ValueError, match="byte-size"):
        restore_endpoint_checkpoint(
            tmp_path, expected_compatibility=compatibility
        )


def test_endpoint_checkpoint_compaction_is_verified_and_selective(tmp_path):
    endpoints = {
        "123": {
            "MIXER-ONLY": {"kernel": jnp.zeros((32, 32))},
            "JOINT-MIXER-DECODER": {"kernel": jnp.ones((32, 32))},
            "COUNTERFACTUAL-CONTRIBUTION": {
                "kernel": 2 * jnp.ones((32, 32))
            },
        }
    }
    compatibility = {"source": "teacher@revision", "seeds": [123]}
    full = save_endpoint_checkpoint(
        tmp_path, endpoints, compatibility=compatibility
    )
    compact = compact_endpoint_checkpoint(
        tmp_path,
        retained_arms={"JOINT-MIXER-DECODER"},
        expected_compatibility=compatibility,
    )
    assert compact["checkpoint_bytes"] < full["checkpoint_bytes"]
    assert compact["retained_arms"] == ["JOINT-MIXER-DECODER"]
    assert not endpoint_checkpoint_supports_arms(tmp_path, {"MIXER-ONLY"})
    assert endpoint_checkpoint_supports_arms(
        tmp_path, {"JOINT-MIXER-DECODER"}
    )
    restored, _ = restore_endpoint_checkpoint(
        tmp_path, expected_compatibility=compatibility
    )
    assert set(restored["123"]) == {"JOINT-MIXER-DECODER"}
    np.testing.assert_array_equal(
        restored["123"]["JOINT-MIXER-DECODER"]["kernel"],
        endpoints["123"]["JOINT-MIXER-DECODER"]["kernel"],
    )
