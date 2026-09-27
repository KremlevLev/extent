import jax.numpy as jnp
import optax

from scripts import m3q_mixer_gain_campaign as campaign


def test_gate_needs_both_seeds_and_unchanged_start_improvement():
    result = {"branches": {
        "123": {
            "GLOBAL": {"complete": True, "locked_test_nll": 8.0, "start_test_nll": 9.0},
            "PER-LAYER": {"complete": True, "locked_test_nll": 7.9},
        },
        "456": {
            "GLOBAL": {"complete": True, "locked_test_nll": 8.0, "start_test_nll": 9.0},
            "PER-LAYER": {"complete": True, "locked_test_nll": 7.9},
        },
    }}
    assert campaign.aggregate(result)["scientific_gate_passed"]
    result["branches"]["456"]["PER-LAYER"]["locked_test_nll"] = 7.97
    assert not campaign.aggregate(result)["scientific_gate_passed"]


def test_tiny_optimizer_state_survives_json_roundtrip():
    raw = jnp.zeros((3,), dtype=jnp.float32)
    tx = optax.adam(0.01)
    state = tx.init(raw)
    _, state = tx.update(jnp.ones_like(raw), state, raw)
    saved = campaign._optimizer_record(state)
    restored = campaign._restore_optimizer(tx, raw, saved)
    assert campaign._optimizer_record(restored) == saved
