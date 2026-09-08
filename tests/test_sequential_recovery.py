from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from extent.config import tiny_config
from extent.model import HybridDecoderLayer
from extent.sequential_recovery import make_conditional_recovery_step, recovery_examples
from scripts.m3q_sequential_recovery_campaign import ARMS, MILESTONES, SEEDS, aggregate
from scripts import m3q_full_depth_sequential_confirmation as confirmation
from scripts import m3q_sequential_recovery_campaign as campaign


def test_recovery_arms_have_expected_matched_examples():
    values = [jnp.full((2, 3, 4), i) for i in range(4)]
    t_in, t_out, h_in, h_out = values
    assert recovery_examples("TEACHER", *values) == (t_in, t_out)
    assert recovery_examples("ONPOLICY", *values) == (h_in, h_out)
    x, y = recovery_examples("MIXED", *values)
    assert x.shape[0] == y.shape[0] == 4
    np.testing.assert_array_equal(x[:2], t_in)
    np.testing.assert_array_equal(x[2:], h_in)
    with pytest.raises(ValueError):
        recovery_examples("bad", *values)


def test_real_mamba_only_step_reduces_conditional_decoder_error():
    cfg = replace(tiny_config(), attention_layer_indices=(1,), compute_dtype="float32", param_dtype="float32")
    layer = HybridDecoderLayer(cfg, 0)
    inputs = jax.random.normal(jax.random.key(2), (2, 4, cfg.hidden_size))
    params = layer.init(jax.random.key(3), inputs, jnp.arange(4)[None], None)["params"]
    target_params = params["mamba"]
    candidate = jax.tree.map(lambda x: x + jax.random.normal(jax.random.key(x.size), x.shape) * .01, target_params)
    target = layer.apply({"params": params}, inputs, jnp.arange(4)[None], None)
    frozen = dict(params, mamba=candidate)
    tx = optax.adam(1e-3)
    step = jax.jit(make_conditional_recovery_step(layer, tx, bf16_gradients=False))
    state = tx.init(candidate)
    first = None
    for _ in range(8):
        candidate, state, metrics = step(candidate, state, frozen, inputs, target)
        first = float(metrics["loss"]) if first is None else first
    assert bool(metrics["grads_finite"])
    assert float(metrics["loss"]) < first
    # Frozen direct tensors are not optimizer parameters.
    for key in ("input_layernorm", "post_attention_layernorm", "mlp"):
        for a, b in zip(jax.tree.leaves(params[key]), jax.tree.leaves(frozen[key])):
            np.testing.assert_array_equal(a, b)


def test_scientific_gate_requires_both_mixed_seed_pairs_and_auc():
    result = {"branches": {}}
    for seed in SEEDS:
        result["branches"][str(seed)] = {}
        for arm in ARMS:
            offset = 0.0 if arm == "TEACHER" else (-0.1 if arm == "MIXED" else -0.05)
            result["branches"][str(seed)][arm] = {"evaluations": {
                str(depth): {"student_nll": 5.0 + offset} for depth in MILESTONES
            }}
    summary = aggregate(result)
    assert summary["completed_primary_pairs"] == 2
    assert summary["scientific_gate_passed"]
    del result["branches"][str(SEEDS[-1])]["MIXED"]["evaluations"][str(MILESTONES[-1])]
    assert not aggregate(result)["scientific_gate_passed"]


def test_full_depth_wrapper_applies_and_restores_registered_overrides(monkeypatch):
    original = {name: getattr(campaign, name) for name in confirmation.OVERRIDES}
    observed = {}

    def fake_main(argv):
        observed.update({name: getattr(campaign, name) for name in confirmation.OVERRIDES})
        return {"argv": argv}

    monkeypatch.setattr(campaign, "main", fake_main)
    returned = confirmation.main(["--no-telegram"])
    assert returned["argv"] == [
        "--no-telegram", "--state-dir", "/dev/shm/extent-exp072-state",
        "--qwen-cache-dir", "/dev/shm/qwen3-1.7b-exp072-weights",
    ]
    assert observed == confirmation.OVERRIDES
    assert {name: getattr(campaign, name) for name in confirmation.OVERRIDES} == original
