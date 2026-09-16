import jax
import jax.numpy as jnp
import optax
from dataclasses import replace

from extent.config import tiny_config
from extent.model import HybridDecoderLayer
from extent.sequential_recovery import replace_mamba

from extent.block_recovery import make_block_recovery_step
from scripts.m3q_joint_pair_campaign import configured_contract
from scripts import m3q_joint_pair_campaign as joint_campaign
from scripts import m3q_dual_domain_trust_campaign as base_campaign


def test_joint_step_updates_both_composed_parameters_under_jit():
    def forward(params, frozen, inputs):
        return inputs * params["first"] * params["second"] + frozen

    tx = optax.sgd(0.01)
    params = {"first": jnp.array(1.0), "second": jnp.array(2.0)}
    step = jax.jit(make_block_recovery_step(forward, tx, bf16_gradients=False))
    updated, _, metrics = step(
        params, tx.init(params), jnp.array(0.0), jnp.ones((1, 2, 3)),
        jnp.ones((1, 2, 3)) * 4,
    )
    assert updated["first"] > params["first"]
    assert updated["second"] > params["second"]
    assert jnp.isfinite(metrics["loss"])


def test_exp087_changes_proposal_not_selector():
    contract = configured_contract()
    extension = contract["interaction_aware_extension"]
    assert extension["group_proposal_mode"] == {
        "INDEPENDENT-PAIR": "independent", "JOINT-PAIR": "joint_segment",
    }
    assert extension["primary_group_selection_mode"] == "pair_consensus"
    assert extension["control_group_selection_mode"] == "pair_consensus"
    assert contract["proposal_and_primary_calibration"]["proposal_offsets"][0] == 1_671_168


def test_joint_step_compiles_real_two_mamba_decoder_segment():
    cfg = replace(tiny_config(), attention_layer_indices=(1,),
                  compute_dtype="float32", param_dtype="float32")
    layer = HybridDecoderLayer(cfg, 0)
    inputs = jax.random.normal(jax.random.key(8), (1, 3, cfg.hidden_size))
    positions = jnp.arange(3)[None]
    frozen = {
        str(index): layer.init(jax.random.key(index), inputs, positions, None)["params"]
        for index in (0, 1)
    }
    params = {key: value["mamba"] for key, value in frozen.items()}

    def forward(candidate, frozen, inputs):
        value = inputs
        for key in ("0", "1"):
            value = layer.apply(
                {"params": replace_mamba(frozen[key], candidate[key])},
                value, positions, None,
            )
        return value

    target = forward(params, frozen, inputs) + 0.01
    tx = optax.sgd(1e-4)
    updated, _, metrics = jax.jit(make_block_recovery_step(
        forward, tx, bf16_gradients=False,
    ))(params, tx.init(params), frozen, inputs, target)
    assert bool(metrics["grads_finite"])
    for key in params:
        assert any(bool(jnp.any(a != b)) for a, b in zip(
            jax.tree.leaves(params[key]), jax.tree.leaves(updated[key]), strict=True,
        ))


def test_joint_campaign_runtime_wrapper_builds_replication_without_arm_key_errors(monkeypatch):
    monkeypatch.setattr(base_campaign, "main", lambda argv: base_campaign.replication_overrides(0))
    result = joint_campaign.main([])
    assert result["GROUP_PROPOSAL_MODE"] == {
        "INDEPENDENT-PAIR": "independent", "JOINT-PAIR": "joint_segment",
    }
    assert result["ARM_MIN_RELATIVE_GAIN"] == {
        "INDEPENDENT-PAIR": 0.001, "JOINT-PAIR": 0.001,
    }
