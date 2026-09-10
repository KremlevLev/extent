import copy

import jax
import jax.numpy as jnp

from extent.config import tiny_config
from scripts.m3q_full_depth_sequential_confirmation import configured_contract
from scripts.m3q_sequential_joint_recovery_campaign import (
    ARMS, CHECKPOINTS, SEEDS, aggregate, detach_donated_tree,
    experiment_contract,
)
from scripts import m3q_sequential_joint_recovery_campaign as joint
from scripts import m3q_stable_joint_recovery_campaign as stable
from scripts import m3q_protected_joint_recovery_campaign as protected
from scripts import m3q_objective_bridge_campaign as objective_bridge
from scripts import m3q_adamw_recovery_campaign as adamw_recovery
from scripts import m3q_trust_ratio_recovery_campaign as trust_recovery


def _row(values):
    return {"complete": True, "evaluations": {
        str(step): {"student_nll": value} for step, value in zip(CHECKPOINTS, values)
    }}


def test_exp072_contract_helper_does_not_leak_overrides():
    first = configured_contract(tiny_config())
    second = configured_contract(tiny_config())
    assert first == second
    assert first["protocol"] == "exp072-full-depth-sequential-confirmation-v2"


def test_joint_recovery_gate_requires_both_seed_final_and_auc_wins():
    result = {"arms": {}}
    teacher = [8.0, 7.8, 7.5, 7.0, 6.8, 6.5]
    onpolicy = [7.0, 6.9, 6.7, 6.4, 6.1, 5.9]
    for seed in SEEDS:
        result["arms"][str(seed)] = {
            "TEACHER": _row(teacher), "ONPOLICY": _row(onpolicy)
        }
    assert aggregate(result)["scientific_gate_passed"]
    incomplete = copy.deepcopy(result)
    incomplete["arms"][str(SEEDS[-1])][ARMS[-1]]["complete"] = False
    assert not aggregate(incomplete)["scientific_gate_passed"]


def test_joint_contract_records_equal_update_comparison():
    contract = experiment_contract(tiny_config())
    assert contract["arms"] == ["TEACHER", "ONPOLICY"]
    assert contract["total_steps"] == 8192
    assert contract["trainable"] == "all student parameters"


def test_detached_student_tree_preserves_values_and_has_distinct_buffer():
    source = {"w": jnp.arange(8, dtype=jnp.float32)}
    layout = {"w": source["w"].sharding}
    detached = detach_donated_tree(source, layout)
    assert jnp.array_equal(detached["w"], source["w"])
    assert detached["w"].unsafe_buffer_pointer() != source["w"].unsafe_buffer_pointer()


def test_stability_gate_requires_final_improvement_and_bounded_curve(monkeypatch):
    original = {name: getattr(joint, name) for name in stable.OVERRIDES}
    try:
        for name, value in stable.OVERRIDES.items():
            monkeypatch.setattr(joint, name, value)
        result = {"arms": {}}
        for seed in SEEDS:
            result["arms"][str(seed)] = {
                "LR3E-6": _row([10.0, 10.5, 9.8, 9.4, 9.2, 9.0]),
                "LR1E-6": _row([10.0, 10.1, 9.9, 9.8, 9.7, 9.6]),
            }
        assert joint.aggregate(result)["scientific_gate_passed"]
        result["arms"][str(SEEDS[-1])]["LR3E-6"] = _row(
            [10.0, 13.0, 9.8, 9.4, 9.2, 9.0]
        )
        assert not joint.aggregate(result)["scientific_gate_passed"]
    finally:
        for name, value in original.items():
            setattr(joint, name, value)


def test_stable_wrapper_applies_and_restores_overrides(monkeypatch):
    original = {name: getattr(joint, name) for name in stable.OVERRIDES}
    observed = {}

    def fake_main(argv):
        observed.update({name: getattr(joint, name) for name in stable.OVERRIDES})
        return {"argv": argv}

    monkeypatch.setattr(joint, "main", fake_main)
    result = stable.main(["--no-telegram"])
    assert result["argv"] == [
        "--no-telegram", "--state-dir", "/dev/shm/extent-exp074-state",
        "--qwen-cache-dir", "/dev/shm/qwen3-1.7b-exp074-weights",
    ]
    assert observed == stable.OVERRIDES
    assert {name: getattr(joint, name) for name in stable.OVERRIDES} == original


def test_protected_mask_freezes_copied_qwen_weights():
    params = {
        "layers_0": {
            "mamba": {"in_proj": {"kernel": jnp.ones((2, 2))}},
            "input_layernorm": {"scale": jnp.ones((2,))},
            "mlp": {"up_proj": {"kernel": jnp.ones((2, 2))}},
        },
        "final_norm": {"scale": jnp.ones((2,))},
    }
    mamba_only = protected.trainable_mask("MAMBA-ONLY", params)
    with_norms = protected.trainable_mask("MAMBA-NORMS", params)
    assert mamba_only["layers_0"]["mamba"]["in_proj"]["kernel"]
    assert not mamba_only["layers_0"]["input_layernorm"]["scale"]
    assert not mamba_only["layers_0"]["mlp"]["up_proj"]["kernel"]
    assert with_norms["layers_0"]["input_layernorm"]["scale"]
    assert with_norms["final_norm"]["scale"]
    assert not with_norms["layers_0"]["mlp"]["up_proj"]["kernel"]


def test_protected_wrapper_applies_and_restores_overrides(monkeypatch):
    original = {name: getattr(joint, name) for name in protected.OVERRIDES}
    observed = {}

    def fake_main(argv):
        observed.update({name: getattr(joint, name) for name in protected.OVERRIDES})
        return {"argv": argv}

    monkeypatch.setattr(joint, "main", fake_main)
    result = protected.main(["--no-telegram"])
    assert result["argv"] == [
        "--no-telegram", "--state-dir", "/dev/shm/extent-exp075-state",
        "--qwen-cache-dir", "/dev/shm/qwen3-1.7b-exp075-weights",
    ]
    assert observed == protected.OVERRIDES
    assert {name: getattr(joint, name) for name in protected.OVERRIDES} == original


def test_objective_bridge_factory_selects_registered_hidden_modes(monkeypatch):
    calls = []

    def fake_factory(*args, **kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(objective_bridge, "make_hidden_bridge_distill_step", fake_factory)
    config = tiny_config()
    mask = {"x": True}
    delta = objective_bridge.train_step_factory(
        "DELTA-BRIDGE", object(), object(), object(), config, mask
    )
    state = objective_bridge.train_step_factory(
        "STATE-BRIDGE", object(), object(), object(), config, mask
    )
    assert delta["hidden_mode"] == "delta"
    assert state["hidden_mode"] == "state"
    assert delta["layer_indices"] == config.mamba_layer_indices
    assert delta["prediction_weight"] == 0.0
    assert delta["cross_entropy_weight"] == 0.0
    assert delta["trainable_mask"] is mask
    assert len(calls) == 2


def test_objective_bridge_wrapper_applies_and_restores_overrides(monkeypatch):
    original = {name: getattr(joint, name) for name in objective_bridge.OVERRIDES}
    observed = {}

    def fake_main(argv):
        observed.update({
            name: getattr(joint, name) for name in objective_bridge.OVERRIDES
        })
        return {"argv": argv}

    monkeypatch.setattr(joint, "main", fake_main)
    result = objective_bridge.main(["--no-telegram"])
    assert result["argv"] == [
        "--no-telegram", "--state-dir", "/dev/shm/extent-exp076-state",
        "--qwen-cache-dir", "/dev/shm/qwen3-1.7b-exp076-weights",
    ]
    assert observed == objective_bridge.OVERRIDES
    assert {
        name: getattr(joint, name) for name in objective_bridge.OVERRIDES
    } == original


def test_adamw_recovery_factory_and_wrapper(monkeypatch):
    factory_calls = []

    def fake_optimizer(**kwargs):
        factory_calls.append(kwargs)
        return "adamw"

    monkeypatch.setattr(adamw_recovery, "create_adamw", fake_optimizer)
    assert adamw_recovery.optimizer_factory(
        "ADAMW-3E-6", 3e-6, 512, 8192, 0.3
    ) == "adamw"
    assert factory_calls == [{
        "learning_rate": 3e-6,
        "warmup_steps": 512,
        "total_steps": 8192,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
    }]

    original = {name: getattr(joint, name) for name in adamw_recovery.OVERRIDES}
    observed = {}

    def fake_main(argv):
        observed.update({
            name: getattr(joint, name) for name in adamw_recovery.OVERRIDES
        })
        return {"argv": argv}

    monkeypatch.setattr(joint, "main", fake_main)
    result = adamw_recovery.main(["--no-telegram"])
    assert result["argv"] == [
        "--no-telegram", "--state-dir", "/dev/shm/extent-exp077-state",
        "--qwen-cache-dir", "/dev/shm/qwen3-1.7b-exp077-weights",
    ]
    assert observed == adamw_recovery.OVERRIDES
    assert {
        name: getattr(joint, name) for name in adamw_recovery.OVERRIDES
    } == original


def test_trust_recovery_factory_and_wrapper(monkeypatch):
    factory_calls = []

    def fake_optimizer(**kwargs):
        factory_calls.append(kwargs)
        return "lamb"

    monkeypatch.setattr(trust_recovery, "create_lamb", fake_optimizer)
    assert trust_recovery.optimizer_factory(
        "TRUST-1E-4", 1e-4, 512, 8192, 0.3
    ) == "lamb"
    assert factory_calls == [{
        "learning_rate": 1e-4,
        "warmup_steps": 512,
        "total_steps": 8192,
        "weight_decay": 0.0,
        "max_grad_norm": 0.3,
    }]

    original = {name: getattr(joint, name) for name in trust_recovery.OVERRIDES}
    observed = {}

    def fake_main(argv):
        observed.update({
            name: getattr(joint, name) for name in trust_recovery.OVERRIDES
        })
        return {"argv": argv}

    monkeypatch.setattr(joint, "main", fake_main)
    result = trust_recovery.main(["--no-telegram"])
    assert result["argv"] == [
        "--no-telegram", "--state-dir", "/dev/shm/extent-exp078-state",
        "--qwen-cache-dir", "/dev/shm/qwen3-1.7b-exp078-weights",
    ]
    assert observed == trust_recovery.OVERRIDES
    assert {
        name: getattr(joint, name) for name in trust_recovery.OVERRIDES
    } == original
