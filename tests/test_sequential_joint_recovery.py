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
