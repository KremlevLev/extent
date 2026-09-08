import copy

from extent.config import tiny_config
from scripts.m3q_full_depth_sequential_confirmation import configured_contract
from scripts.m3q_sequential_joint_recovery_campaign import (
    ARMS, CHECKPOINTS, SEEDS, aggregate, experiment_contract,
)


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
