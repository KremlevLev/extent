"""EXP-086: conservative paired-lookahead acceptance threshold campaign."""
from __future__ import annotations

from scripts import m3q_dual_domain_trust_campaign as campaign


CONTROL_ARM = "STANDARD-PAIR-0.1PCT"
PRIMARY_ARM = "STRICT-PAIR-0.5PCT"

OVERRIDES = {
    "PROTOCOL": "exp086-pair-threshold-recovery-v1",
    "HF_PREFIX": "experiments/exp086-pair-threshold",
    "RESULT_STEM": "extent-m3q-pair-threshold",
    "SUMMARY_TITLE": "EXP-086 conservative pair threshold recovery",
    "OUTPUT_SUBDIR": "exp086",
    "STATE_PREFIX": "extent-exp086",
    "QWEN_CACHE_DIR": "/dev/shm/qwen3-1.7b-exp086-weights",
    "DERIVED_FROM": "EXP-085 replication-0 seed-dependent over-acceptance",
    "REPLICATIONS": tuple(range(4)),
    "PRIMARY_ARM": PRIMARY_ARM,
    "CONTROL_ARM": CONTROL_ARM,
    "PRIMARY_SELECTION_MODE": "dual_consensus",
    "CONTROL_SELECTION_MODE": "dual_consensus",
    "ARM_MIN_RELATIVE_GAIN": {
        CONTROL_ARM: 0.001,
        PRIMARY_ARM: 0.005,
    },
    "DECISION_GROUP_SIZE": 2,
    "PRIMARY_GROUP_SELECTION_MODE": "pair_consensus",
    "CONTROL_GROUP_SELECTION_MODE": "pair_consensus",
    "REQUIRED_NEGATIVE_REPLICATIONS": 3,
    "REQUIRED_NONZERO_REPLICATIONS": 3,
    "REQUIRED_JOINT_RESCUE_REPLICATIONS": 0,
    "INNER_MAX_WALL_HOURS": 3.25,
    "MINIMUM_START_HEADROOM_MINUTES": 210,
    "INNER_PROTOCOL_PREFIX": "exp086-pair-threshold-replication",
    "INNER_RESULT_PREFIX": "exp086-pair-threshold-replication",
    "INNER_OUTPUT_PREFIX": "exp086/replication",
    "BASE_DATA_SEED": 86_000,
    # Disjoint ranges immediately after all four planned EXP-085 replications.
    "BASE_TRAIN_OFFSET": 1_392_640,
    "BASE_SECONDARY_OFFSET": 114_688,
    "BASE_EVAL_OFFSET": 163_840,
}


def configured_contract():
    original = {name: getattr(campaign, name) for name in OVERRIDES}
    try:
        for name, value in OVERRIDES.items():
            setattr(campaign, name, value)
        return campaign.experiment_contract()
    finally:
        for name, value in original.items():
            setattr(campaign, name, value)


def configured_aggregate(result):
    original = {name: getattr(campaign, name) for name in OVERRIDES}
    try:
        for name, value in OVERRIDES.items():
            setattr(campaign, name, value)
        return campaign.aggregate(result)
    finally:
        for name, value in original.items():
            setattr(campaign, name, value)


def main(argv=None):
    original = {name: getattr(campaign, name) for name in OVERRIDES}
    try:
        for name, value in OVERRIDES.items():
            setattr(campaign, name, value)
        return campaign.main(argv)
    finally:
        for name, value in original.items():
            setattr(campaign, name, value)


if __name__ == "__main__":
    main()
