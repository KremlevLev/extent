"""EXP-085: interaction-aware paired-coordinate lookahead recovery."""
from __future__ import annotations

from scripts import m3q_dual_domain_trust_campaign as campaign


OVERRIDES = {
    "PROTOCOL": "exp085-paired-lookahead-recovery-v1",
    "HF_PREFIX": "experiments/exp085-paired-lookahead",
    "RESULT_STEM": "extent-m3q-paired-lookahead",
    "SUMMARY_TITLE": "EXP-085 interaction-aware paired lookahead recovery",
    "OUTPUT_SUBDIR": "exp085",
    "STATE_PREFIX": "extent-exp085",
    "QWEN_CACHE_DIR": "/dev/shm/qwen3-1.7b-exp085-weights",
    "DERIVED_FROM": "EXP-082 through EXP-084 scalar coordinate-selection failure",
    "REPLICATIONS": tuple(range(4)),
    "PRIMARY_ARM": "PAIR-LOOKAHEAD",
    "CONTROL_ARM": "GREEDY-PAIR-GRID",
    "PRIMARY_SELECTION_MODE": "dual_consensus",
    "CONTROL_SELECTION_MODE": "dual_consensus",
    "DECISION_GROUP_SIZE": 2,
    "PRIMARY_GROUP_SELECTION_MODE": "pair_consensus",
    "CONTROL_GROUP_SELECTION_MODE": "greedy_pair_grid",
    "REQUIRED_NEGATIVE_REPLICATIONS": 3,
    "REQUIRED_NONZERO_REPLICATIONS": 3,
    "REQUIRED_JOINT_RESCUE_REPLICATIONS": 1,
    "INNER_MAX_WALL_HOURS": 2.0,
    "MINIMUM_START_HEADROOM_MINUTES": 130,
    "INNER_PROTOCOL_PREFIX": "exp085-paired-lookahead-replication",
    "INNER_RESULT_PREFIX": "exp085-paired-lookahead-replication",
    "INNER_OUTPUT_PREFIX": "exp085/replication",
    "BASE_DATA_SEED": 85_000,
    # Fresh ranges immediately after the complete planned EXP-084 ranges.
    "BASE_TRAIN_OFFSET": 1_114_112,
    "BASE_SECONDARY_OFFSET": 98_304,
    "BASE_EVAL_OFFSET": 131_072,
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
