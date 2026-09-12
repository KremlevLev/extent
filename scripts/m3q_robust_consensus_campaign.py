"""EXP-084: paired-window robust dual-domain trust confirmation."""
from __future__ import annotations

from scripts import m3q_dual_domain_trust_campaign as campaign


OVERRIDES = {
    "PROTOCOL": "exp084-window-robust-dual-consensus-v1",
    "HF_PREFIX": "experiments/exp084-robust-consensus",
    "RESULT_STEM": "extent-m3q-robust-consensus",
    "SUMMARY_TITLE": "EXP-084 paired-window robust dual consensus",
    "OUTPUT_SUBDIR": "exp084",
    "STATE_PREFIX": "extent-exp084",
    "QWEN_CACHE_DIR": "/dev/shm/qwen3-1.7b-exp084-weights",
    "DERIVED_FROM": "EXP-083 mean dual-consensus seed-456 instability",
    "PRIMARY_ARM": "ROBUST-CONSENSUS",
    "CONTROL_ARM": "MEAN-CONSENSUS",
    "PRIMARY_SELECTION_MODE": "robust_consensus",
    "CONTROL_SELECTION_MODE": "dual_consensus",
    # EXP-083 consumed offsets [0, 557056) from WikiText-103 train.
    "BASE_TRAIN_OFFSET": 557_056,
    # Fresh PG-19 validation/test ranges beyond the first 65536 tokens.
    "BASE_SECONDARY_OFFSET": 65_536,
    "BASE_EVAL_OFFSET": 65_536,
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
