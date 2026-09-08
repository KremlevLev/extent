"""EXP-072: fresh-data, long-budget sequential recovery over all 24 Mamba layers."""
from __future__ import annotations

from scripts import m3q_sequential_recovery_campaign as campaign


OVERRIDES = {
    "PROTOCOL": "exp072-full-depth-sequential-confirmation-v1",
    "HF_PREFIX": "experiments/exp072-full-depth-sequential-confirmation",
    "OUTPUT_SUBDIR": "exp072",
    "RESULT_STEM": "extent-m3q-full-depth-sequential-confirmation",
    "SUMMARY_TITLE": "EXP-072 full-depth sequential recovery confirmation",
    "PRIMARY_ARM": "ONPOLICY",
    "CONTRACT_EXTRA": {
        "registered_primary_arm": "ONPOLICY",
        "confirmation_of": "EXP-071",
        "data_seed": 72_000,
    },
    "DATA_SEED": 72_000,
    "MILESTONES": (1, 4, 8, 12, 16, 20, 24),
    "STEPS_PER_LAYER": 4096,
    "TRAIN_WINDOWS": 512,
    "TRAIN_LENGTH": 128,
    "EVAL_WINDOWS": 32,
    "EVAL_LENGTH": 256,
    "TRAIN_OFFSET": 2_621_440,
    "EVAL_OFFSET": 98_304,
}


def main(argv=None):
    arguments = list(argv or ())
    defaults = {
        "--state-dir": "/dev/shm/extent-exp072-state",
        "--qwen-cache-dir": "/dev/shm/qwen3-1.7b-exp072-weights",
    }
    for option, value in defaults.items():
        if option not in arguments:
            arguments.extend((option, value))
    original = {name: getattr(campaign, name) for name in OVERRIDES}
    try:
        for name, value in OVERRIDES.items():
            setattr(campaign, name, value)
        return campaign.main(arguments)
    finally:
        for name, value in original.items():
            setattr(campaign, name, value)


if __name__ == "__main__":
    main()
