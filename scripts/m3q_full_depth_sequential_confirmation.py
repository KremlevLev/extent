"""EXP-072: fresh-data, long-budget sequential recovery over all 24 Mamba layers."""
from __future__ import annotations

from scripts import m3q_sequential_recovery_campaign as campaign


OVERRIDES = {
    "PROTOCOL": "exp072-full-depth-sequential-confirmation-v2",
    "HF_PREFIX": "experiments/exp072-full-depth-sequential-confirmation-v2",
    "OUTPUT_SUBDIR": "exp072-v2",
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
    # The pinned Qwen tokenizer yields 2,540,999 tokens for WikiText-2 train.
    # This slice is fresh relative to EXP-071 and leaves a safety margin at EOF.
    "TRAIN_OFFSET": 2_424_832,
    "EVAL_OFFSET": 98_304,
}

PINNED_TRAIN_TOKEN_CAPACITY = 2_540_999


def validate_data_ranges():
    required_end = OVERRIDES["TRAIN_OFFSET"] + (
        OVERRIDES["TRAIN_WINDOWS"] * OVERRIDES["TRAIN_LENGTH"]
    )
    if required_end > PINNED_TRAIN_TOKEN_CAPACITY:
        raise ValueError(
            "EXP-072 train slice exceeds the observed pinned WikiText/Qwen "
            f"token capacity: {required_end} > {PINNED_TRAIN_TOKEN_CAPACITY}"
        )
    return required_end


def configured_contract(config):
    """Build the frozen EXP-072 contract without leaking module overrides."""
    original = {name: getattr(campaign, name) for name in OVERRIDES}
    try:
        for name, value in OVERRIDES.items():
            setattr(campaign, name, value)
        return campaign.experiment_contract(config)
    finally:
        for name, value in original.items():
            setattr(campaign, name, value)


def main(argv=None):
    validate_data_ranges()
    arguments = list(argv or ())
    defaults = {
        "--state-dir": "/dev/shm/extent-exp072-v2-state",
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
