"""EXP-077: protected AdamW recovery control after Lion instability."""
from __future__ import annotations

from extent.optimizer import create_adamw
from scripts import m3q_sequential_joint_recovery_campaign as campaign
from scripts.m3q_protected_joint_recovery_campaign import trainable_mask


def optimizer_factory(arm, learning_rate, warmup_steps, total_steps, clip_norm):
    del arm
    return create_adamw(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=0.0,
        max_grad_norm=clip_norm,
    )


OVERRIDES = {
    "PROTOCOL": "exp077-protected-adamw-recovery-v1",
    "HF_PREFIX": "experiments/exp077-adamw-recovery",
    "OUTPUT_SUBDIR": "exp077",
    "RESULT_STEM": "extent-m3q-adamw-recovery",
    "SUMMARY_TITLE": "EXP-077 protected AdamW recovery",
    "ARMS": ("ADAMW-3E-6", "ADAMW-1E-5"),
    "SOURCE_ARM_BY_ARM": {
        "ADAMW-3E-6": "ONPOLICY", "ADAMW-1E-5": "ONPOLICY"
    },
    "LR_BY_ARM": {"ADAMW-3E-6": 3e-6, "ADAMW-1E-5": 1e-5},
    "WARMUP_STEPS": 512,
    "CLIP_NORM": 0.3,
    "OPTIMIZER_DESCRIPTION": (
        "BF16 AdamW moments and gradients; beta1=0.9; beta2=0.999; "
        "warmup=512; cosine end=0.1x peak; no decay; clip=0.3"
    ),
    "PRIMARY_ARM": "ADAMW-3E-6",
    "CONTROL_ARM": "ADAMW-1E-5",
    "AGGREGATE_MODE": "stability",
    "TRAINABLE_MASK_FACTORY": trainable_mask,
    "TRAIN_STEP_FACTORY": None,
    "OPTIMIZER_FACTORY": optimizer_factory,
    "CONTRACT_EXTRA": {
        "registered_primary": (
            "ADAMW-3E-6 exact-LR optimizer control; ADAMW-1E-5 "
            "optimizer-calibrated exploratory arm"
        ),
        "source_arm_by_arm": {
            "ADAMW-3E-6": "ONPOLICY", "ADAMW-1E-5": "ONPOLICY"
        },
        "learning_rate_by_arm": {
            "ADAMW-3E-6": 3e-6, "ADAMW-1E-5": 1e-5
        },
        "warmup_steps": 512,
        "clip_norm": 0.3,
        "trainable": "all and only Mamba subtree parameters",
        "causal_control": (
            "EXP-075 MAMBA-ONLY uses the same source, mask, tokens, objective, "
            "updates, schedule shape and peak lr=3e-6 with Lion"
        ),
    },
}


def main(argv=None):
    arguments = list(argv or ())
    defaults = {
        "--state-dir": "/dev/shm/extent-exp077-state",
        "--qwen-cache-dir": "/dev/shm/qwen3-1.7b-exp077-weights",
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
