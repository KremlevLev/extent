"""EXP-078: protected per-tensor trust-ratio recovery."""
from __future__ import annotations

from extent.optimizer import create_lamb
from scripts import m3q_sequential_joint_recovery_campaign as campaign
from scripts.m3q_protected_joint_recovery_campaign import trainable_mask


def optimizer_factory(arm, learning_rate, warmup_steps, total_steps, clip_norm):
    del arm
    return create_lamb(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=0.0,
        max_grad_norm=clip_norm,
    )


OVERRIDES = {
    "PROTOCOL": "exp078-protected-trust-ratio-recovery-v1",
    "HF_PREFIX": "experiments/exp078-trust-ratio-recovery",
    "OUTPUT_SUBDIR": "exp078",
    "RESULT_STEM": "extent-m3q-trust-ratio-recovery",
    "SUMMARY_TITLE": "EXP-078 protected trust-ratio recovery",
    "ARMS": ("TRUST-1E-4", "TRUST-3E-5"),
    "SOURCE_ARM_BY_ARM": {
        "TRUST-1E-4": "ONPOLICY", "TRUST-3E-5": "ONPOLICY"
    },
    "LR_BY_ARM": {"TRUST-1E-4": 1e-4, "TRUST-3E-5": 3e-5},
    "WARMUP_STEPS": 512,
    "CLIP_NORM": 0.3,
    "OPTIMIZER_DESCRIPTION": (
        "BF16 LAMB; beta1=0.9; beta2=0.999; per-leaf trust ratio; "
        "min_norm=1e-6; warmup=512; cosine end=0.1x peak; no decay; clip=0.3"
    ),
    "PRIMARY_ARM": "TRUST-1E-4",
    "CONTROL_ARM": "TRUST-3E-5",
    "AGGREGATE_MODE": "stability",
    "TRAINABLE_MASK_FACTORY": trainable_mask,
    "TRAIN_STEP_FACTORY": None,
    "OPTIMIZER_FACTORY": optimizer_factory,
    "CONTRACT_EXTRA": {
        "registered_primary": (
            "TRUST-1E-4 per-tensor relative-step recovery; TRUST-3E-5 "
            "conservative trust-ratio control"
        ),
        "source_arm_by_arm": {
            "TRUST-1E-4": "ONPOLICY", "TRUST-3E-5": "ONPOLICY"
        },
        "learning_rate_by_arm": {
            "TRUST-1E-4": 1e-4, "TRUST-3E-5": 3e-5
        },
        "warmup_steps": 512,
        "clip_norm": 0.3,
        "trainable": "all and only Mamba subtree parameters",
        "trust_ratio": (
            "each parameter leaf's Adam direction is rescaled to its parameter "
            "norm before the scheduled scalar step"
        ),
        "historical_controls": (
            "EXP-077 AdamW and EXP-075 Lion use the same source, mask, tokens, "
            "prediction objective and update count"
        ),
    },
}


def main(argv=None):
    arguments = list(argv or ())
    defaults = {
        "--state-dir": "/dev/shm/extent-exp078-state",
        "--qwen-cache-dir": "/dev/shm/qwen3-1.7b-exp078-weights",
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
