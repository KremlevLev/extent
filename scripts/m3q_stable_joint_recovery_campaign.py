"""EXP-074: locked low-LR stability comparison from EXP-072 ONPOLICY starts."""
from __future__ import annotations

from scripts import m3q_sequential_joint_recovery_campaign as campaign


OVERRIDES = {
    "PROTOCOL": "exp074-stable-joint-recovery-v1",
    "HF_PREFIX": "experiments/exp074-stable-joint-recovery",
    "OUTPUT_SUBDIR": "exp074",
    "RESULT_STEM": "extent-m3q-stable-joint-recovery",
    "SUMMARY_TITLE": "EXP-074 stabilized joint recovery",
    "ARMS": ("LR3E-6", "LR1E-6"),
    "SOURCE_ARM_BY_ARM": {"LR3E-6": "ONPOLICY", "LR1E-6": "ONPOLICY"},
    "LR_BY_ARM": {"LR3E-6": 3e-6, "LR1E-6": 1e-6},
    "WARMUP_STEPS": 512,
    "CLIP_NORM": 0.3,
    "OPTIMIZER_DESCRIPTION": (
        "BF16 Lion; arm peak LR in learning_rate_by_arm; warmup=512; "
        "cosine end=0.1x peak; no decay; clip=0.3"
    ),
    "PRIMARY_ARM": "LR3E-6",
    "CONTROL_ARM": "LR1E-6",
    "AGGREGATE_MODE": "stability",
    "CONTRACT_EXTRA": {
        "registered_primary": "LR3E-6 stability recipe; LR1E-6 exploratory conservative control",
        "source_arm_by_arm": {"LR3E-6": "ONPOLICY", "LR1E-6": "ONPOLICY"},
        "learning_rate_by_arm": {"LR3E-6": 3e-6, "LR1E-6": 1e-6},
        "warmup_steps": 512,
        "clip_norm": 0.3,
        "confirmation_baseline": "EXP-073 constant-3e-5 ONPOLICY trajectories",
    },
}


def main(argv=None):
    arguments = list(argv or ())
    defaults = {
        "--state-dir": "/dev/shm/extent-exp074-state",
        "--qwen-cache-dir": "/dev/shm/qwen3-1.7b-exp074-weights",
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
