"""EXP-075: protect transplanted Qwen weights during joint Mamba recovery."""
from __future__ import annotations

from flax import traverse_util
from flax.core import FrozenDict, freeze

from scripts import m3q_sequential_joint_recovery_campaign as campaign


def trainable_mask(arm, params):
    flat = traverse_util.flatten_dict(params)
    values = {}
    for path in flat:
        in_mamba = "mamba" in path
        qwen_norm = path[-1] == "scale" and not in_mamba
        values[path] = bool(in_mamba or (arm == "MAMBA-NORMS" and qwen_norm))
    mask = traverse_util.unflatten_dict(values)
    return freeze(mask) if isinstance(params, FrozenDict) else mask


OVERRIDES = {
    "PROTOCOL": "exp075-protected-joint-recovery-v1",
    "HF_PREFIX": "experiments/exp075-protected-joint-recovery",
    "OUTPUT_SUBDIR": "exp075",
    "RESULT_STEM": "extent-m3q-protected-joint-recovery",
    "SUMMARY_TITLE": "EXP-075 protected joint recovery",
    "ARMS": ("MAMBA-ONLY", "MAMBA-NORMS"),
    "SOURCE_ARM_BY_ARM": {
        "MAMBA-ONLY": "ONPOLICY", "MAMBA-NORMS": "ONPOLICY"
    },
    "LR_BY_ARM": {"MAMBA-ONLY": 3e-6, "MAMBA-NORMS": 3e-6},
    "WARMUP_STEPS": 512,
    "CLIP_NORM": 0.3,
    "OPTIMIZER_DESCRIPTION": (
        "BF16 Lion; peak lr=3e-6; warmup=512; cosine end=3e-7; "
        "no decay; clip=0.3; static gradient mask by arm"
    ),
    "PRIMARY_ARM": "MAMBA-ONLY",
    "CONTROL_ARM": "MAMBA-NORMS",
    "AGGREGATE_MODE": "stability",
    "TRAINABLE_MASK_FACTORY": trainable_mask,
    "CONTRACT_EXTRA": {
        "registered_primary": "MAMBA-ONLY protected recovery; MAMBA-NORMS exploratory control",
        "source_arm_by_arm": {
            "MAMBA-ONLY": "ONPOLICY", "MAMBA-NORMS": "ONPOLICY"
        },
        "learning_rate_by_arm": {"MAMBA-ONLY": 3e-6, "MAMBA-NORMS": 3e-6},
        "warmup_steps": 512, "clip_norm": 0.3,
        "trainable": "static gradient mask; see trainable_by_arm",
        "trainable_by_arm": {
            "MAMBA-ONLY": "all and only Mamba subtree parameters",
            "MAMBA-NORMS": "Mamba subtrees plus copied Qwen normalization scales",
        },
        "causal_control": "EXP-074 LR3E-6 all-parameter arm",
    },
}


def main(argv=None):
    arguments = list(argv or ())
    defaults = {
        "--state-dir": "/dev/shm/extent-exp075-state",
        "--qwen-cache-dir": "/dev/shm/qwen3-1.7b-exp075-weights",
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
