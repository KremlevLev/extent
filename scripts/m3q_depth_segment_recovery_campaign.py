"""EXP-079: protected recovery atlas over four attention-bounded Mamba segments."""
from __future__ import annotations

from flax import traverse_util
from flax.core import FrozenDict, freeze

from extent.optimizer import create_lamb
from scripts import m3q_sequential_joint_recovery_campaign as campaign


SEGMENTS = {
    "SEGMENT-1": (0, 1, 2, 3, 4, 5),
    "SEGMENT-2": (7, 8, 9, 10, 11, 12),
    "SEGMENT-3": (14, 15, 16, 17, 18, 19),
    "SEGMENT-4": (21, 22, 23, 24, 25, 26),
}


def trainable_mask(arm, params):
    selected = set(SEGMENTS[arm])
    flat = traverse_util.flatten_dict(params)
    values = {}
    for path in flat:
        layer = None
        if path and path[0].startswith("layers_"):
            layer = int(path[0].split("_", 1)[1])
        values[path] = bool(layer in selected and "mamba" in path)
    mask = traverse_util.unflatten_dict(values)
    return freeze(mask) if isinstance(params, FrozenDict) else mask


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
    "PROTOCOL": "exp079-attention-bounded-depth-segment-recovery-v1",
    "HF_PREFIX": "experiments/exp079-depth-segment-recovery",
    "OUTPUT_SUBDIR": "exp079",
    "RESULT_STEM": "extent-m3q-depth-segment-recovery",
    "SUMMARY_TITLE": "EXP-079 attention-bounded depth segment recovery",
    "ARMS": tuple(SEGMENTS),
    "SOURCE_ARM_BY_ARM": {arm: "ONPOLICY" for arm in SEGMENTS},
    "LR_BY_ARM": {arm: 3e-5 for arm in SEGMENTS},
    "WARMUP_STEPS": 384,
    "CLIP_NORM": 0.3,
    "OPTIMIZER_DESCRIPTION": (
        "BF16 LAMB; peak relative step=3e-5; beta1=0.9; beta2=0.999; "
        "per-leaf trust ratio; warmup=384; cosine end=3e-6; no decay; clip=0.3"
    ),
    "PRIMARY_ARM": "SEGMENT-4",
    "CONTROL_ARM": "SEGMENT-1",
    "AGGREGATE_MODE": "stability",
    "TRAINABLE_MASK_FACTORY": trainable_mask,
    "TRAIN_STEP_FACTORY": None,
    "OPTIMIZER_FACTORY": optimizer_factory,
    "TOTAL_STEPS": 6144,
    "CHECKPOINTS": (0, 1024, 2048, 4096, 6144),
    "CONTRACT_EXTRA": {
        "registered_primary": (
            "SEGMENT-4, the deepest Mamba run bounded by retained attention; "
            "segments 1-3 are a pre-registered depth atlas"
        ),
        "source_arm_by_arm": {arm: "ONPOLICY" for arm in SEGMENTS},
        "learning_rate_by_arm": {arm: 3e-5 for arm in SEGMENTS},
        "warmup_steps": 384,
        "clip_norm": 0.3,
        "trainable": "exactly one six-layer Mamba segment per arm",
        "trainable_layers_by_arm": {
            arm: list(layers) for arm, layers in SEGMENTS.items()
        },
        "segment_boundaries": (
            "the four six-Mamba runs preceding retained GQA layers 6/13/20/27"
        ),
        "selection_source": (
            "EXP-078 exploratory TRUST-3E-5 is the only full-depth recipe to "
            "cross below step zero for both seeds at any fixed checkpoint"
        ),
    },
}


def main(argv=None):
    arguments = list(argv or ())
    defaults = {
        "--state-dir": "/dev/shm/extent-exp079-state",
        "--qwen-cache-dir": "/dev/shm/qwen3-1.7b-exp079-weights",
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
