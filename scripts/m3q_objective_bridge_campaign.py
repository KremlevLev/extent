"""EXP-076: protected state-versus-contribution bridge recovery."""
from __future__ import annotations

from extent.full_model_distillation import make_hidden_bridge_distill_step
from scripts import m3q_sequential_joint_recovery_campaign as campaign
from scripts.m3q_protected_joint_recovery_campaign import trainable_mask


def train_step_factory(
    arm, student_apply, teacher_apply, tx, config, static_trainable_mask
):
    mode = {"STATE-BRIDGE": "state", "DELTA-BRIDGE": "delta"}[arm]
    return make_hidden_bridge_distill_step(
        student_apply,
        teacher_apply,
        tx,
        layer_indices=config.mamba_layer_indices,
        temperature=2.0,
        cross_entropy_weight=0.0,
        prediction_weight=0.0,
        hidden_weight=1.0,
        bf16_gradients=True,
        hidden_mode=mode,
        trainable_mask=static_trainable_mask,
    )


OVERRIDES = {
    "PROTOCOL": "exp076-protected-objective-bridge-v1",
    "HF_PREFIX": "experiments/exp076-objective-bridge",
    "OUTPUT_SUBDIR": "exp076",
    "RESULT_STEM": "extent-m3q-objective-bridge",
    "SUMMARY_TITLE": "EXP-076 protected objective bridge",
    "ARMS": ("DELTA-BRIDGE", "STATE-BRIDGE"),
    "SOURCE_ARM_BY_ARM": {
        "DELTA-BRIDGE": "ONPOLICY", "STATE-BRIDGE": "ONPOLICY"
    },
    "LR_BY_ARM": {"DELTA-BRIDGE": 3e-6, "STATE-BRIDGE": 3e-6},
    "WARMUP_STEPS": 512,
    "CLIP_NORM": 0.3,
    "OPTIMIZER_DESCRIPTION": (
        "BF16 Lion; peak lr=3e-6; warmup=512; cosine end=3e-7; "
        "no decay; clip=0.3; Mamba-only static gradient mask"
    ),
    "PRIMARY_ARM": "DELTA-BRIDGE",
    "CONTROL_ARM": "STATE-BRIDGE",
    "AGGREGATE_MODE": "stability",
    "TRAINABLE_MASK_FACTORY": trainable_mask,
    "TRAIN_STEP_FACTORY": train_step_factory,
    "CONTRACT_EXTRA": {
        "objective": (
            "Mamba-layer hidden alignment only; DELTA-BRIDGE matches each "
            "decoder block contribution h_l-h_(l-1), STATE-BRIDGE matches "
            "accumulated h_l; no output KL or causal cross entropy"
        ),
        "registered_primary": (
            "DELTA-BRIDGE protected recovery; STATE-BRIDGE objective control"
        ),
        "source_arm_by_arm": {
            "DELTA-BRIDGE": "ONPOLICY", "STATE-BRIDGE": "ONPOLICY"
        },
        "learning_rate_by_arm": {
            "DELTA-BRIDGE": 3e-6, "STATE-BRIDGE": 3e-6
        },
        "warmup_steps": 512,
        "clip_norm": 0.3,
        "trainable": "all and only Mamba subtree parameters",
        "hidden_layers": "all 24 replaced Mamba decoder positions",
        "causal_control": (
            "EXP-075 MAMBA-ONLY uses the same start, mask, data, optimizer and "
            "update count with output KL plus 0.1 causal cross entropy"
        ),
    },
}


def main(argv=None):
    arguments = list(argv or ())
    defaults = {
        "--state-dir": "/dev/shm/extent-exp076-state",
        "--qwen-cache-dir": "/dev/shm/qwen3-1.7b-exp076-weights",
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
