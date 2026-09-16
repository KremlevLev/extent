"""EXP-087: jointly trained composed-pair proposals versus independent proposals."""
from scripts import m3q_pair_threshold_campaign as wrapper


OVERRIDES = dict(wrapper.OVERRIDES, **{
    "PROTOCOL": "exp087-joint-pair-recovery-v1",
    "HF_PREFIX": "experiments/exp087-joint-pair",
    "RESULT_STEM": "extent-m3q-joint-pair",
    "SUMMARY_TITLE": "EXP-087 composed joint-pair recovery",
    "OUTPUT_SUBDIR": "exp087",
    "STATE_PREFIX": "extent-exp087",
    "QWEN_CACHE_DIR": "/dev/shm/qwen3-1.7b-exp087-weights",
    "DERIVED_FROM": "EXP-085 interactions without cross-seed selection superiority",
    "PRIMARY_ARM": "JOINT-PAIR",
    "CONTROL_ARM": "INDEPENDENT-PAIR",
    "ARM_MIN_RELATIVE_GAIN": {"JOINT-PAIR": 0.001, "INDEPENDENT-PAIR": 0.001},
    "PRIMARY_PROPOSAL_MODE": "joint_segment",
    "CONTROL_PROPOSAL_MODE": "independent",
    "INNER_PROTOCOL_PREFIX": "exp087-joint-pair-replication",
    "INNER_RESULT_PREFIX": "exp087-joint-pair-replication",
    "INNER_OUTPUT_PREFIX": "exp087/replication",
    "BASE_DATA_SEED": 87_000,
    "BASE_TRAIN_OFFSET": 1_671_168,
    "BASE_SECONDARY_OFFSET": 131_072,
    "BASE_EVAL_OFFSET": 196_608,
})


def configured_contract():
    original = wrapper.OVERRIDES
    try:
        wrapper.OVERRIDES = OVERRIDES
        return wrapper.configured_contract()
    finally:
        wrapper.OVERRIDES = original


def main(argv=None):
    original = wrapper.OVERRIDES
    try:
        wrapper.OVERRIDES = OVERRIDES
        return wrapper.main(argv)
    finally:
        wrapper.OVERRIDES = original


if __name__ == "__main__":
    main()
