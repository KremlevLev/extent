"""EXP-101: registered second-stage late-distillation."""
from scripts.m3q_subspace_engine import Arm, Campaign, Schedule
from scripts.m3q_plateau_campaign import run

SPEC = Campaign(101, "late-distillation", (
    Arm("CE", "INOUT-LORA", 32, "ce", protected=True),
    Arm("QWEN-01", "INOUT-LORA", 32, "weak_teacher", protected=True),
    Arm("QWEN-05", "INOUT-LORA", 32, "weak_teacher", protected=True),
), "QWEN-01", "CE")
SCHEDULE = Schedule((2048, 4096, 8192), 8192,
    (("train", "train", 37_748_736, 8192), ("validation", "validation", 98_304, 32),
     ("locked_test", "test", 278_528, 64)), 278_528)
SETTINGS = {'CE': (3e-05, 0), 'QWEN-01': (3e-05, 0.1), 'QWEN-05': (3e-05, 0.5)}

def main(argv=None):
    return run(SPEC, SCHEDULE, SETTINGS, argv)

if __name__ == "__main__":
    main()
