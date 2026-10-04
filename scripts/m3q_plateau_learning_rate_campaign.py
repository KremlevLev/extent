"""EXP-102: registered second-stage plateau-learning-rate."""
from scripts.m3q_subspace_engine import Arm, Campaign, Schedule
from scripts.m3q_plateau_campaign import run

SPEC = Campaign(102, "plateau-learning-rate", (
    Arm("LR-3E4", "INOUT-LORA", 32, "ce", protected=True),
    Arm("LR-1E4", "INOUT-LORA", 32, "ce", protected=True),
    Arm("LR-3E5", "INOUT-LORA", 32, "ce", protected=True),
), "LR-3E5", "LR-3E4")
SCHEDULE = Schedule((2048, 4096, 8192), 8192,
    (("train", "train", 37_748_736, 8192), ("validation", "validation", 98_304, 32),
     ("locked_test", "test", 278_528, 64)), 278_528)
SETTINGS = {'LR-3E4': (0.0003, 0), 'LR-1E4': (0.0001, 0), 'LR-3E5': (3e-05, 0)}

def main(argv=None):
    return run(SPEC, SCHEDULE, SETTINGS, argv)

if __name__ == "__main__":
    main()
