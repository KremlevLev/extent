"""EXP-103: registered second-stage late-dynamics-release."""
from scripts.m3q_subspace_engine import Arm, Campaign, Schedule
from scripts.m3q_plateau_campaign import run

SPEC = Campaign(103, "late-dynamics-release", (
    Arm("PROTECTED", "INOUT-LORA", 32, "ce", protected=True),
    Arm("RELEASED", "INOUT-LORA", 32, "ce", protected=False),
), "RELEASED", "PROTECTED")
SCHEDULE = Schedule((2048, 4096, 8192), 8192,
    (("train", "train", 37_748_736, 8192), ("validation", "validation", 98_304, 32),
     ("locked_test", "test", 278_528, 64)), 278_528)
SETTINGS = {'PROTECTED': (3e-05, 0), 'RELEASED': (3e-05, 0)}

def main(argv=None):
    return run(SPEC, SCHEDULE, SETTINGS, argv)

if __name__ == "__main__":
    main()
