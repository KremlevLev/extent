"""EXP-099: stable protected rank16/32/64 recovery capacity comparison."""
from scripts.m3q_subspace_engine import Arm, Campaign, Schedule
from scripts.m3q_safe_recovery_campaign import run

SPEC = Campaign(99, "protected-capacity", (
    Arm("SAFE-PROTECTED-R16", "INOUT-LORA", 16, protected=True),
    Arm("SAFE-PROTECTED-R32", "INOUT-LORA", 32, protected=True),
    Arm("SAFE-PROTECTED-R64", "INOUT-LORA", 64, protected=True),
), "SAFE-PROTECTED-R64", "SAFE-PROTECTED-R32")
SCHEDULE = Schedule((4096, 8192, 12288), 12288,
    (("train", "train", 33_554_432, 12288), ("validation", "validation", 81_920, 32),
     ("locked_test", "test", 262_144, 64)), 262_144)

def main(argv=None):
    return run(SPEC, SCHEDULE, argv)

if __name__ == "__main__":
    main()
