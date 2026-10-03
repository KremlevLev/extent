"""EXP-098: rank-matched dynamics protection and doubled recovery horizon."""
from scripts.m3q_subspace_engine import Arm, Campaign, Schedule
from scripts.m3q_safe_recovery_campaign import run

SPEC = Campaign(98, "dynamics-protection", (
    Arm("SAFE-R32", "INOUT-LORA", 32),
    Arm("SAFE-PROTECTED-R32", "INOUT-LORA", 32, protected=True),
), "SAFE-PROTECTED-R32", "SAFE-R32")
SCHEDULE = Schedule((4096, 8192, 16384), 16384,
    (("train", "train", 33_554_432, 16384), ("validation", "validation", 81_920, 32),
     ("locked_test", "test", 262_144, 64)), 262_144)

def main(argv=None):
    return run(SPEC, SCHEDULE, argv)

if __name__ == "__main__":
    main()
