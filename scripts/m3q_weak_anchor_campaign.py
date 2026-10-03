"""EXP-100: CE-first Qwen/self anchoring of stable protected rank32 recovery."""
from scripts.m3q_subspace_engine import Arm, Campaign, Schedule
from scripts.m3q_safe_recovery_campaign import run

SPEC = Campaign(100, "weak-anchor", (
    Arm("CE", "INOUT-LORA", 32, protected=True),
    Arm("QWEN-ANCHOR", "INOUT-LORA", 32, "weak_teacher", protected=True),
    Arm("START-ANCHOR", "INOUT-LORA", 32, "self", protected=True),
), "QWEN-ANCHOR", "CE")
SCHEDULE = Schedule((2048, 4096, 8192), 8192,
    (("train", "train", 33_554_432, 8192), ("validation", "validation", 81_920, 32),
     ("locked_test", "test", 262_144, 64)), 262_144)

def main(argv=None):
    return run(SPEC, SCHEDULE, argv)

if __name__ == "__main__":
    main()
