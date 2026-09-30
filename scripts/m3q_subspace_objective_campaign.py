"""EXP-095: objective bridge with the same rank-8 INOUT corrections."""
from scripts.m3q_subspace_engine import Arm, Campaign, run_campaign

SPEC = Campaign(95, "subspace-objectives", (
    Arm("CE", "INOUT-LORA", 8),
    Arm("TEACHER-KD", "INOUT-LORA", 8, "teacher"),
    Arm("SELF-ANCHOR", "INOUT-LORA", 8, "self"),
    Arm("DELTA-THEN-KD", "INOUT-LORA", 8, "delta_then_teacher"),
), "DELTA-THEN-KD", "TEACHER-KD")

def main(argv=None):
    return run_campaign(SPEC, argv)

if __name__ == "__main__":
    main()
