"""EXP-096: correction capacity versus protecting Mamba dynamics."""
from scripts.m3q_subspace_engine import Arm, Campaign, run_campaign

SPEC = Campaign(96, "subspace-capacity", (
    Arm("RANK8", "INOUT-LORA", 8),
    Arm("RANK32", "INOUT-LORA", 32),
    Arm("RANK64", "INOUT-LORA", 64),
    Arm("RANK32-PROTECTED", "INOUT-LORA", 32, protected=True),
), "RANK32-PROTECTED", "RANK32")

def main(argv=None):
    return run_campaign(SPEC, argv)

if __name__ == "__main__":
    main()
