"""EXP-094: which foldable correction subspace recovers the assembled hybrid?"""
from scripts.m3q_subspace_engine import Arm, Campaign, run_campaign

SPEC = Campaign(94, "subspace", (
    Arm("OUT-LORA", "OUT-LORA", 8),
    Arm("INOUT-LORA", "INOUT-LORA", 8),
    Arm("MLP-READOUT-LORA", "MLP-READOUT-LORA", 8),
    Arm("HEAD-GAIN", "HEAD-GAIN", 8),
), "INOUT-LORA", "OUT-LORA")

def main(argv=None):
    return run_campaign(SPEC, argv)

if __name__ == "__main__":
    main()
