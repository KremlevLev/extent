"""EXP105: full Mamba-only capacity versus protected rank32 corrections."""
from scripts.m3q_paper_decoder_recovery_campaign import main as run


def main(argv=None):
    return run(argv, experiment=105)


if __name__ == "__main__":
    main()
