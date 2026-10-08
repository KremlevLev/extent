"""EXP107: layer-group adaptive clipping; matched EXP106 is on the other account."""
from scripts.m3q_layer_clip_campaign import main as run


def main(argv=None):
    return run(argv, experiment=107)


if __name__ == "__main__":
    main()
