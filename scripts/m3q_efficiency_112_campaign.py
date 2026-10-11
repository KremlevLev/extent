"""EXP112: standalone account entry."""
from scripts.m3q_efficiency_campaign import main as run
def main(argv=None):
    return run(argv,experiment=112)
if __name__=="__main__":
    main()
