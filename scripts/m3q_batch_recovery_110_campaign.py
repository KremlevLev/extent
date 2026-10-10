"""EXP-110: registered batch recovery account."""
from scripts.m3q_batch_recovery_campaign import main as run
def main(argv=None):
    return run(argv, experiment=110)
if __name__ == "__main__":
    main()
