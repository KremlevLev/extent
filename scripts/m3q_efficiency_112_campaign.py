"""EXP112: standalone account entry."""
from scripts.m3q_efficiency_campaign import main as run
def main(argv=None):
    # Operational cache preparation only; legacy scientific contract is unchanged.
    import argparse
    import sys
    from extent.pinned_pg19_manifest import prepare_pg19_test_manifest
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset-cache-dir', default='/kaggle/working/extent-dataset-cache')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--sync-only', action='store_true')
    parsed, _ = parser.parse_known_args(args)
    if not parsed.plan_only and not parsed.sync_only and '--help' not in args and '-h' not in args:
        prepare_pg19_test_manifest(parsed.dataset_cache_dir)
    return run(args,experiment=112)
if __name__=="__main__":
    main()
