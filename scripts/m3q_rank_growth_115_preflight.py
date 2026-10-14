"""CPU-only token capacity/SHA preflight; no model weights or TPU initialization."""
import argparse,json
from pathlib import Path
from extent.rank_growth_data import load_tokens

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-dir',required=True)
    args=p.parse_args(argv);tokens,meta=load_tokens(args.cache_dir)
    dest=Path(__file__).resolve().parents[1]/'results/EXP-115-data-preflight.json'
    dest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('EXP115 token SHA/capacity passed',meta['panels'],flush=True)
if __name__=='__main__':main()
