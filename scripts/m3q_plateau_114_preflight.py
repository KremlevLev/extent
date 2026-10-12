"""Read-only HF source metadata pin + real token capacity/SHA. No weights."""
import argparse,json,hashlib
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from extent.plateau_data import load_tokens
from extent.campaign_checkpoint import write_json_atomic

ROOT=Path(__file__).resolve().parents[1]
PREFIX='experiments/exp111-recovery-efficiency'

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',default=str(ROOT/'work/exp114-preflight'))
    p.add_argument('--source-only',action='store_true')
    p.add_argument('--data-only',action='store_true')
    args=p.parse_args(argv); cache=Path(args.cache_dir)
    if args.data_only:
        tokens,meta=load_tokens(cache)
        old=json.loads((ROOT/'results/EXP-111-113-data-preflight.json').read_text())
        if meta['panels']['A']!=old['validation']:raise ValueError('legacy A packing differs')
        write_json_atomic(ROOT/'results/EXP-114-data-preflight.json',meta)
        print('All EXP114 token ranges fit; SHA/provenance recorded',flush=True)
        return meta
    repo='levkremlev888/extent'; api=HfApi(token=False)
    revision=api.repo_info(repo,repo_type='dataset').sha
    def read(path):
        f=Path(hf_hub_download(repo,path,repo_type='dataset',revision=revision,
            token=False,cache_dir=cache)); raw=f.read_bytes()
        return json.loads(raw),hashlib.sha256(raw).hexdigest()
    latest,latest_sha=read(f'{PREFIX}/latest.json')
    if latest['contract']['experiment']!=111 or latest['status']!='completed' or latest['pending_slots']:
        raise ValueError('EXP111 is not completed/durable')
    manifests={};paths=[]
    for seed in (123,456):
        row=latest['branches'][str(seed)]['MAIN']
        if row['step']!=32768 or row['durable_step']!=32768:raise ValueError('endpoint mismatch')
        name=f'{PREFIX}/checkpoints/{seed}/MAIN/checkpoint.json';meta,sha=read(name)
        if meta['step']!=32768 or meta['contract']!=dict(latest['contract'],seed=seed,arm='MAIN'):
            raise ValueError('manifest contract mismatch')
        manifests[str(seed)]=dict(path=name,sha256=sha,metadata=meta)
        paths += [f'{PREFIX}/checkpoints/{seed}/MAIN/{c["file"]}' for r in meta['leaves'] for c in r['chunks']]
    found=api.get_paths_info(repo,paths,repo_type='dataset',revision=revision)
    if {x.path for x in found}!=set(paths):raise FileNotFoundError('patient chunks unavailable')
    pin=dict(repo_id=repo,revision=revision,latest_sha256=latest_sha,
        source_contract=latest['contract'],source_models=latest['source_models'],runtime=latest['runtime'],
        reference_final={str(s):latest['branches'][str(s)]['MAIN']['final'] for s in (123,456)},
        manifests=manifests,scope='metadata and binary path existence; chunk SHA checked on runtime restore')
    write_json_atomic(ROOT/'results/EXP-114-source-pin.json',pin)
    print(f'Pinned durable EXP111: {revision}; {len(paths)} chunks exist',flush=True)
    if not args.source_only:
        tokens,meta=load_tokens(cache)
        # A/T-seen must reproduce exactly the original111 tokens.
        old=json.loads((ROOT/'results/EXP-111-113-data-preflight.json').read_text())
        if meta['panels']['A']!=old['validation']:raise ValueError('legacy A packing differs')
        write_json_atomic(ROOT/'results/EXP-114-data-preflight.json',meta)
        print('All EXP114 token ranges fit; SHA/provenance recorded',flush=True)
    return pin

if __name__=='__main__':main()
