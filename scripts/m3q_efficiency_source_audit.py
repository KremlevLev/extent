"""Read ONLY small source manifests at the immutable source revision."""
import json,tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import hf_hub_download,HfApi
from extent.config import load_config
from scripts import m3q_subspace_engine as engine

ROOT=Path(__file__).resolve().parents[1]

def main():
    pin=json.loads((ROOT/'results/EXP-111-113-source-pin.json').read_text())
    config,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    seq=engine.configured_contract(config)
    def manifest(prefix,slot):
        path=hf_hub_download(pin['repo_id'],f'{prefix}/checkpoints/{slot}/checkpoint.json',
            repo_type='dataset',revision=pin['revision'],token=False,
            local_dir=Path(tempfile.gettempdir())/'extent-efficiency-source-audit')
        return json.loads(Path(path).read_text())
    def check(pair):
        seed,layer,depth=pair
        prep=manifest(engine.EXP069_PREFIX,f'prep/seed-{seed}/layer-{layer}')
        if prep['contract']!=dict(seq['source_contract'],seed=seed,layer=layer,kind='prepared_mamba'):
            raise ValueError('EXP069 manifest contract differs')
        endpoint=manifest(engine.EXP072['HF_PREFIX'],f'seed-{seed}/ONPOLICY/layer-{layer}')
        expected=engine.sequential.endpoint_contract(seq,seed,'ONPOLICY',layer,depth,prep['checkpoint_sha256'])
        if endpoint['contract']!=expected or endpoint['checkpoint_sha256']!=pin['seeds'][str(seed)][str(layer)]:
            raise ValueError('EXP072 manifest contract/SHA differs')
        return dict(seed=seed,layer=layer,depth=depth,sha256=endpoint['checkpoint_sha256'],payloads=[
            f"{engine.EXP069_PREFIX}/checkpoints/prep/seed-{seed}/layer-{layer}/{prep['checkpoint_file']}",
            f"{engine.EXP072['HF_PREFIX']}/checkpoints/seed-{seed}/ONPOLICY/layer-{layer}/{endpoint['checkpoint_file']}"])
    tasks=[(seed,layer,depth) for seed in (123,456) for depth,layer in enumerate(seq['replacement_order'],1)]
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,tasks))
    paths=[p for r in rows for p in r['payloads']]
    info=HfApi(token=False).get_paths_info(pin['repo_id'],paths,repo_type='dataset',revision=pin['revision'])
    if {i.path for i in info}!=set(paths) or not all(hasattr(i,'size') for i in info):raise FileNotFoundError('source binary paths unavailable')
    report=dict(repo_id=pin['repo_id'],revision=pin['revision'],passed=True,
        source_payload_bytes=sum(i.size for i in info),
        scope='96 metadata files and 96 referenced binary paths exist; payload SHA checked during TPU restore; no weights downloaded',endpoints=rows)
    (ROOT/'results/EXP-111-113-source-preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('All 48 endpoints and 48 preparation manifests matched')
    return report

if __name__=='__main__':main()
