"""Long training slice with pinned capacity from EXP115; reused validation."""
from pathlib import Path
import json
import numpy as np
from extent.plateau_data import pack_panels,hash_array
from extent.calibration_data import WIKITEXT_CONFIG_FILES,WIKITEXT_REPO,WIKITEXT_REVISION
from extent.qwen_source import QWEN3_1_7B_BASE

PANELS={'A':('validation',122880,64),'B':('validation',180224,32),
        'C':('validation',196608,64),'FIT':('train',74973184,16384)}


def load_data(directory,check=lambda:None):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    file=directory/'scale-panels.npz';meta_file=directory/'scale-panels.json'
    if file.exists() and meta_file.exists():
        arrays=dict(np.load(file,allow_pickle=False));meta=json.loads(meta_file.read_text())
        if meta['spec']!=json.loads(json.dumps(PANELS)):raise ValueError('scale data spec changed')
        if any(hash_array(v)!=meta['panels'][k]['sha256'] for k,v in arrays.items()):
            raise ValueError('scale data cache SHA mismatch')
        return arrays,meta
    tokenizer=AutoTokenizer.from_pretrained(QWEN3_1_7B_BASE.repo_id,revision=QWEN3_1_7B_BASE.revision,
        cache_dir=directory,trust_remote_code=False)
    arrays={};provenance={}
    for split in ('validation','train'):
        specs={k:(o,n) for k,(s,o,n) in PANELS.items() if s==split}
        def rows():
            for filename in WIKITEXT_CONFIG_FILES['wikitext-103-raw-v1'][split]:
                check();p=hf_hub_download(WIKITEXT_REPO,filename,repo_type='dataset',
                    revision=WIKITEXT_REVISION,cache_dir=directory)
                for text in pq.read_table(p,columns=['text'])['text'].to_pylist():
                    check();yield text
        found,prov=pack_panels(tokenizer,rows(),specs);arrays.update(found);provenance[split]=prov
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'results/EXP-114-data-preflight.json').read_text())['panels']
    growth=json.loads((root/'results/EXP-115-data-preflight.json').read_text())['panels']
    expected={'A':old['A'],'B':old['B'],'C':old['C_wiki'],'FIT':growth['train']}
    manifest={k:dict(sha256=hash_array(v),shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()}
    if manifest!=expected:raise ValueError('scale data differs from existing pinned tokens')
    meta=dict(spec=PANELS,panels=manifest,provenance=provenance,
        tokenizer_revision=QWEN3_1_7B_BASE.revision,wiki_revision=WIKITEXT_REVISION,
        interpretation='A/B/C reused validation; FIT same as115 (including119 prefix); no independent test claim')
    tmp=directory/'scale-panels.tmp.npz';np.savez(tmp,**arrays);tmp.replace(file)
    meta_file.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return arrays,meta


def training_index(step):
    # Fixed same permutation for all recipes/GPUs/experiments. Coprime stride
    # visits each of16384 windows exactly once per epoch, then cycles4 epochs.
    return (int(step)*7919+120123)%16384
