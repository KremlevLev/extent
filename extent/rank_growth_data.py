"""New post-EXP114 train range, reused selection and held-out EXP115 panels."""
from pathlib import Path
import numpy as np
from extent.plateau_data import pack_panels,hash_array
from extent.calibration_data import (WIKITEXT_CONFIG_FILES,WIKITEXT_REPO,WIKITEXT_REVISION,
    PG19_REVISION,load_pg19_tokens)
from extent.qwen_source import QWEN3_1_7B_BASE
from extent.pinned_pg19_manifest import prepare_pg19_test_manifest

PANELS={'train':('train',74973184,16384),'validation':('validation',122880,64),
        'locked_wiki':('test',262144,64)}
PG19=(524288,128)

def load_tokens(cache):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    prepare_pg19_test_manifest(cache)
    tokenizer=AutoTokenizer.from_pretrained(QWEN3_1_7B_BASE.repo_id,
        revision=QWEN3_1_7B_BASE.revision,cache_dir=cache,trust_remote_code=False)
    tokens={};packing={}
    for split in ('train','validation','test'):
        specs={k:(o,n) for k,(s,o,n) in PANELS.items() if s==split}
        def rows():
            for filename in WIKITEXT_CONFIG_FILES['wikitext-103-raw-v1'][split]:
                path=hf_hub_download(WIKITEXT_REPO,filename,repo_type='dataset',revision=WIKITEXT_REVISION,cache_dir=cache)
                yield from pq.read_table(path,columns=['text'])['text'].to_pylist()
        found,meta=pack_panels(tokenizer,rows(),specs);tokens.update(found);packing[split]=meta
    tokens['locked_pg19']=load_pg19_tokens(PG19[1]*256,cache,token_offset=PG19[0],
        dataset_split='test',tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
        tokenizer_revision=QWEN3_1_7B_BASE.revision).reshape(PG19[1],256)
    hashes={k:set(hash_array(w) for w in v) for k,v in tokens.items()}
    duplicate={f'{a}/{b}':len(hashes[a]&hashes[b]) for a,b in
        (('train','validation'),('train','locked_wiki'),('train','locked_pg19'),('validation','locked_wiki'))}
    if any(duplicate.values()):raise ValueError(f'exact duplicate windows: {duplicate}')
    return tokens,dict(panels={k:dict(sha256=hash_array(v),shape=list(v.shape),dtype=str(v.dtype)) for k,v in tokens.items()},
        packing=packing,duplicates=duplicate,wiki_revision=WIKITEXT_REVISION,pg19_revision=PG19_REVISION,
        tokenizer_revision=QWEN3_1_7B_BASE.revision,
        limitations='Selection intentionally reused; locked panels not used for selection. Historical corpus exposure and book independence are not established.')
