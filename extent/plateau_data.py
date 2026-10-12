"""Fixed EXP114 panels; CPU-only pinning without model allocation."""
from pathlib import Path
import hashlib
import re
import numpy as np
from extent.calibration_data import (load_pg19_tokens, WIKITEXT_CONFIG_FILES,
    WIKITEXT_REPO, WIKITEXT_REVISION, PG19_REVISION)
from extent.qwen_source import QWEN3_1_7B_BASE

PANELS = {'A': ('validation',122880,64), 'B': ('validation',180224,32),
    'C_wiki': ('validation',196608,64), 'FIT': ('train',73924608,4096),
    'T_seen': ('train',65536000,64)}
PG19 = (458752,128)

def hash_array(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def pack_panels(tokenizer, rows, panels, length=256):
    """Exactly the legacy row+EOS packing, with article/row provenance.

    Only retain requested slices; never allocate the entire 74M-token prefix.
    Capacity means traversed prefix capacity (a lower bound), not corpus total.
    """
    arrays={name:np.empty(n*length,np.int32) for name,(_,n) in panels.items()}
    provenance={name:[set() for _ in range(n)] for name,(_,n) in panels.items()}
    end=max(offset+n*length for offset,n in panels.values())
    cursor=0; article='unassigned'; eos=tokenizer.eos_token_id
    if eos is None:raise ValueError('tokenizer EOS required')
    for row_id,text in enumerate(rows):
        if not text or not text.strip():continue
        if re.match(r'^\s*=\s+[^=].*?\s+=\s*$',text):article=f'article-row-{row_id}'
        ids=np.asarray(tokenizer(text,add_special_tokens=False)['input_ids']+[int(eos)],np.int32)
        stop=cursor+len(ids)
        for name,(offset,n) in panels.items():
            lo=max(cursor,offset); hi=min(stop,offset+n*length)
            if hi>lo:
                arrays[name][lo-offset:hi-offset]=ids[lo-cursor:hi-cursor]
                for w in range((lo-offset)//length,(hi-offset-1)//length+1):
                    provenance[name][w].add(article)
        cursor=stop
        if cursor>=end:break
    if cursor<end:raise ValueError(f'capacity {cursor} below required {end}')
    return {k:v.reshape(-1,length) for k,v in arrays.items()}, {
        'traversed_capacity_lower_bound':cursor,
        'articles':{k:[sorted(s) for s in v] for k,v in provenance.items()}}

def load_tokens(cache):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(QWEN3_1_7B_BASE.repo_id,
        revision=QWEN3_1_7B_BASE.revision,cache_dir=cache,trust_remote_code=False)
    tokens={}; metadata={}
    for split in ('train','validation'):
        specs={n:(o,c) for n,(s,o,c) in PANELS.items() if s==split}
        def rows():
            for filename in WIKITEXT_CONFIG_FILES['wikitext-103-raw-v1'][split]:
                path=hf_hub_download(WIKITEXT_REPO,filename,repo_type='dataset',
                    revision=WIKITEXT_REVISION,cache_dir=cache)
                yield from pq.read_table(path,columns=['text'])['text'].to_pylist()
        found,meta=pack_panels(tokenizer,rows(),specs); tokens.update(found);metadata[split]=meta
    tokens['T_new']=tokens['FIT'][:64]
    tokens['C_pg19']=load_pg19_tokens(PG19[1]*256,cache,
        token_offset=PG19[0],dataset_split='test',tokenizer_repo=QWEN3_1_7B_BASE.repo_id,
        tokenizer_revision=QWEN3_1_7B_BASE.revision).reshape(PG19[1],256)
    manifest={k:dict(sha256=hash_array(v),shape=list(v.shape),dtype=str(v.dtype)) for k,v in tokens.items()}
    articles={k:set(a for w in metadata['validation']['articles'][k] for a in w) for k in ('A','B','C_wiki')}
    overlap={f'{a}/{b}':sorted(articles[a]&articles[b]) for a,b in (('A','B'),('A','C_wiki'),('B','C_wiki'))}
    hashes={k:set(hash_array(w) for w in v) for k,v in tokens.items()}
    duplicates={f'{a}/{b}':len(hashes[a]&hashes[b]) for a,b in
        (('A','B'),('A','C_wiki'),('B','C_wiki'),('FIT','C_wiki'),('FIT','C_pg19'))}
    if any(duplicates.values()):raise ValueError(f'exact duplicate windows: {duplicates}')
    return tokens,dict(panels=manifest,packing=metadata,article_overlap=overlap,
        exact_window_duplicates=duplicates,pg19_document_independence='not established; contiguous book slice',
        wiki_revision=WIKITEXT_REVISION,pg19_revision=PG19_REVISION,
        tokenizer_revision=QWEN3_1_7B_BASE.revision)
