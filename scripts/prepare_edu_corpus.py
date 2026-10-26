"""Bounded CPU FineWeb-Edu pack; one source Parquet shard at a time."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from extent.broad_corpus import PackedCorpus,build_corpus,document_digest

REPO='HuggingFaceFW/fineweb-edu'
REVISION='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
VALIDATION_DOCUMENTS=4096


def rows_from_documents(documents,c4_db,check,done):
    """Stable split by raw row index; C4 validation digests never enter either split."""
    db=sqlite3.connect(Path(c4_db).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        for index,row in enumerate(documents):
            check()
            split='validation' if index<VALIDATION_DOCUMENTS else 'train'
            if done('validation') and done('train'):break
            text=row['text'];url=row.get('url','')
            # Keep raw stable indices even for excluded documents: builder rejects
            # empty replacements and records rejection, preserving resume cursors.
            query='SELECT 1 FROM documents WHERE digest=?'+(" AND split='validation'" if split=='train' else '')
            if db.execute(query,(document_digest(text),)).fetchone():
                text=''
            yield dict(split=split,index=index if split=='validation' else index-VALIDATION_DOCUMENTS,
                       text=text,url=url)
    finally:db.close()


def source_documents(root,check):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    # Verified inventory at pinned revision: 000..013, no mutable main/API lookup.
    files=[f'sample/10BT/{i:03d}_00000.parquet' for i in range(14)]
    for name in files:
        check()
        with tempfile.TemporaryDirectory(prefix='edu-source-',dir=root) as cache:
            path=hf_hub_download(REPO,name,repo_type='dataset',revision=REVISION,cache_dir=cache)
            parquet=pq.ParquetFile(path)
            try:
                for batch in parquet.iter_batches(batch_size=256,columns=['text','url']):
                    check()
                    yield from batch.to_pylist()
            finally:parquet.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--c4-dir',required=True)
    p.add_argument('--train-tokens',type=int,default=100000000)
    p.add_argument('--max-hours',type=float,default=3.)
    p.add_argument('--max-local-gb',type=float,default=8.)
    p.add_argument('--stop-file')
    a=p.parse_args(argv)
    if a.max_hours<=0 or a.max_local_gb<=0:raise ValueError('positive budgets required')
    c4=PackedCorpus(a.c4_dir);c4_db=Path(a.c4_dir)/'documents.sqlite'
    if not c4_db.exists():raise FileNotFoundError('C4 document digest journal required')
    deadline=time.time()+a.max_hours*3600
    def check():
        if time.time()>=deadline or (a.stop_file and Path(a.stop_file).exists()):
            raise TimeoutError('Edu preparation stopped; same command resumes journal')
    os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(deadline)
    from extent.hf_download_retry import install_download_retries
    install_download_retries()
    from extent.qwen_source import QWEN3_1_7B_BASE as source
    from transformers import AutoTokenizer
    from filelock import FileLock
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'build.lock'),timeout=0):
        tokenizer=AutoTokenizer.from_pretrained(source.repo_id,revision=source.revision,cache_dir=root/'tokenizer')
        contract=dict(dataset=REPO,revision=REVISION,config='sample-10BT',
            tokenizer=source.repo_id,tokenizer_revision=source.revision,vocab_size=source.vocab_size,
            eos_token_id=tokenizer.eos_token_id,sequence_length=256,train_tokens=a.train_tokens,
            validation_tokens=131072,validation_source_documents=VALIDATION_DOCUMENTS,
            excluded_c4_identity=c4.identity,
            packing='document concatenation with EOS; no implicit epochs',
            dedup='SHA256 NFKC normalized exact documents; validation reserved first; Edu validation excludes ALL C4 docs; Edu train excludes C4 validation; Wikipedia URLs excluded')
        counts={};goals=dict(train=a.train_tokens,validation=131072);last_log=[0.]
        def progress(state):
            counts.update(state['counts'])
            if time.monotonic()-last_log[0]>=30:
                print('Engine data preparation:',dict(counts),flush=True);last_log[0]=time.monotonic()
        documents=source_documents(root,check)
        rows=rows_from_documents(documents,c4_db,check,lambda split:counts.get(split,0)>=goals[split])
        try:
            result=build_corpus(root,rows,lambda text:tokenizer.encode(text,add_special_tokens=False),contract,
                check=check,progress=progress,max_local_bytes=int(a.max_local_gb*1e9))
        finally:rows.close();documents.close()
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()
