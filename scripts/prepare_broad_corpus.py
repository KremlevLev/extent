"""Prepare C4 on CPU before booking an accelerator. One compressed source shard at a time."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile
import time
from extent.broad_corpus import C4_REVISION, build_corpus


def source_rows(root, check, done):
    from huggingface_hub import hf_hub_download
    # Validation first so exact documents already reserved there cannot train.
    for split, count in [('validation',8),('train',1024)]:
        if done(split): continue
        index = 0
        for shard in range(count):
            check()
            name = f'en/c4-{split}.{shard:05d}-of-{count:05d}.json.gz'
            with tempfile.TemporaryDirectory(prefix='source-', dir=root) as cache:
                path = hf_hub_download('allenai/c4',name,repo_type='dataset',revision=C4_REVISION,cache_dir=cache)
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        check(); row = json.loads(line)
                        yield dict(split=split,index=index,text=row['text'],url=row.get('url',''))
                        index += 1
                        if done(split): break
            if done(split): break


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--train-tokens',type=int,default=100_000_000)
    p.add_argument('--validation-windows',type=int,default=512)
    p.add_argument('--max-hours',type=float,default=6.)
    p.add_argument('--max-local-gb',type=float,default=8.)
    p.add_argument('--stop-file')
    a=p.parse_args(argv)
    if a.max_hours<=0 or a.max_local_gb<=0: raise ValueError('positive budgets required')
    deadline=time.time()+a.max_hours*3600
    def check():
        if a.stop_file and Path(a.stop_file).exists():raise TimeoutError('requested stop during CPU corpus preparation; restart resumes journal')
        if time.time()>=deadline: raise TimeoutError('corpus preparation deadline; restart same command to resume')
    import os
    os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(deadline)
    from extent.hf_download_retry import install_download_retries
    install_download_retries()
    from transformers import AutoTokenizer
    from filelock import FileLock
    from extent.qwen_source import QWEN3_1_7B_BASE as source
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'build.lock'),timeout=0):
        tokenizer=AutoTokenizer.from_pretrained(source.repo_id,revision=source.revision,cache_dir=root/'tokenizer')
        contract=dict(dataset='allenai/c4',revision=C4_REVISION,config='en',
            tokenizer=source.repo_id,tokenizer_revision=source.revision,vocab_size=source.vocab_size,
            eos_token_id=tokenizer.eos_token_id,sequence_length=256,train_tokens=a.train_tokens,
            validation_tokens=a.validation_windows*256,packing='document concatenation with EOS; no implicit epochs',
            dedup='SHA256 NFKC whitespace-normalized document, validation reserved first; Wikipedia URLs excluded')
        counts={}
        goals={'train':contract['train_tokens'],'validation':contract['validation_tokens']}
        rows=source_rows(root,check,lambda split:counts.get(split,0)>=goals[split])
        try:
            result=build_corpus(root,rows,lambda t:tokenizer.encode(t,add_special_tokens=False),contract,
                check=check,progress=lambda state:counts.update(state['counts']),max_local_bytes=int(a.max_local_gb*1e9))
        finally: rows.close()
    print(json.dumps(result,indent=2));return result

if __name__=='__main__': main()
