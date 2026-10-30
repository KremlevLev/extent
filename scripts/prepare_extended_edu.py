"""Rebuild a larger pinned pack in a new directory; prove old prefix unchanged."""
import argparse,hashlib,json,time
from pathlib import Path
from extent.broad_corpus import PackedCorpus,digest_file,write_json


def verify_prefix(old_dir,new_dir):
    old=PackedCorpus(old_dir);new=PackedCorpus(new_dir)
    left=dict(old.manifest['contract']);right=dict(new.manifest['contract'])
    left.pop('train_tokens');right.pop('train_tokens')
    if left!=right:raise ValueError('extension changes corpus/tokenizer/split contract')
    if old.manifest['files']['validation']!=new.manifest['files']['validation']:
        raise ValueError('extension changes validation bytes')
    size=old.manifest['files']['train']['tokens']*4
    if new.manifest['files']['train']['tokens']<=size//4:raise ValueError('not an extension')
    h=hashlib.sha256()
    with (Path(new_dir)/'train.bin').open('rb') as stream:
        remaining=size
        while remaining:
            block=stream.read(min(8<<20,remaining))
            if not block:raise ValueError('truncated extended prefix')
            h.update(block);remaining-=len(block)
    if h.hexdigest()!=old.manifest['files']['train']['sha256']:raise ValueError('old training prefix differs')
    value=dict(old_identity=old.identity,new_identity=new.identity,old_train_tokens=size//4,
        new_train_tokens=new.manifest['files']['train']['tokens'],prefix_sha256=h.hexdigest(),
        validation_sha256=new.manifest['files']['validation']['sha256'],verified=True)
    write_json(Path(new_dir)/'extension-proof.json',value);return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--c4-dir',required=True);p.add_argument('--train-tokens',type=int,default=600000000)
    p.add_argument('--max-hours',type=float,default=8.);p.add_argument('--stop-file')
    a=p.parse_args(argv)
    if Path(a.old_dir).resolve()==Path(a.output).resolve():raise ValueError('preserve original pack; distinct output required')
    from scripts.prepare_edu_corpus import main as prepare
    # Rebuilding from the pinned source preserves the truncated last old document
    # correctly; copying its digest journal would lose the remainder of that doc.
    prepare(['--output',a.output,'--c4-dir',a.c4_dir,'--train-tokens',str(a.train_tokens),
        '--max-hours',str(a.max_hours),'--max-local-gb','12']+(['--stop-file',a.stop_file] if a.stop_file else []))
    return verify_prefix(a.old_dir,a.output)


if __name__=='__main__':main()
