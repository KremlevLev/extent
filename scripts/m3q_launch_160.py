"""One last-cell entry point: prepare pinned C4, then spend remaining T4 time."""
from pathlib import Path
import json
import time


def launch(corpus_dir, session_started_unix, *, clock=time.time, prepare=None, train=None):
    from extent.broad_corpus import PackedCorpus
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'results/EXP-160-corpus-manifest.json').read_text())
    corpus_dir=Path(corpus_dir)
    deadline=session_started_unix+12*3600-5*60
    remaining=lambda:deadline-clock()
    if remaining()<=3600:
        raise TimeoutError('Less than one hour remains in this kernel session; do not start preparation')
    if not (corpus_dir/'manifest.json').exists():
        if prepare is None:
            from scripts.prepare_broad_corpus import main as prepare
        budget=min(2.,(remaining()-3600)/3600)
        print('EXP160: preparing100M C4 on CPU; this time counts inside12h',flush=True)
        prepare(['--output',str(corpus_dir),'--train-tokens','100000000',
                 '--validation-windows','512','--max-hours',str(budget),'--max-local-gb','8'])
    corpus=PackedCorpus(corpus_dir)
    if corpus.manifest['contract']!=expected['contract'] or corpus.manifest['files']!=expected['files']:
        raise ValueError('Prepared corpus differs from registered EXP160 bytes; refusing training')
    hours=min(11.8,remaining()/3600)
    if hours<=.5:raise TimeoutError('Insufficient remaining time including18min save reserve')
    if train is None:
        from scripts.m3q_corpus_recovery_160 import main as train
    print(f'EXP160: corpus verified; runner budget={hours:.3f}h INCLUDING18min save reserve',flush=True)
    return train(['--corpus-dir',str(corpus_dir),'--max-wall-hours',str(hours),
                  '--save-reserve-minutes','18','--max-steps','262144','--max-new-hf-gb','8'])


def main():
    import psutil
    return launch('/kaggle/working/extent-c4-en-100m-v1',psutil.Process().create_time())


if __name__=='__main__':main()
