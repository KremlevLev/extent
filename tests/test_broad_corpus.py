import json
import numpy as np
import pytest
from extent.broad_corpus import build_corpus, PackedCorpus, CorpusExhausted


def contract():
    return dict(sequence_length=4,train_tokens=16,validation_tokens=8,
                eos_token_id=99,vocab_size=100,source='fixture-v1')


def rows():
    # Validation document appears twice in training, including normalization.
    held='held out '+('a '*110)
    yield dict(split='validation',index=0,text=held)
    for i,text in enumerate([held,'  '+held.replace(' ','  '), 'fresh '+('b '*110), 'next '+('c '*110)]):
        yield dict(split='train',index=i,text=text)


def tokenize(text):
    return [1 if text.strip().startswith('held') else 2 if text.startswith('fresh') else 3]*7


def build(path,stream=None,**kwargs):
    return build_corpus(path,rows() if stream is None else stream,tokenize,contract(),
                        min_free_bytes=0,commit_every=1,**kwargs)


def test_cross_split_dedup_and_no_implicit_epoch(tmp_path):
    m=build(tmp_path)
    assert m['statistics']['duplicates']==2
    corpus=PackedCorpus(tmp_path)
    assert set(corpus.arrays['train'])=={2,3,99}
    assert set(corpus.arrays['validation'])=={1,99}
    assert corpus.batch(0).shape==(1,4)
    with pytest.raises(CorpusExhausted):corpus.batch(4)
    with pytest.raises(CorpusExhausted):corpus.batch(-1)
    assert build(tmp_path)==m
    c=contract();c['train_tokens']=20
    with pytest.raises(ValueError,match='contract differs'):
        build_corpus(tmp_path,rows(),tokenize,c)


def test_resume_rolls_back_uncommitted_documents_and_bytes(tmp_path):
    def broken():
        for i,row in enumerate(rows()):
            yield row
            if i==3:raise ConnectionError('interrupted after appending first training doc')
    with pytest.raises(ConnectionError):
        build_corpus(tmp_path,broken(),tokenize,contract(),commit_every=3,min_free_bytes=0)
    assert not (tmp_path/'manifest.json').exists()
    resumed=build(tmp_path)
    other=tmp_path/'uninterrupted'; reference=build(other)
    assert resumed['files']==reference['files']
    assert resumed['statistics']==reference['statistics']


def test_corrupt_completed_corpus_and_partial_exhaustion(tmp_path):
    with pytest.raises(CorpusExhausted):build(tmp_path,iter(list(rows())[:1]))
    build(tmp_path)
    with open(tmp_path/'train.bin','r+b') as f:f.write(b'bad!')
    with pytest.raises(ValueError,match='SHA'):PackedCorpus(tmp_path)


def test_preparation_stops_each_source_split_without_scanning_entire_c4(tmp_path,monkeypatch):
    import gzip
    import huggingface_hub
    from scripts.prepare_broad_corpus import source_rows
    calls=[];done=set()
    def download(repo,name,**kwargs):
        calls.append(name);path=tmp_path/'source.gz'
        with gzip.open(path,'wt') as f:
            for i in range(5):f.write(json.dumps(dict(text='doc'+str(i),url='example.org'))+'\n')
        return path
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',download)
    stream=source_rows(tmp_path,lambda:None,lambda split:split in done)
    a=next(stream);done.add('validation')
    b=next(stream);done.add('train')
    assert list(stream)==[] and a['split']=='validation' and b['split']=='train'
    assert len(calls)==2 and not list(tmp_path.glob('source-*'))
