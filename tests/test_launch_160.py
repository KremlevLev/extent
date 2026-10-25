import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.m3q_launch_160 import launch


def test_preparation_is_subtracted_from_same_twelve_hour_budget(tmp_path,monkeypatch):
    import extent.broad_corpus
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda p:SimpleNamespace(manifest=expected))
    now=[600.];prepared=[];trained=[]
    def prepare(argv):prepared.append(argv);now[0]+=1200
    def train(argv):trained.append(argv);return {'ok':True}
    assert launch(tmp_path,0,clock=lambda:now[0],prepare=prepare,train=train)=={'ok':True}
    args=trained[0];hours=float(args[args.index('--max-wall-hours')+1])
    assert hours*3600==12*3600-300-1800
    assert args[args.index('--save-reserve-minutes')+1]=='18'
    assert len(prepared)==1


def test_existing_corpus_skips_preparation_and_wrong_bytes_reject(tmp_path,monkeypatch):
    import extent.broad_corpus
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    (tmp_path/'manifest.json').write_text('{}')
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda p:SimpleNamespace(manifest=expected))
    def forbidden(a):raise AssertionError('must not prepare existing corpus')
    assert launch(tmp_path,0,clock=lambda:0,prepare=forbidden,train=lambda a:'ran')=='ran'
    expected['files']['train']['sha256']='wrong'
    with pytest.raises(ValueError,match='differs'):
        launch(tmp_path,0,clock=lambda:0,prepare=forbidden,train=lambda a:'ran')
