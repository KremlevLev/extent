import copy
import json
from types import SimpleNamespace
import pytest
from engine_dashboard import Mirror,points,run_identity


def fixture():
    return dict(status='running',pending_slots=['one'],contract=dict(experiment=161,batch=6,parent_plan_sha='fixed'),branches={
        'KD-C4-FRESH':dict(step=120183,source_step=120181,best_nll=4.8,
            source_controls={'A':{'nll':5.},'D':{'nll':7.}},
            history=[dict(step=120182,loss=3.1,ce=4.8,kl=1.4,seconds=1.,engine_gate={'selected':'compiled'}),
                     dict(step=120183,loss=3.,ce=4.7,kl=1.3,seconds=.9)],
            evaluations=[dict(step=120183,panels={'A':{'nll':4.9}},fresh_nll={'english_stories':{'nll':6.,'windows':[6.]*16}})],
            final=dict(last=dict(step=120183,panels={'D':{'nll':6.9}},fresh_nll={'english_stories':{'nll':6.1,'windows':[6.1]*128}}),
                       best=dict(step=120182,panels={'A':{'nll':1.}})))})


class SDK:
    def __init__(self):self.runs=[]
    def Settings(self,**kw):return kw
    def init(self,**kw):
        run=SimpleNamespace(url='https://example.test/run',summary={},logs=[],settings=kw,
            define_metric=lambda *a,**kw:None,finish=lambda:None)
        run.log=lambda value:run.logs.append(value)
        self.runs.append(run);return run


def test_axes_batch_tokens_validation_subsets_and_no_best_as_current():
    result=points(fixture(),'KD-C4-FRESH')
    assert [p['progress/new_input_tokens'] for p in result]==[0,1536,3072]
    assert 'performance/input_tokens_per_second' not in result[1] # startup audit excluded
    assert result[2]['validation/en_stories_16windows_nll']==6.
    assert result[2]['validation/en_stories_128windows_nll']==6.1
    assert result[2]['validation/wiki_A_nll']==4.9
    assert 'validation/wiki_A_nll' not in result[1] # historical BEST not injected


def test_readonly_backfill_dedup_late_evaluation_and_persistent_run_identity(tmp_path):
    report=fixture();before=copy.deepcopy(report);sdk=SDK()
    mirror=Mirror(sdk,tmp_path,'unit')
    mirror.update(report);run=sdk.runs[0]
    initial=[x for x in run.logs if 'progress/new_input_tokens' in x]
    assert len(initial)==3 and report==before
    mirror.update(report)
    assert len([x for x in run.logs if 'progress/new_input_tokens' in x])==3
    report['branches']['KD-C4-FRESH']['evaluations'][0]['panels']['B']={'nll':5.1}
    mirror.update(report)
    assert any(x.get('validation/wiki_B_nll')==5.1 for x in run.logs)
    other=Mirror(sdk,tmp_path,'unit');other.update(report)
    assert not any('progress/new_input_tokens' in x for x in sdk.runs[-1].logs)
    settings=run.settings['settings']
    assert settings['x_disable_stats'] and settings['x_disable_meta'] and settings['disable_git']
    assert run.settings['config'].get('HF_TOKEN') is None and run.settings['config']['observer_only']
    assert run.settings['resume']=='allow' and run.settings['reinit']=='create_new'


def test_state_recovery_starts_separate_curve_instead_of_mixing_rollback():
    r=fixture();key=run_identity(r,'KD-C4-FRESH','unit',None)
    r['branches']['KD-C4-FRESH']['rollbacks']=[dict(reported=120190,restored=120183)]
    assert run_identity(r,'KD-C4-FRESH','unit',None)!=key


def test_linux_lock_detection_only_reads_owned_inode(tmp_path,monkeypatch):
    import engine_dashboard as module
    from pathlib import Path
    lock=tmp_path/'server.lock';lock.touch();inode=lock.stat().st_ino
    monkeypatch.setattr(module.os,'major',lambda d:8,raising=False)
    monkeypatch.setattr(module.os,'minor',lambda d:1,raising=False)
    original=Path.read_text;body=[f'1: FLOCK ADVISORY WRITE 123 08:01:{inode} 0 EOF']
    def read(path,*a,**kw):return body[0] if str(path).replace('\\','/')=='/proc/locks' else original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    assert module.trainer_active(tmp_path) is True
    body[0]='1: FLOCK ADVISORY WRITE 999 08:01:1 0 EOF'
    assert module.trainer_active(tmp_path) is False


def test_real_sdk_offline_two_runs_no_training_imports(tmp_path,monkeypatch):
    wandb=pytest.importorskip('wandb')
    report=fixture();report['branches']['KD-WIKI-CYCLIC']=copy.deepcopy(report['branches']['KD-C4-FRESH'])
    mirror=Mirror(wandb,tmp_path,'engine-dashboard-unit',mode='offline')
    try:
        mirror.update(report)
        assert len(mirror.runs)==2
        assert all(not r._is_finished for r in mirror.runs.values())
        assert all(r.settings.x_disable_meta and r.settings.x_disable_stats for r in mirror.runs.values())
    finally:mirror.finish()
    assert (tmp_path/'links.json').exists()
