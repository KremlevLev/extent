import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent.capacity_probe import measure_batches,recommend
from tests.test_device_execution import arguments


def test_real_batched_gradients_are_finite_and_optimizer_never_advances(tmp_path):
    p,tx,args,spec=arguments()
    before=jax.tree.map(np.array,args[:2]);t=np.asarray(args[4])
    report=measure_batches(p.cfg,p.tc,p.columns,spec,tx,args,
        lambda batch:np.repeat(t,batch,axis=0),tmp_path/'result.json',[1,2],time.time()+120)
    assert report['status']=='completed' and report['accepted_updates']==0
    assert [v['batch'] for v in report['trials']]==[1,2]
    assert all(v['finite'] and len(v['samples_seconds'])==3 for v in report['trials'])
    assert abs(report['trials'][0]['objective']['ce']-report['trials'][1]['objective']['ce'])<.03
    assert report['recommended_batch'] in (1,2)
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(args[:2])):np.testing.assert_array_equal(a,b)


def test_recommendation_ignores_unfinished_and_chooses_smaller_near_tie():
    report=dict(trials=[dict(batch=1,status='measured',finite=True,input_tokens_per_second=100),
                       dict(batch=2,status='measured',finite=True,input_tokens_per_second=102),
                       dict(batch=4,status='compiling',input_tokens_per_second=999)])
    assert recommend(report)==1
    report['trials'][1]['input_tokens_per_second']=120
    assert recommend(report)==2


def test_parent_budget_keeps_completed_trial_and_does_not_touch_checkpoint(tmp_path,monkeypatch):
    import engine_capacity as entry
    from types import SimpleNamespace
    monkeypatch.setattr(entry.shutil,'disk_usage',lambda p:SimpleNamespace(free=10**12))
    root=tmp_path;(root/'results').mkdir()
    source=root/'results/runtime-report.json';source.write_text('{"parent":true}')
    checkpoint=root/'states'/'keep.bin';checkpoint.parent.mkdir();checkpoint.write_bytes(b'checkpoint')
    class Process:
        returncode=None
        pid=123456789
        def __init__(self,cmd,**kw):
            assert cmd[cmd.index('--work-dir')+1]==str(root)
            report=dict(status='profiling',trials=[dict(batch=1,status='measured',finite=True,input_tokens_per_second=400),dict(batch=2,status='compiling')])
            (root/'results/runtime-capacity.json').write_text(json.dumps(report))
        def wait(self,timeout):raise entry.subprocess.TimeoutExpired('probe',timeout)
        def poll(self):return self.returncode
    monkeypatch.setattr(entry.subprocess,'Popen',Process)
    def stop(p):p.returncode=-15
    monkeypatch.setattr(entry,'stop_probe',stop)
    report=entry.main(['--work-dir',str(root),'--max-minutes','.01','--no-telegram'])
    assert report['recommended_batch']==1 and report['status']=='partial'
    assert report['accepted_updates']==0 and not report['applied_to_training']
    assert source.read_text()=='{"parent":true}' and checkpoint.read_bytes()==b'checkpoint'


def test_running_training_lock_prevents_parallel_probe(tmp_path,monkeypatch):
    import engine_capacity as entry
    from types import SimpleNamespace
    monkeypatch.setattr(entry.shutil,'disk_usage',lambda p:SimpleNamespace(free=10**12))
    from filelock import FileLock
    (tmp_path/'results').mkdir();(tmp_path/'results/runtime-report.json').write_text('{}')
    with FileLock(str(tmp_path/'server.lock'),timeout=0):
        with pytest.raises(RuntimeError,match='still running'):
            entry.main(['--work-dir',str(tmp_path)])


def test_worker_uses_read_only_state_and_never_publishes_parent_changes(tmp_path,monkeypatch):
    import scripts.engine_probe_worker as worker
    import extent.capacity_probe,extent.broad_corpus,extent.hf_artifact_sync
    from scripts import m3q_corpus_recovery_160 as runner
    from extent import corpus_recovery_160 as core
    root=tmp_path;(root/'results').mkdir()
    parent=dict(contract=dict(horizon=1000000000),branches={},status='deadline_partial',cost={'accepted_updates':123})
    path=root/'results/runtime-report.json';path.write_text(json.dumps(parent));before=path.read_bytes()
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda p:object())
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda **kw:object())
    actual=[]
    def run(options):
        session=runner.session(options,parent['contract'],object())
        assert session.store.hub is None
        assert session.result['cost']['attempted_updates']==0
        session.checkpoint('any',{},12345);session.persist(force=True)
        assert not (root/'states').exists()
        session.result['cost']['accepted_updates']=999
        actual.append(options)
        return {'status':'deadline_partial'}
    monkeypatch.setattr(runner,'run',run)
    original_session=runner.session;original_factory=core.make_kernel
    worker.main(['--work-dir',str(root),'--deadline',str(time.time()+120)])
    assert path.read_bytes()==before and len(actual)==1
    assert runner.session is original_session and core.make_kernel is original_factory
