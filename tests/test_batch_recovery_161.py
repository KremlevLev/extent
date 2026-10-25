from types import SimpleNamespace
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent.batch_recovery_161 import training_windows,windows_seen,ForkStore,sha,parent_replay
from extent.batch_execution_161 import BatchStep
from extent.recovery_durability import RecoveryCheckpointStore
from tests.test_hidden_distillation_159 import fixture
from extent.hidden_distillation_159 import optimizer,adam_counts


def initialized_parent(tmp_path):
    p,t,q,spec,tp=fixture(0);spec.update(name='KD-C4-FRESH',data_stream='c4')
    tx=optimizer(q,spec);state=tx.init(q)
    for _ in range(2):
        delta,state=tx.update(jax.tree.map(lambda v:jnp.ones_like(v)*.01,q),state,q)
        q=jax.tree.map(lambda a,b:a+b,q,delta)
    payload=dict(parameters=q,optimizer=state,best=jax.tree.map(lambda v:v+.01,q),best_step=jnp.asarray(118305,jnp.int32))
    parent=RecoveryCheckpointStore(tmp_path/'parent','parent')
    meta=parent.save(spec['name'],{'parent':'unit'},payload,118306)
    source=dict(step=118306,parent_windows=2,adam_count=2,metadata=meta,
                manifest_sha=sha(parent.directory(spec['name'])/'checkpoint.json'))
    return p,t,q,spec,tp,tx,payload,parent,source


def test_batch6_windows_are_disjoint_across_updates_and_resume(monkeypatch):
    import extent.competitor_tpu
    c=dict(batch=6,parent=dict(branches={'c4':dict(step=120181,parent_windows=1877),'wiki':dict(step=120181,parent_windows=1877)}),
           recipes=[dict(name='c4',data_stream='c4'),dict(name='wiki',data_stream='wiki')])
    class Corpus:
        def batch(self,i):return np.full((1,256),i,np.int32)
    a=training_windows(None,Corpus(),c,'c4',120181)[:,0].tolist()
    b=training_windows(None,Corpus(),c,'c4',120182)[:,0].tolist()
    assert a==list(range(1877,1883)) and b==list(range(1883,1889))
    assert windows_seen(c,'c4',120191)==1937
    monkeypatch.setattr(extent.competitor_tpu,'training_batch',lambda pool,i,*a:np.full((1,256),i,np.int32))
    assert training_windows(None,Corpus(),c,'wiki',120181)[:,0].tolist()==list(range(120181,120187))


def test_fork_preserves_current_adam_best_and_resumes_without_parent(tmp_path):
    p,t,q,spec,tp,tx,payload,parent,source=initialized_parent(tmp_path)
    pointer=parent.directory(spec['name'])/'checkpoint.json';before=pointer.read_bytes()
    child=ForkStore(tmp_path/'child','child',parent_root=parent.root,source_plan={'branches':{spec['name']:source}})
    c={'child':'unit'}
    template=dict(parameters=q,optimizer=tx.init(q),best=q,best_step=jnp.asarray(-1,jnp.int32))
    restored,meta=child.restore(spec['name'],c,template)
    assert meta['step']==118306 and adam_counts(restored['optimizer'])==[2]*6
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(payload)):np.testing.assert_array_equal(a,b)
    assert pointer.read_bytes()==before
    pointer.write_text('parent advanced/pruned')
    restored,_=child.restore(spec['name'],c,template)
    assert adam_counts(restored['optimizer'])==[2]*6


def test_real_batch6_gradient_audit_and_exactly_one_adam_increment(tmp_path):
    p,t,q,spec,tp,tx,payload,parent,source=initialized_parent(tmp_path)
    tokens=jnp.concatenate([(t+i)%p.cfg.vocab_size for i in range(6)],axis=0)
    state=payload['optimizer'];before=jax.tree.map(np.array,(q,state))
    engine=BatchStep(p.cfg,p.tc,p.columns,spec,tx)
    result=engine(q,state,p.base,tp,tokens,jnp.asarray(118306.,jnp.float32))
    gate=result[2]['engine_gate']
    assert gate['reference_microsteps']==6 and gate['comparison']['passed'],gate
    assert gate['selected']=='compiled-batch6' and adam_counts(result[1])==[3]*6
    again=engine(result[0],result[1],p.base,tp,tokens,jnp.asarray(118307.,jnp.float32))
    assert adam_counts(again[1])==[4]*6 and bool(again[2]['finite'])
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves((q,state))):np.testing.assert_array_equal(a,b)


def test_parent_replay_rejects_wrong_current_not_historical_best():
    source={'current_reference':{'panels':{'A':{'windows':[4.,5.]}}}}
    assert parent_replay({'A':{'windows':[4.,5.]}},source)['passed']
    assert not parent_replay({'A':{'windows':[4.,6.]}},source)['passed']
    assert parent_replay({'A':{'windows':[4.,5.]}},{})['passed'] is None


def test_freeze_parent_records_current_and_reuses_immutable_plan(tmp_path):
    from extent.batch_recovery_161 import freeze_parent,recipes
    p,t,q,spec,tp,tx,payload,_,source=initialized_parent(tmp_path/'fixture')
    c=dict(protocol='exp160-corpus-server-v1',code={},corpus_identity='unit')
    store=RecoveryCheckpointStore(tmp_path/'states/runtime-report','parent')
    report=dict(contract=c,runtime={},status='deadline_partial',branches={})
    for recipe in recipes(161):
        name=recipe['name'];store.save(name,dict(c,arm=name),payload,118306)
        report['branches'][name]=dict(step=118306,source_step=118304,source_replay={'passed':True})
    path=tmp_path/'results/runtime-report.json';path.parent.mkdir();path.write_text(json.dumps(report))
    before=path.read_bytes()
    plan,value=freeze_parent(tmp_path)
    assert len(value['branches'])==2 and all(s['adam_count']==2 for s in value['branches'].values())
    assert path.read_bytes()==before
    path.write_text('old report later changed')
    assert freeze_parent(tmp_path)[1]==value and plan.exists()


def test_batch_launcher_uses_same_cache_but_new_report_and_source_plan(tmp_path,monkeypatch):
    import sys,engine_batch
    from scripts import m3q_batch_recovery_161 as runner
    import extent.broad_corpus,extent.batch_recovery_161,extent.hf_artifact_sync,extent.hf_download_retry
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    corpus=tmp_path/'data-pack';corpus.mkdir();(corpus/'manifest.json').write_text('{}')
    monkeypatch.setenv('HF_TOKEN','offline-test-not-a-token')
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda p:SimpleNamespace(manifest=expected))
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda **kw:object())
    plan=tmp_path/'batch6/source.json';plan.parent.mkdir();plan.write_text('{}')
    monkeypatch.setattr(extent.batch_recovery_161,'freeze_parent',lambda *a,**kw:(plan,{}))
    monkeypatch.setattr(extent.hf_download_retry,'install_download_retries',lambda:None)
    device=SimpleNamespace(device_kind='CPU mocked launcher only',memory_stats=lambda:{})
    monkeypatch.setitem(sys.modules,'jax',SimpleNamespace(devices=lambda p:[device],config=SimpleNamespace(update=lambda *a:None)))
    calls=[]
    def run(args):calls.append(args);return dict(status='deadline_partial',pending_slots=[])
    monkeypatch.setattr(runner,'run',run)
    r=engine_batch.main(['--work-dir',str(tmp_path),'--max-wall-hours','21','--no-telegram'])
    assert r['status']=='deadline_partial' and calls[0].experiment==161
    assert calls[0].cache_dir==str(tmp_path/'cache') and calls[0].checkpoint_dir==str(tmp_path/'states')
    assert calls[0].source_plan==str(plan) and (tmp_path/'runtime-b6-status.json').exists()


def test_real_session_has_independent_namespace_and_pending_bootstrap(tmp_path,monkeypatch):
    import extent.hf_artifact_sync
    from importlib.metadata import version
    from extent.batch_recovery_161 import make_session,PREFIX
    from scripts.m3q_batch_recovery_161 import parser
    p,t,q,spec,tp,tx,payload,parent,source=initialized_parent(tmp_path/'fixture')
    copied=RecoveryCheckpointStore(tmp_path/'states/runtime-report','parent')
    meta=copied.save(spec['name'],source['metadata']['contract'],payload,118306)
    source.update(metadata=meta,manifest_sha=sha(copied.directory(spec['name'])/'checkpoint.json'))
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    c=dict(parent=dict(runtime=runtime,branches={spec['name']:source}))
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',lambda *a:False)
    args=parser().parse_args(['--source-plan','unit','--corpus-dir','unit',
        '--output-dir',str(tmp_path/'results'),'--checkpoint-dir',str(tmp_path/'states'),
        '--max-wall-hours','21','--no-telegram'])
    session=make_session(args,c,None)
    assert session.prefix==PREFIX and session.path.name=='runtime-b6.json'
    restored,_=session.store.restore(spec['name'],dict(c,arm=spec['name']),payload)
    assert spec['name'] in session.result['pending_slots'] and adam_counts(restored['optimizer'])==[2]*6
    assert session.result['branches'][spec['name']]['source_step']==118306


@pytest.mark.parametrize('bad_replay',[False,True])
def test_actual_runner_batch6_resume_tokens_and_source_gate(tmp_path,monkeypatch,bad_replay):
    import sys
    from importlib.metadata import version
    from scripts import m3q_batch_recovery_161 as runner
    from scripts.m3q_hidden_distillation_159 import contract_for
    from extent import batch_recovery_161 as model
    from extent.competitor_tpu import token_nll
    import extent.config,extent.reset_long_patient,extent.t4_scale_data,extent.hf_artifact_sync,extent.quality_benchmark_156,extent.broad_corpus
    p,t,q,spec,tp,tx,parent_payload,parent,source=initialized_parent(tmp_path)
    p.teacher=tp
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    ref=float(token_nll(p.cfg,p.base,q,t,p.columns,spec,jnp.asarray(118306),block_fn=model.block_forward).mean())
    source['current_reference']={'panels':{k:{'windows':[ref+int(bad_replay)]} for k in ('A','B','C','D')}}
    c=contract_for(159,0);c.update(recipes=[spec],horizon=2,eval_steps=[0],batch=6,parent=dict(branches={spec['name']:source}))
    pin=dict(source_step=118304,adam_step=0,best_step=118304,runtime=runtime,data={})
    monkeypatch.setattr(runner,'contract_for',lambda *a:c)
    monkeypatch.setattr(model,'source_pin',lambda:pin)
    monkeypatch.setattr(model,'restore_source',lambda *a:({'parameters':q},pin))
    monkeypatch.setattr(model,'base_coordinates',lambda *a:{})
    monkeypatch.setattr(model,'prepare',lambda *a:(p.base,q,118304))
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}))
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    requested=[]
    class Corpus:
        def __init__(self,*a):pass
        arrays={'train':np.zeros(1000,dtype=np.int32)}
        identity='tiny'
        def validation(self):return np.asarray(t)
        def batch(self,i):
            requested.append(i);return (np.asarray(t)+i)%p.cfg.vocab_size
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',Corpus)
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:object())
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    def data(path,*a):
        path.mkdir(parents=True,exist_ok=True);np.savez(path/'fresh-panels.npz',english_stories=np.arange(40960,dtype=np.int32)%p.cfg.vocab_size)
        (path/'data-manifest.json').write_text('{}')
    monkeypatch.setattr(extent.quality_benchmark_156,'prepare_data',data)
    devices=jax.devices;monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu') if platform=='gpu' else devices(platform))
    class Session:
        def __init__(self):
            self.result=dict(runtime=runtime,pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0),
                branches={spec['name']:dict(step=118306,source_step=118306,history=[],evaluations=[])})
            self.store=ForkStore(tmp_path/'child-run','child',parent_root=parent.root,source_plan={'branches':{spec['name']:source}})
            self.wall=time.monotonic()+3600;self.deadline=self.wall-1800
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=1:raise TimeoutError('unit work deadline')
        def persist(self,**kw):return True
        def emit(self,*a):pass
        def checkpoint(self,n,v,step):self.store.save(n,dict(c,arm=n),v,step)
    session=Session();monkeypatch.setattr(runner,'session',lambda *a:session)
    args=SimpleNamespace(experiment=161,worker=None,execution='server-balanced',source_plan='unit',corpus_dir='unit',max_steps=1000000000,cache_dir=str(tmp_path/'cache'))
    if bad_replay:
        with pytest.raises(RuntimeError):runner.run(args)
        assert not requested and session.result['cost'].get('accepted_updates',0)==0
    else:
        result=runner.run(args);row=result['branches'][spec['name']]
        assert requested==list(range(2,8))
        assert result['cost']['accepted_input_tokens']==6*t.size
        assert result['cost']['ce_targets']==6*(t.shape[1]-1)
        assert row['step']==118307 and row['training_windows_seen']==8
        assert row['adam_count_at_restore']==[2]*6
        result['cost']['accepted_updates']=0
        runner.run(args)
        assert requested==list(range(2,14)) and row['step']==118308 and row['training_windows_seen']==14
        assert row['adam_count_at_restore']==[3]*6
