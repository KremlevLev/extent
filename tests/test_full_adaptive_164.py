import json,time,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from extent import full_adaptive_164 as model
from extent import batch_planner as planner
from tests.test_batch_recovery_161 import initialized_parent
from extent.method_sweep_163 import coordinates
from extent.recovery_durability import RecoveryCheckpointStore


def test_scan_all_batches_selects_throughput_not_largest_and_releases_oom():
    ticks=[0.];attempts=[];released=[]
    def invoke(engine,batch):
        attempts.append(batch)
        if batch>=7:raise MemoryError('test limit')
        ticks[0]+= {1:1.,2:1.1,3:1.2,4:2.,5:3.,6:4.}[batch]
    result=planner.benchmark(lambda n:n,invoke,lambda:released.append(True),clock=lambda:ticks[0])
    assert result['selected_batch']==3 and result['accepted_updates']==0
    assert {r['batch'] for r in result['trials']}==set(range(1,9))
    assert len(released)==8 and len(attempts)==20
    with pytest.raises(RuntimeError,match='No compiled batch'):
        planner.benchmark(lambda n:n,lambda *a:(_ for _ in ()).throw(MemoryError()),lambda:None)


def test_real_full_batch_scan_does_not_update_current_or_adam(tmp_path):
    p,t,current,_,tp,_,_,_,_=initialized_parent(tmp_path)
    spec=dict(model.recipes(164)[0],align_layers=list(range(p.cfg.num_layers)))
    q=coordinates(p.base,current,p.cfg,p.columns,spec);tx=model.optimizer(q,spec);state=tx.init(q)
    batch=jnp.concatenate([(t+i)%p.cfg.vocab_size for i in range(8)])
    before=jax.tree.map(np.array,(q,state));records=[]
    kernel,result=planner.plan_kernel(p.cfg,p.tc,p.columns,spec,tx,q,state,p.base,tp,batch,jnp.asarray(157684.),lambda:None,records.append)
    assert len(result['trials'])==8 and result['gradient_audit']['passed']
    assert result['accepted_updates']==0 and result['probe_input_presentations']>0
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves((q,state))):np.testing.assert_array_equal(a,b)
    out=kernel(q,state,p.base,tp,batch[:result['selected_batch']],jnp.asarray(157684.))
    assert model.adam_counts(out[1])==[1]*6 and bool(out[2]['finite'])


def test_source_full_fork_carries_moments_best_and_data_and_own_resume(tmp_path):
    p,t,current,_,tp,_,_,_,_=initialized_parent(tmp_path)
    spec=model.recipes(164)[0];q=coordinates(p.base,current,p.cfg,p.columns,spec);tx=model.optimizer(q,spec)
    # Nonzero optimizer moments plus exact inherited count.
    _,state=tx.update(jax.tree.map(lambda v:jnp.ones_like(v)*.1,q),tx.init(q),q)
    state=jax.tree.map(lambda v:jnp.asarray(2832,v.dtype) if v.shape==() and jnp.issubdtype(v.dtype,jnp.integer) else v,state)
    payload=dict(parameters=q,optimizer=state,best=jax.tree.map(lambda v:v+.01,q),best_step=jnp.asarray(156900,jnp.int32))
    store=RecoveryCheckpointStore(tmp_path/'parent163','parent');meta=store.save('FULL-KD',{'parent':'163'},payload,157684)
    pointer=store.directory('FULL-KD')/'checkpoint.json';original=pointer.read_bytes()
    source=dict(metadata=meta,manifest_sha=model.sha(pointer))
    child=model.ForkStore(tmp_path/'child','child',source=source,parent_root=store.root)
    template=dict(parameters=q,optimizer=tx.init(q),best=q,best_step=jnp.asarray(-1,jnp.int32),data_windows=jnp.asarray(0,jnp.int32))
    got,metadata=child.restore('FULL-KD',{'child':'164'},template)
    assert metadata['step']==157684 and int(got['data_windows'])==16992 and model.adam_counts(got['optimizer'])==[2832]*6
    for a,b in zip(jax.tree.leaves(payload),jax.tree.leaves({k:v for k,v in got.items() if k!='data_windows'})):np.testing.assert_array_equal(a,b)
    assert pointer.read_bytes()==original
    got['data_windows']=jnp.asarray(17003,jnp.int32);child.save('FULL-KD',{'child':'164'},got,157686)
    pointer.write_text('parent no longer consulted after bootstrap')
    resumed,meta=child.restore('FULL-KD',{'child':'164'},template)
    assert int(resumed['data_windows'])==17003 and meta['step']==157686


def test_actual_runner_adaptive_tokens_resume_cursors_and_final_diagnostics(tmp_path,monkeypatch):
    from scripts import m3q_full_adaptive_164 as runner
    from scripts.m3q_hidden_distillation_159 import contract_for
    import extent.config,extent.reset_long_patient,extent.t4_scale_data,extent.hf_artifact_sync,extent.quality_benchmark_156,extent.broad_corpus
    from importlib.metadata import version
    p,t,current,_,tp,_,_,_,_=initialized_parent(tmp_path);p.teacher=tp
    spec=dict(model.recipes(164)[0],align_layers=list(range(p.cfg.num_layers)))
    q=coordinates(p.base,current,p.cfg,p.columns,spec);tx=model.optimizer(q,spec)
    state=jax.tree.map(lambda v:jnp.asarray(2832,v.dtype) if v.shape==() and jnp.issubdtype(v.dtype,jnp.integer) else v,tx.init(q))
    source_store=RecoveryCheckpointStore(tmp_path/'source163','src')
    meta=source_store.save('FULL-KD',{'parent':'163'},dict(parameters=q,optimizer=state,best=q,best_step=jnp.asarray(157684,jnp.int32)),157684)
    source=dict(step=157684,windows=16992,adam_count=2832,metadata=meta,manifest_sha=model.sha(source_store.directory('FULL-KD')/'checkpoint.json'))
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    c=contract_for(159,0);c.update(experiment=164,recipes=[spec],horizon=2,eval_steps=[0],batch=None,full_source=source,parent=dict(branches={'FULL-KD':{'adam_count':36548}}))
    pin=dict(source_step=118304,runtime=runtime,data={})
    monkeypatch.setattr(runner,'contract_for',lambda *a:c)
    monkeypatch.setattr(model,'source_pin',lambda:pin);monkeypatch.setattr(model,'restore_source',lambda *a:({'parameters':current},pin))
    monkeypatch.setattr(model,'base_coordinates',lambda *a:{})
    monkeypatch.setattr(model,'prepare',lambda *a,**kw:(p.base,q,157684))
    monkeypatch.setattr(model,'teacher_validation',lambda *a,**kw:dict(nll=1.,windows=[1.]))
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}));monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    requested=[]
    class Corpus:
        arrays={'train':np.zeros((16992+100)*256,np.int32)};identity='tiny'
        def __init__(self,*a):pass
        def validation(self):return np.asarray(t)
        def batch(self,i):requested.append(i);return np.resize((np.asarray(t)+i)%p.cfg.vocab_size,(1,256))
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',Corpus);monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:object())
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    def data(path,*a):
        path.mkdir(parents=True,exist_ok=True);np.savez(path/'fresh-panels.npz',english_stories=np.arange(32768,dtype=np.int32)%p.cfg.vocab_size);(path/'data-manifest.json').write_text('{}')
    monkeypatch.setattr(extent.quality_benchmark_156,'prepare_data',data)
    devices=jax.devices;monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu') if platform=='gpu' else devices(platform))
    selections=iter([2,3]);planned=[]
    def plan(cfg,tc,columns,spec,tx,q,state,base,teacher,tokens,cursor,check,record):
        batch=next(selections);planned.append(batch)
        def update(q,state,*a):
            _,after=tx.update(jax.tree.map(jnp.zeros_like,q),state,q)
            return q,after,dict(finite=jnp.asarray(True),loss=jnp.asarray(1.),ce=jnp.asarray(1.),kl=jnp.asarray(1.))
        return update,dict(selected_batch=batch,probe_input_presentations=100)
    monkeypatch.setattr(planner,'plan_kernel',plan)
    class Session:
        def __init__(self):
            self.result=dict(runtime=runtime,pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0),branches={'FULL-KD':dict(step=157684,source_step=157684,history=[],evaluations=[])})
            self.store=model.ForkStore(tmp_path/'child-run','child',source=source,parent_root=source_store.root)
            self.wall=time.monotonic()+3600;self.deadline=self.wall-1800;self.invocation={}
        def check(self):pass
        def persist(self,**kw):return True
        def emit(self,*a):pass
        def checkpoint(self,name,value,step):self.store.save(name,dict(c,arm=name),value,step)
    session=Session();monkeypatch.setattr(runner,'session',lambda *a:session)
    args=SimpleNamespace(experiment=164,worker=None,execution='server-balanced',source_plan='unit',corpus_dir='c4',edu_dir='edu',max_steps=1000000000,cache_dir=str(tmp_path/'cache'),checkpoint_dir=str(tmp_path/'states'))
    first=runner.run(args);row=first['branches']['FULL-KD']
    assert row['step']==157686 and row['training_windows_seen']==16996
    assert row['accepted_input_tokens']==4*256 and first['finalization_status']=='completed'
    assert len(row['final']['last']['fresh_nll']['english_stories']['windows'])==128
    # Resume after a pause with a different selected physical batch.
    session.result['status']='deadline_partial';c['horizon']=3
    # Scientific contract unchanged in a real resume; fixture horizon extends only to exercise next update.
    pointer=session.store.directory('FULL-KD')/'checkpoint.json';meta=json.loads(pointer.read_text());meta['contract']=dict(c,arm='FULL-KD');pointer.write_text(json.dumps(meta))
    second=runner.run(args);row=second['branches']['FULL-KD']
    assert planned==[2,3] and row['step']==157687 and row['training_windows_seen']==16999
    assert row['accepted_input_tokens']==7*256 and row['adam_count_at_restore']==[2834]*6
    # Probe windows are deliberately read but never consumed; training starts at same cursor.
    assert requested[:8]==list(range(16992,17000)) and requested[8:12]==list(range(16992,16996))


def test_actual_contract_session_and_soft_stop_force_save(tmp_path,monkeypatch):
    from importlib.metadata import version
    from scripts import m3q_full_adaptive_164 as runner
    import extent.broad_corpus,extent.hf_artifact_sync
    from scripts.prepare_edu_corpus import REVISION
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    c4=SimpleNamespace(identity='fixed-c4',manifest=expected)
    edu=SimpleNamespace(identity='fixed-edu',arrays={'train':np.zeros(20000*256,np.int32)},manifest=dict(contract=dict(
        dataset='HuggingFaceFW/fineweb-edu',revision=REVISION,excluded_c4_identity='fixed-c4',sequence_length=256,
        tokenizer_revision=expected['contract']['tokenizer_revision'])))
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda path:c4 if path=='c4' else edu)
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    source=dict(step=157684,windows=16992,adam_count=2832,runtime=runtime)
    parent=dict(format='methods-parent-v1',corpus_identity='fixed-c4',runtime=runtime,branches={'FULL-KD':dict(step=154852,parent_windows=0)},full_source=source)
    plan=tmp_path/'source.json';plan.write_text(json.dumps(parent))
    c=model.contract(164,None,1000000000,'c4','server-balanced',plan,'edu')
    assert c['experiment']==164 and c['batch'] is None and len(c['recipes'])==1 and c['recipes'][0]['mode']=='carry-full-adam'
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',lambda *a:False)
    args=runner.parser().parse_args(['--source-plan',str(plan),'--corpus-dir','c4','--edu-dir','edu','--output-dir',str(tmp_path/'results'),
        '--checkpoint-dir',str(tmp_path/'states'),'--stop-file',str(tmp_path/'STOP'),'--max-wall-hours','14','--no-telegram'])
    s=model.make_session(args,c,None);assert s.path.name=='runtime-full.json'
    assert s.result['branches']['FULL-KD']['source_step']==157684
    writes=[];s.store.save=lambda name,contract,payload,step:writes.append(step)
    payload=dict(best_step=jnp.asarray(157684));s.checkpoint('FULL-KD',payload,157684);s.checkpoint('FULL-KD',payload,157685)
    assert writes==[157684]
    (tmp_path/'STOP').touch()
    with pytest.raises(TimeoutError,match='soft stop'):s.check()
    s.finalizing=True;s.checkpoint('FULL-KD',payload,157685);assert writes==[157684,157685]
