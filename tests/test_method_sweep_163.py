import json,time,sys
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.test_batch_recovery_161 import initialized_parent
from extent import method_sweep_163 as model
from extent.competitor_tpu import token_nll
from extent.rank_growth_153 import block_forward as original_block


@pytest.mark.parametrize('scope',['rank','upper','full'])
def test_current_replay_real_batch6_gradients_and_host_roundtrip(tmp_path,scope):
    p,t,current,source_spec,tp,_,payload,parent,source=initialized_parent(tmp_path)
    spec=next(s for s in model.recipes(163) if s['scope']==scope)
    spec=dict(spec,align_layers=list(range(p.cfg.num_layers)))
    q=model.coordinates(p.base,current,p.cfg,p.columns,spec)
    old=token_nll(p.cfg,p.base,current,t,p.columns,spec,jnp.asarray(154852),block_fn=original_block)
    new=token_nll(p.cfg,p.base,q,t,p.columns,spec,jnp.asarray(154852),block_fn=model.block_forward)
    np.testing.assert_allclose(np.asarray(old),np.asarray(new),atol=.03,rtol=0)
    tx=model.optimizer(q,spec);state=tx.init(q)
    assert model.adam_counts(state)==[0]*6
    batch=jnp.concatenate([(t+i)%p.cfg.vocab_size for i in range(6)])
    step=model.make_kernel(p.cfg,p.tc,p.columns,spec,tx)
    original_update=step.reference.update;update_calls=[]
    def tracked_update(*args):update_calls.append(True);return original_update(*args)
    step.reference.update=tracked_update
    result=step(q,state,p.base,tp,batch,jnp.asarray(154852,jnp.float32))
    assert bool(result[2]['finite']) and model.adam_counts(result[1])==[1]*6
    assert result[2]['engine_gate']['selected'] in ('compiled-batch6','accumulated-batch6')
    assert len(update_calls)==1  # not six discarded dense Adam proposals
    if scope=='upper':
        assert any(np.any(np.asarray(g)!=0) for i in result[3].values() for g in jax.tree.leaves(i.get('dense_mlp',{})))
    if scope=='full':
        assert any(np.any(np.asarray(g)!=0) for i in result[3].values() for g in jax.tree.leaves(i['dense_layer']['mlp']))
    saved=dict(parameters=result[0],optimizer=result[1],best=q,best_step=jnp.asarray(154852))
    host=model.to_host(saved);back=jax.device_put(host)
    for a,b in zip(jax.tree.leaves(saved),jax.tree.leaves(back)):np.testing.assert_array_equal(a,b)


def test_pilot_budget_uses_sum_of_rates_and_extends_without_rollback():
    rows={n:dict(step=154868,source_step=154852,timing_samples=[rate]*6) for n,rate in [('rank',1.),('upper',2.),('full',5.)]}
    target=model.matched_target(rows,1000,80)
    assert target==24  # eight more per arm cost64s <=.85*80=68s
    assert model.matched_target(rows,20,1000)==20
    assert model.matched_target(rows,1000,0)==16
    rows['rank']['step']+=10
    assert model.matched_target(rows,1000,0)>=26


def test_actual_contract_and_session_stop_force_final_checkpoint(tmp_path,monkeypatch):
    import extent.broad_corpus,extent.hf_artifact_sync
    from scripts import m3q_method_sweep_163 as runner
    from scripts.prepare_edu_corpus import REVISION
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    c4=SimpleNamespace(identity='fixed-c4',manifest=expected)
    edu=SimpleNamespace(identity='fixed-edu',arrays={'train':np.zeros(1536*100,np.int32)},manifest=dict(contract=dict(
        dataset='HuggingFaceFW/fineweb-edu',revision=REVISION,excluded_c4_identity='fixed-c4',sequence_length=256,
        tokenizer_revision=expected['contract']['tokenizer_revision'])))
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda path:c4 if path=='c4' else edu)
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    parent=dict(format='methods-parent-v1',corpus_identity='fixed-c4',runtime=runtime,branches={s['name']:dict(
        step=154852,parent_windows=0) for s in model.recipes(163)})
    plan=tmp_path/'source.json';plan.write_text(json.dumps(parent))
    c=model.contract(163,None,1000000000,'c4','server-balanced',plan,'edu')
    assert c['horizon']==100 and c['batch']==6 and len(c['recipes'])==3
    assert c['eval_steps']==[0,100]
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',lambda *args:False)
    stop=tmp_path/'STOP'
    args=runner.parser().parse_args(['--source-plan',str(plan),'--corpus-dir','c4','--edu-dir','edu',
        '--output-dir',str(tmp_path/'results'),'--checkpoint-dir',str(tmp_path/'states'),
        '--stop-file',str(stop),'--no-telegram','--max-wall-hours','14'])
    session=model.make_session(args,c,None)
    assert session.path.name=='runtime-methods.json'
    writes=[];session.store.save=lambda name,contract,payload,step:writes.append(step)
    payload=dict(best_step=jnp.asarray(154852))
    session.checkpoint('R256-CONTROL',payload,154852)
    session.checkpoint('R256-CONTROL',payload,154853)
    assert writes==[154852]
    stop.write_text('pause')
    with pytest.raises(TimeoutError,match='soft stop'):session.check()
    session.finalizing=True;session.checkpoint('R256-CONTROL',payload,154853)
    assert writes==[154852,154853]


def test_prepare_reads_same_current_then_new_adam_in_all_scopes(tmp_path,monkeypatch):
    from extent.recovery_durability import RecoveryCheckpointStore
    p,t,current,spec,tp,tx,payload,parent,source=initialized_parent(tmp_path)
    store=RecoveryCheckpointStore(tmp_path/'parent163','parent')
    meta=store.save(model.SOURCE_ARM,{'source':'fixture'},payload,154852)
    source=dict(source,step=154852,metadata=meta,manifest_sha=model.sha(store.directory(model.SOURCE_ARM)/'checkpoint.json'))
    monkeypatch.setattr(model,'old_prepare',lambda *args:(p.base,current,118304))
    for spec in model.recipes(163):
        base,q,cursor=model.prepare(p,spec,{'parameters':current},{},parent={'branches':{spec['name']:source}},parent_root=store.root)
        assert cursor==154852
        tx=model.optimizer(q,spec);template=dict(parameters=q,optimizer=tx.init(q),best=q,best_step=jnp.asarray(cursor))
        child=model.ForkStore(tmp_path/spec['name'],'child')
        restored,metadata=child.restore(spec['name'],dict(arm=spec['name']),template)
        assert metadata['step']==154852 and model.adam_counts(restored['optimizer'])==[0]*6


@pytest.mark.parametrize('bad_replay,pilot',[(False,False),(True,False),(False,True)])
def test_actual_three_method_runner_saves_offloaded_fresh_optimizers(tmp_path,monkeypatch,bad_replay,pilot):
    from scripts import m3q_method_sweep_163 as runner
    from scripts.m3q_hidden_distillation_159 import contract_for
    import extent.config,extent.reset_long_patient,extent.t4_scale_data,extent.hf_artifact_sync,extent.quality_benchmark_156,extent.broad_corpus
    p,t,current,old_spec,tp,_,payload,parent,source=initialized_parent(tmp_path);p.teacher=tp
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    specs=[dict(s,align_layers=list(range(p.cfg.num_layers))) for s in model.recipes(163)]
    ref=np.asarray(token_nll(p.cfg,p.base,current,t,p.columns,old_spec,jnp.asarray(154852),block_fn=original_block)).mean(axis=1)
    source=dict(source,step=154852,current_reference=dict(panels={k:dict(windows=(ref+int(bad_replay)).tolist()) for k in ('A','B','C','D')}))
    goal=20 if pilot else 2
    c=contract_for(159,0);c.update(recipes=specs,horizon=goal,eval_steps=[0],batch=6,parent=dict(branches={s['name']:source for s in specs}))
    pin=dict(source_step=118304,adam_step=0,runtime=runtime,data={})
    monkeypatch.setattr(runner,'contract_for',lambda *args:c)
    monkeypatch.setattr(model,'source_pin',lambda:pin)
    monkeypatch.setattr(model,'restore_source',lambda *args:({'parameters':current},pin))
    monkeypatch.setattr(model,'base_coordinates',lambda *args:{})
    monkeypatch.setattr(model,'prepare',lambda patient,spec,*args,**kw:(p.base,model.coordinates(p.base,current,p.cfg,p.columns,spec),154852))
    budgets=[]
    if pilot:
        class QuickKernel:
            check=lambda self:None
            def __init__(self,tx):
                def update(q,state):
                    _,after=tx.update(jax.tree.map(jnp.zeros_like,q),state,q)
                    return q,after,dict(finite=jnp.asarray(True),loss=jnp.asarray(1.),ce=jnp.asarray(1.),kl=jnp.asarray(1.))
                self.fn=jax.jit(update)
            def __call__(self,q,state,*args):return jax.block_until_ready(self.fn(q,state))
        monkeypatch.setattr(model,'make_kernel',lambda cfg,tc,columns,spec,tx,**kw:QuickKernel(tx))
        monkeypatch.setattr(model,'teacher_validation',lambda *args,**kw:dict(nll=1.,windows=[1.]))
        def target(*args):budgets.append(True);return 18 if len(budgets)==1 else 20
        monkeypatch.setattr(model,'matched_target',target)
    monkeypatch.setattr(extent.config,'load_config',lambda *args:(p.cfg,{}))
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *args,**kw:p)
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *args:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    requested=[]
    class Corpus:
        arrays={'train':np.zeros(1000,np.int32)};identity='tiny'
        def __init__(self,*args):pass
        def validation(self):return np.asarray(t)
        def batch(self,i):requested.append(i);return (np.asarray(t)+i)%p.cfg.vocab_size
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',Corpus)
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:object())
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kw:object())))
    def data(path,*args):
        path.mkdir(parents=True,exist_ok=True);np.savez(path/'fresh-panels.npz',english_stories=np.arange(40960,dtype=np.int32)%p.cfg.vocab_size)
        (path/'data-manifest.json').write_text('{}')
    monkeypatch.setattr(extent.quality_benchmark_156,'prepare_data',data)
    devices=jax.devices;monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu') if platform=='gpu' else devices(platform))
    class Session:
        def __init__(self):
            self.result=dict(runtime=runtime,pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0),
                branches={s['name']:dict(step=154852,source_step=154852,history=[],evaluations=[]) for s in specs})
            self.store=model.ForkStore(tmp_path/'child-run','child');self.wall=time.monotonic()+3600;self.deadline=self.wall-1800;self.invocation={}
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=3*goal:raise TimeoutError('unit pause')
        def persist(self,**kw):return True
        def emit(self,*args):pass
        def checkpoint(self,name,value,step):self.store.save(name,dict(c,arm=name),value,step)
    session=Session();monkeypatch.setattr(runner,'session',lambda *args:session)
    args=SimpleNamespace(experiment=163,worker=None,execution='server-balanced',source_plan='unit',corpus_dir='c4',edu_dir='edu',
                         max_steps=1000000000,cache_dir=str(tmp_path/'cache'),checkpoint_dir=str(tmp_path/'states'))
    if bad_replay:
        with pytest.raises(RuntimeError):runner.run(args)
        assert not requested and session.result['cost'].get('accepted_updates',0)==0
    else:
        result=runner.run(args)
        assert len(requested)==goal*6*3
        assert result['cost']['accepted_input_tokens']==3*goal*6*t.size
        if not pilot:assert requested==list(range(12))*3
        else:
            assert len(budgets)==2
            for row in result['branches'].values():assert any(e['step']==154870 for e in row['evaluations'])
        for spec in specs:
            row=result['branches'][spec['name']]
            assert row['step']==154852+goal and row['training_windows_seen']==6*goal
            assert row['adam_count_at_restore']==[0]*6
            q=model.coordinates(p.base,current,p.cfg,p.columns,spec);tx=model.optimizer(q,spec)
            template=dict(parameters=q,optimizer=tx.init(q),best=q,best_step=jnp.asarray(154852))
            saved,metadata=session.store.restore(spec['name'],dict(c,arm=spec['name']),template)
            assert model.adam_counts(saved['optimizer'])==[goal]*6
