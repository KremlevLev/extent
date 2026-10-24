from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.test_rank_growth_153 import mature_source
from extent.rank_growth_153 import grow_coordinates,block_forward as source_block
from extent import decoder_recovery_157 as d
from extent.competitor_tpu import token_nll

def fixture():
    p,t,q=mature_source();s=d.recipes(157)[1];q=grow_coordinates(p.base,q,p.cfg,256)
    # Nonzero added rank factors: do not test only a zero-effect growth fixture.
    for i in p.cfg.mamba_layer_indices:q[str(i)]['extra_lora']['out_proj']['b']=jnp.full_like(q[str(i)]['extra_lora']['out_proj']['b'],.003)
    dense=d.dense_coordinates(p.base,q,p.cfg,p.columns,s)
    return p,t,q,dense,s

def test_materialized_decoder_replays_nonzero_growth_and_has_real_dense_gradients(tmp_path):
    from extent.recovery_durability import RecoveryCheckpointStore
    p,t,source,q,s=fixture()
    old=jax.jit(lambda q:token_nll(p.cfg,p.base,q,t,p.columns,s,jnp.asarray(118304.),block_fn=source_block))(source)
    new=jax.jit(lambda q:token_nll(p.cfg,p.base,q,t,p.columns,s,jnp.asarray(118304.),block_fn=d.block_forward))(q)
    np.testing.assert_allclose(old,new,atol=.003,rtol=0)
    tx=d.optimizer(q,s);state=tx.init(q);engine=d.make_kernel(p.cfg,p.tc,p.columns,s,tx,tpu=True)
    result=engine(q,state,p.base,{},t,jnp.asarray(118304.,jnp.float32))
    assert bool(result[2]['finite']) and engine.audit['passed']
    g=result[3]
    for i in p.cfg.attention_layer_indices:
        assert any(np.any(np.asarray(v)) for v in jax.tree.leaves(g[str(i)]['dense_layer']))
    for i in range(p.cfg.num_layers):
        assert any(np.any(np.asarray(v)) for v in jax.tree.leaves(g[str(i)]['dense_layer']['mlp']))
    assert d.adam_counts(result[1])==[1]*6
    # Full dense state must actually serialize/restore, including optimizer and best.
    payload=dict(parameters=result[0],optimizer=result[1],best=q,best_step=jnp.asarray(118304,jnp.int32))
    store=RecoveryCheckpointStore(tmp_path,'offline');store.save('dense',{'test':True},payload,118305)
    restored,_=store.restore('dense',{'test':True},payload)
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(payload)):np.testing.assert_array_equal(a,b)

def test_wall_budget_and_immutable_source():
    from scripts.m3q_decoder_recovery_157_campaign import session_budget,contract_for,parser
    work,reserve=session_budget(157,9,18)
    assert work==31320 and reserve==18
    with pytest.raises(ValueError):session_budget(157,9,17)
    c=contract_for(157);assert len(c['recipes'])==2 and all(s['mode']=='fresh-adam' for s in c['recipes'])
    assert d.source_pin()['source_step']==118304
    assert parser().parse_args([]).max_steps==262144

def test_quality_followup_is_separate_frozen_source_contract():
    from scripts.m3q_quality_followup_158 import contract
    c=contract();assert c['source_steps']==[80260,118304] and c['questions_per_task']==256
    assert c['source_pins'][0]!=c['source_pins'][1]

@pytest.mark.parametrize('bad_replay',[False,True])
def test_actual_two_arm_runner_checkpoint_and_replay_rejection(tmp_path,monkeypatch,bad_replay):
    import json,time,sys
    from types import SimpleNamespace
    from importlib.metadata import version
    from scripts import m3q_decoder_recovery_157_campaign as runner
    from extent.recovery_durability import RecoveryCheckpointStore
    import extent.config,extent.reset_long_patient,extent.t4_scale_data,extent.hf_artifact_sync,extent.quality_benchmark_156
    p,t,q,dense,s=fixture();cursor=118304;c=runner.contract_for(157);c['horizon']=1;c['eval_steps']=[0]
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    refs=np.asarray(token_nll(p.cfg,p.base,q,t,p.columns,s,jnp.asarray(cursor),block_fn=source_block)).mean(axis=1)
    pin=dict(source_step=cursor,adam_step=0,best_step=cursor,runtime=runtime,data={},reference={k:dict(windows=(refs+int(bad_replay)).tolist()) for k in 'ABC'})
    monkeypatch.setattr(d,'source_pin',lambda:pin)
    payload=dict(parameters=q,best=q,best_step=jnp.asarray(cursor))
    monkeypatch.setattr(d,'restore_source',lambda *a:(payload,pin));monkeypatch.setattr(d,'base_coordinates',lambda *a:{})
    monkeypatch.setattr(d,'prepare',lambda patient,spec,*args:(p.base,dense if spec['decoder_dense'] else q,cursor))
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}))
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    import extent.competitor_tpu
    monkeypatch.setattr(extent.competitor_tpu,'training_batch',lambda *a:np.asarray(t))
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:extent.hf_artifact_sync.HubArtifactConfig('offline','dataset',''))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    def data(path,*args):
        path.mkdir(parents=True,exist_ok=True);np.savez(path/'fresh-panels.npz',english_stories=np.arange(40960,dtype=np.int32)%p.cfg.vocab_size)
        (path/'data-manifest.json').write_text('{}')
    monkeypatch.setattr(extent.quality_benchmark_156,'prepare_data',data)
    devices=jax.devices
    class Devices(list):
        def __len__(self):return 8
    monkeypatch.setattr(jax,'devices',lambda platform=None:Devices(devices('cpu')) if platform=='tpu' else devices(platform))
    monkeypatch.setattr(runner,'contract_for',lambda *args:c)
    class Session:
        def __init__(self):
            self.result=dict(branches={},runtime=runtime,cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0),pending_slots=[])
            self.store=RecoveryCheckpointStore(tmp_path/'states','157');self.wall=time.monotonic()+3600;self.deadline=self.wall-1800
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=2:raise TimeoutError('test work deadline')
        def emit(self,*a):pass
        def persist(self,**kw):return True
        def checkpoint(self,n,v,step):self.store.save(n,dict(c,arm=n),v,step)
    session=Session();monkeypatch.setattr(runner,'session',lambda *args:session)
    args=SimpleNamespace(experiment=157,worker=None,max_steps=262144,cache_dir=str(tmp_path/'cache'))
    if bad_replay:
        with pytest.raises(RuntimeError):runner.run(args)
        assert session.result['cost'].get('accepted_updates',0)==0
    else:
        r=runner.run(args)
        assert r['cost']['accepted_updates']==2 and len(r['branches'])==2
        for name,row in r['branches'].items():
            assert row['source_replay']['passed'] and row['step']==118305
            assert row['adam_count_at_restore']==[0]*6
            assert session.store.metadata(name,dict(c,arm=name))['step']==118305

def test_fresh_nll_comparison_requires_complete_panels():
    from scripts.m3q_quality_followup_158 import comparisons
    workers={'old':{'nll':{'en':{'windows':[5.,7.],'expected':2}}},'new':{'nll':{'en':{'windows':[4.,6.],'expected':2}}}}
    mc,nll=comparisons(workers);assert nll['en']['new_minus_old']==-1
    workers['new']['nll']['en']['windows'].pop()
    assert comparisons(workers)[1]=={}

@pytest.mark.skipif(len(jax.devices())!=8,reason='separate eight-device CPU engineering invocation')
def test_eight_device_dense_gradient_and_adam_layouts():
    from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding,count_partitioned_arrays
    from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer
    p,t,source,q,s=fixture();mesh=create_v5e_mesh();layout=coordinate_layout(q,mesh)
    q=jax.device_put(q,layout);base=jax.device_put(p.base,named_sharding_tree(p.base,mesh));tx=d.optimizer(q,s)
    state,sl=sharded_optimizer(tx,q,layout,mesh)
    assert count_partitioned_arrays(q)[0]>0 and count_partitioned_arrays(state)[0]>0
    engine=d.make_kernel(p.cfg,p.tc,p.columns,s,tx,tpu=True)
    result=engine(q,state,base,{},jax.device_put(t,batch_sharding(mesh)),jnp.asarray(118304.,jnp.float32))
    assert bool(result[2]['finite']) and d.adam_counts(result[1])==[1]*6

def test_quality_worker_runs_fresh_panels_questions_and_generations(tmp_path,monkeypatch):
    import sys,json,time,hashlib
    from types import SimpleNamespace
    from importlib.metadata import version
    from scripts import m3q_quality_followup_158 as runner
    import extent.quality_benchmark_156 as quality
    import extent.hf_artifact_sync,extent.t4_scale_data
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    pin=dict(source_step=80260,runtime=runtime,data={},reference={k:{'windows':[1.]} for k in 'ABC'})
    monkeypatch.setattr(runner,'old_pin',lambda:pin)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:object())
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.zeros((1,256),np.int32) for k in 'ABC'},{}))
    devices=jax.devices
    monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu') if platform=='gpu' else devices(platform))
    class Engine:
        def token_nll(self,t):return np.ones((1,t.shape[1]-1),np.float32)
        def score_pairs(self,*a,**kw):return [{'loglikelihood':0.},{'loglikelihood':-1.}]
        def generate(self,tok,prompt,mode,i):return {'prompt':prompt,'text':'fixture'}
    monkeypatch.setattr(quality,'load_engine',lambda *a:Engine())
    def data(path,*args):
        path.mkdir(parents=True);arr=np.zeros(40960,np.int32);np.savez(path/'fresh-panels.npz',english_stories=arr)
        questions=[dict(id=task+':0',task=task,gold=0,subject='',pairs=[['Q','A'],['Q','B']],norms=[1,1]) for task in quality.TASKS]
        (path/'questions.json').write_text(json.dumps(questions))
        manifest=dict(questions_sha256=quality.digest_json(questions),provenance={'english_stories':{'token_sha256':hashlib.sha256(arr.tobytes()).hexdigest()}})
        (path/'data-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(runner,'prepare_data',data)
    args=SimpleNamespace(worker='old',output_dir=str(tmp_path/'out'),cache_dir=str(tmp_path/'cache'),deadline_unix=time.time()+60)
    r=runner.worker(args)
    assert r['status']=='completed' and len(r['items'])==6 and len(r['generations'])==24
    assert len(r['nll']['english_stories']['windows'])==128 and r['source_replay']['passed']
    assert r['nll']['english_stories_context_1024']['targets_per_window']==127

def test_checkpoint_space_rejects_before_training(tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace
    from scripts.m3q_decoder_recovery_157_campaign import checkpoint_storage
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=1024))
    with pytest.raises(OSError,match='50GB'):checkpoint_storage(SimpleNamespace(checkpoint_dir=str(tmp_path/'states')))
