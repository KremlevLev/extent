from dataclasses import replace
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM
from tests.test_decoder_recovery_157 import fixture as source_fixture
from extent import hidden_distillation_159 as distill
from extent.competitor_recovery import head_tile,optimizer
from extent.plateau_forensics import norm,mixer,mlp

def fixture(worker):
    p,t,q,_,_=source_fixture();spec=distill.recipes(159)[worker]
    spec=dict(spec,align_layers=list(range(p.cfg.num_layers)))
    teacher_cfg=replace(p.cfg,attention_layer_indices=tuple(range(p.cfg.num_layers)))
    teacher=HybridForCausalLM(teacher_cfg).init(jax.random.key(159),t)['params']
    teacher=dict(teacher,embed_tokens=p.base['embed_tokens'],norm=p.base['norm'])
    return p,t[:,:8],q,spec,jax.device_get(teacher)

def objective(p,t,q,spec,tp):
    h=p.base['embed_tokens']['embedding'][t];th=h;aux=0.
    for i in range(p.cfg.num_layers):
        layer=tp[f'layers_{i}']
        th=th+mixer(norm(th,layer['input_layernorm'],p.cfg),layer,p.cfg,i,p.tc)
        th=th+mlp(norm(th,layer['post_attention_layernorm'],p.cfg),layer,p.cfg)
        h,_=distill.block_forward(p.cfg,i,p.base[f'layers_{i}'],q[str(i)],h,p.columns,spec,jnp.asarray(1.,jnp.float32))
        if i in spec['align_layers']:aux+=distill.normalized_hidden_mse(h,th)/len(spec['align_layers'])
    loss,parts=head_tile(p.cfg,h[:,:-1],p.base['embed_tokens']['embedding'],p.base['norm']['scale'],t[:,1:],th[:,:-1],spec['alpha'],t.shape[1]-1)
    return loss+spec['hidden_beta']*aux

@pytest.mark.parametrize('worker',[0,1])
def test_streamed_intermediate_cotangents_match_whole_chain(worker,tmp_path):
    p,t,q,spec,tp=fixture(worker);tx=optimizer(q,spec)
    engine=distill.DistillationStep(p.cfg,p.tc,p.columns,spec,tx,return_gradients=True)
    before=jax.tree.map(np.array,tp)
    result=engine(q,tx.init(q),p.base,tp,t,jnp.asarray(118304.,jnp.float32))
    loss,g=jax.jit(jax.value_and_grad(lambda q:objective(p,t,q,spec,tp)))(q)
    assert bool(result[2]['finite']) and float(result[2]['kl'])>=-1e-5
    assert abs(float(loss)-float(result[2]['loss']))<.02
    from extent.reset_followup import compare_gradients
    audit=compare_gradients(g,result[3]);assert audit['passed'],audit
    assert distill.adam_counts(result[1])==[1]*6
    if worker:assert result[2]['hidden_mse']>0 and len(result[2]['hidden_by_layer'])==p.cfg.num_layers
    else:assert result[2]['hidden_contribution']==0
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(tp)):np.testing.assert_array_equal(a,b)
    from extent.recovery_durability import RecoveryCheckpointStore
    payload=dict(parameters=result[0],optimizer=result[1],best=q,best_step=jnp.asarray(118304,jnp.int32))
    store=RecoveryCheckpointStore(tmp_path,'test');store.save('arm',{'test':True},payload,118305)
    restored,_=store.restore('arm',{'test':True},payload)
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(payload)):np.testing.assert_array_equal(a,b)

def test_normalized_targets_scale_and_teacher_detachment():
    x=jnp.asarray([[[1e25,2e25],[3e25,4e25],[5e25,6e25]]],jnp.float32)
    assert float(distill.normalized_hidden_mse(x,x*2))<1e-8
    assert np.isfinite(jax.grad(lambda h:distill.normalized_hidden_mse(h,-x))(x)).all()
    assert np.all(np.asarray(jax.grad(lambda target:distill.normalized_hidden_mse(x,target))(-x))==0)

def test_contracts_change_only_hidden_target_and_have_large_horizon():
    from scripts.m3q_hidden_distillation_159 import contract_for,session_budget,parser
    a,b=distill.recipes(159)
    assert {k:v for k,v in a.items() if k not in ('name','hidden_beta')}=={k:v for k,v in b.items() if k not in ('name','hidden_beta')}
    assert a['alpha']==b['alpha']==.5 and b['hidden_beta']==.1
    assert len(contract_for(159,1)['recipes'])==1
    assert session_budget(159,11.8,18)[0]==11.8*3600-1080
    assert parser().parse_args([]).max_steps==262144

@pytest.mark.parametrize('bad_replay',[False,True])
def test_actual_worker_runs_hidden_step_and_rejects_bad_source(tmp_path,monkeypatch,bad_replay):
    import sys,json,time
    from importlib.metadata import version
    from scripts import m3q_hidden_distillation_159 as runner
    from extent.competitor_tpu import token_nll
    from extent.recovery_durability import RecoveryCheckpointStore
    import extent.config,extent.reset_long_patient,extent.t4_scale_data,extent.hf_artifact_sync,extent.quality_benchmark_156,extent.competitor_tpu
    p,t,q,spec,tp=fixture(1);p.teacher=tp;cursor=118304
    c=runner.contract_for(159,1);c['recipes']=[spec];c['horizon']=1;c['eval_steps']=[0]
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    ref=float(token_nll(p.cfg,p.base,q,t,p.columns,spec,jnp.asarray(cursor),block_fn=distill.block_forward).mean())
    pin=dict(source_step=cursor,adam_step=0,best_step=cursor,runtime=runtime,data={},reference={k:{'windows':[ref+int(bad_replay)]} for k in 'ABC'})
    payload=dict(parameters=q,best=q,best_step=jnp.asarray(cursor,jnp.int32))
    monkeypatch.setattr(distill,'source_pin',lambda:pin)
    monkeypatch.setattr(distill,'restore_source',lambda *a:(payload,pin));monkeypatch.setattr(distill,'base_coordinates',lambda *a:{})
    monkeypatch.setattr(distill,'prepare',lambda *a:(p.base,q,cursor))
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}))
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    monkeypatch.setattr(extent.competitor_tpu,'training_batch',lambda *a:np.asarray(t))
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:extent.hf_artifact_sync.HubArtifactConfig('offline','dataset',''))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    def data(path,*args):
        path.mkdir(parents=True,exist_ok=True);np.savez(path/'fresh-panels.npz',english_stories=np.arange(40960,dtype=np.int32)%p.cfg.vocab_size)
        (path/'data-manifest.json').write_text('{}')
    monkeypatch.setattr(extent.quality_benchmark_156,'prepare_data',data)
    devices=jax.devices;monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu') if platform=='gpu' else devices(platform))
    monkeypatch.setattr(runner,'contract_for',lambda *a:c)
    class Session:
        def __init__(self):
            self.result=dict(branches={},runtime=runtime,cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0),pending_slots=[])
            self.store=RecoveryCheckpointStore(tmp_path/'states','159');self.wall=time.monotonic()+3600;self.deadline=self.wall-1800
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=1:raise TimeoutError('test work deadline')
        def emit(self,*a):pass
        def persist(self,**kw):return True
        def checkpoint(self,n,v,step):self.store.save(n,dict(c,arm=n),v,step)
    session=Session();monkeypatch.setattr(runner,'session',lambda *args:session)
    args=SimpleNamespace(experiment=159,worker=1,max_steps=262144,cache_dir=str(tmp_path/'cache'))
    if bad_replay:
        with pytest.raises(RuntimeError):runner.run(args)
        assert session.result['cost'].get('accepted_updates',0)==0
    else:
        result=runner.run(args);row=next(iter(result['branches'].values()))
        assert result['cost']['accepted_updates']==1 and result['cost']['teacher_input_tokens']==t.size
        assert result['teacher_shared_head_verified'] and row['step']==118305
        assert row['first_update_health']['health']['hidden_mse']>0
        assert row['adam_count_at_restore']==[0]*6
        assert session.store.metadata(spec['name'],dict(c,arm=spec['name']))['step']==118305

@pytest.mark.parametrize('fail_worker',[False,True])
def test_parent_spawns_two_isolated_gpus_and_preserves_failure(tmp_path,monkeypatch,fail_worker):
    import json
    from scripts import m3q_hidden_distillation_159 as runner
    from extent.campaign_checkpoint import write_json_atomic
    import extent.hf_artifact_sync,extent.hf_download_retry,extent.reset_quiet
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:object())
    uploaded=[];envs=[]
    def upload(path,*args,**kw):uploaded.append(json.loads(path.read_text()))
    monkeypatch.setattr(extent.hf_artifact_sync,'upload_artifact',upload)
    monkeypatch.setattr(extent.hf_download_retry,'install_download_retries',lambda:None)
    class Notice:
        def __init__(self,*a):pass
        def emit(self,*a):pass
    monkeypatch.setattr(extent.reset_quiet,'ParentNotifier',Notice)
    class Process:
        def __init__(self,cmd,env,**kw):
            envs.append(env['CUDA_VISIBLE_DEVICES']);i=int(cmd[cmd.index('--worker')+1]);self.failed=fail_worker and i==1
            out=Path(cmd[cmd.index('--output-dir')+1]);c=runner.contract_for(159,i)
            write_json_atomic(out/f'extent-m3q-hidden-distill-159-worker-{i}.json',dict(contract=c,status='failed' if self.failed else 'completed',branches={}))
        def poll(self):return 1 if self.failed else 0
    from pathlib import Path
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    args=['--output-dir',str(tmp_path/'out'),'--cache-dir',str(tmp_path/'cache'),'--checkpoint-dir',str(tmp_path/'states'),'--no-telegram']
    if fail_worker:
        with pytest.raises(RuntimeError):runner.main(args)
        assert uploaded[-1]['status']=='failed'
    else:
        r=runner.main(args);assert r['status']=='completed' and len(r['workers'])==2
    assert envs==['0','1']
