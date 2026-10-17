"""Actual full-decoder gradients and runner interfaces on tiny CPU models."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from types import SimpleNamespace
from extent import HybridForCausalLM,tiny_config
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.model import _attention_source_config
from extent.competitor_recovery import arm_for,initialize,optimizer as source_optimizer
from extent.reset_followup import prepare as prepare139,recipes as recipes139
from extent.competitor_tpu import token_nll as nll139
from extent.reset_followup import block_forward as block139
from extent.full_decoder_reset import recipes,prepare,optimizer,objective,token_nll,make_step,Inspector,parameter_group


def patient():
    cfg=replace(tiny_config(),mla=replace(tiny_config().mla,implementation='qwen3_gqa'),param_dtype='float32',compute_dtype='float32',logits_dtype='float32',tie_word_embeddings=True,remat_policy='none')
    t=jnp.arange(8,dtype=jnp.int32)[None]%cfg.vocab_size
    base=HybridForCausalLM(cfg).init(jax.random.key(4),t)['params']
    cols=2*int(cfg.hidden_size*cfg.mamba.expand)+2*cfg.mamba.mimo_rank*cfg.mamba.groups*cfg.mamba.d_state
    old=initialize_corrections(base,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=8,head_dim=cfg.mamba.head_dim)
    p=SimpleNamespace(cfg=cfg,base=base,current=apply_corrections(base,old,protected_input_columns=cols),coordinates=old,columns=cols,tc=_attention_source_config(cfg))
    return p,t


def setup():
    p,t=patient();source={'parameters':initialize(p,arm_for(130,1))};spec=recipes(142)[0]
    base,q,cursor=prepare(p,spec,source)
    return p,t,source,spec,base,q,cursor


def test_full_decoder_starts_at_same_function_as139_dense_and_covers_every_weight():
    p,t,source,spec,base,q,cursor=setup()
    b,r,_=prepare139(p,recipes139(139)[1],source)
    actual=jax.jit(lambda q:token_nll(p.cfg,base,q,t,p.columns,spec,jnp.asarray(0.)))(q)
    ref=jax.jit(lambda q:nll139(p.cfg,b,q,t,p.columns,recipes139(139)[1],jnp.asarray(0.),block_fn=block139))(r)
    np.testing.assert_allclose(actual,ref,rtol=2e-5,atol=2e-5)
    assert cursor==0 and not spec['reset'] and spec['alpha']==0
    assert sum(v.size for v in jax.tree.leaves(q))==sum(v.size for v in jax.tree.leaves(base))+sum(v.size for k,v in jax.tree.leaves_with_path(q) if 'pregate_weight' in [getattr(x,'key',None) for x in k])
    assert {parameter_group(path) for path,v in jax.tree.leaves_with_path(q)}=={'mamba','mlp','gqa','decoder_norm','pregate_norm','tied_embedding'}
    assert all(v.dtype==jnp.float32 for v in jax.tree.leaves(q))


def test_real_eight_device_step_updates_all_groups_and_preserves_exact_resume(tmp_path):
    from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding
    from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer
    from extent.recovery_durability import RecoveryCheckpointStore
    p,t,source,spec,base,q,_=setup();mesh=create_v5e_mesh(jax.devices('cpu')[:8])
    assert mesh.devices.size==8
    layout=coordinate_layout(q,mesh);q=jax.device_put(q,layout);base=jax.device_put(base,named_sharding_tree(base,mesh));tx=optimizer(q,spec)
    state,state_layout=sharded_optimizer(tx,q,layout,mesh)
    step=jax.jit(make_step(p.cfg,p.tc,p.columns,spec,tx))
    proposed,after,h=step(q,state,base,{},jax.device_put(t,batch_sharding(mesh)),jnp.asarray(0.))
    assert bool(h['finite'])
    for name,row in h['groups'].items():
        assert float(row['grad_max_abs'])>0,name
        assert int(row['master_changed'])>0,name
    losses=token_nll(p.cfg,base,q,t,p.columns,spec,jnp.asarray(0.))
    np.testing.assert_allclose(float(h['ce']),float(jnp.mean(losses)),rtol=1e-5)
    payload=dict(parameters=proposed,optimizer=after,best=q,best_step=jnp.asarray(0,jnp.int32))
    store=RecoveryCheckpointStore(tmp_path,'offline',None);store.save('FULL',{'experiment':142},payload,1)
    restored=store.restore('FULL',{'experiment':142},payload)[0]
    for a,b in zip(jax.tree.leaves(payload),jax.tree.leaves(restored)):np.testing.assert_array_equal(a,b)
    next1=step(jax.device_put(proposed,layout),jax.device_put(after,state_layout),base,{},jax.device_put(t,batch_sharding(mesh)),jnp.asarray(1.))
    next2=step(jax.device_put(restored['parameters'],layout),jax.device_put(restored['optimizer'],state_layout),base,{},jax.device_put(t,batch_sharding(mesh)),jnp.asarray(1.))
    for a,b in zip(jax.tree.leaves(next1),jax.tree.leaves(next2)):np.testing.assert_array_equal(a,b)


def test_profiler_uses_learned_embedding_mlp_and_norm_values():
    p,t,source,spec,base,q,_=setup()
    q['_global']['embed_tokens']['embedding']*=1.1
    q['0']['dense_layer']['mlp']['down_proj']['kernel']*=.9
    report=Inspector(p.cfg,base,p.columns,spec).profile(q,np.asarray(t),1)
    assert all(v['accepted'] for v in report['layers'])


def test_runner_defaults_plan_and_budget(capsys):
    from scripts.m3q_full_decoder_142_campaign import main,session_budget
    c=main(['--experiment','142','--plan-only'])
    assert c['experiment']==142 and len(c['recipes'])==1 and c['horizon']==131072
    assert c['recipes'][0]['full_decoder'] and c['recipes'][0]['start']=='130-folded'
    assert session_budget(142,8.5,60)==(27000,60)
    with pytest.raises(ValueError):main(['--experiment','142','--max-wall-hours','11.5','--plan-only'])
    capsys.readouterr()


@pytest.mark.parametrize('healthy',[True,False])
def test_actual_full_decoder_runner_accepts_or_rejects_first_proposal(monkeypatch,tmp_path,healthy):
    # Run the real orchestration with deterministic device kernels/transport doubles.
    import scripts.m3q_full_decoder_142_campaign as runner
    import extent.config as config_module
    import extent.full_decoder_reset as math_module
    import extent.reset_followup as source_module
    import extent.full_decoder_reset as tpu_module
    import extent.plateau_forensics as diagnostics_module
    import extent.t4_patient as patient_module
    import extent.t4_scale_data as data_module
    import extent.hf_artifact_sync as hub_module
    p,t=patient();p.teacher=p.base
    sourceq=initialize(p,arm_for(130,1));tx=source_optimizer(sourceq,arm_for(130,1));source_state=tx.init(sourceq)
    source_state=jax.tree.map(lambda x:x+jnp.asarray(7,x.dtype) if jnp.issubdtype(x.dtype,jnp.integer) else x,source_state)
    payload=dict(parameters=sourceq,optimizer=source_state,best=sourceq,best_step=jnp.asarray(3072,jnp.int32))
    reference={k:dict(nll=0.,windows=[0.]*(32 if k=='B' else 64)) for k in 'ABC'}
    runtime=dict(jax='x',jaxlib='x',flax='x',optax='x',matmul_precision=None,prng_impl='threefry2x32')
    pin=dict(reference=reference,source_runtime=dict(versions={k:'x' for k in ('jax','jaxlib','flax','optax')}))
    monkeypatch.setattr(config_module,'load_config',lambda *a:(p.cfg,None))
    monkeypatch.setattr(patient_module,'Patient',lambda *a:p)
    monkeypatch.setattr(source_module,'source130',lambda *a:(payload,pin))
    monkeypatch.setattr(math_module,'Inspector',lambda *a:SimpleNamespace(profile=lambda *a:dict(layers=[])))
    monkeypatch.setattr(source_module,'ReferenceProbe',lambda *a:SimpleNamespace(compare=lambda *a:dict(passed=True)))
    monkeypatch.setattr(hub_module,'artifact_config_from_env',lambda:object())
    monkeypatch.setattr(data_module,'load_data',lambda *a:({k:np.zeros((n,256),np.int32) for k,n in [('A',64),('B',32),('C',64),('FIT',16384)]},{}))
    actual_devices=jax.devices
    monkeypatch.setattr(jax,'devices',lambda backend=None:actual_devices('cpu')[:8] if backend=='tpu' else actual_devices(backend))
    arities=[]
    def guarded(function):
        def call(*args):
            arities.append(len(args))
            if len(args)==4:assert 'embed_tokens' in args[0] # frozen base is explicit, never a giant closed-over constant
            return function(*args)
        return call
    monkeypatch.setattr(diagnostics_module,'GuardedJit',guarded)
    seen=[];source_counts=[]
    def make(*args,**kwargs):
        def step(q,state,*rest):
            source_counts.append([int(np.asarray(v)) for path,v in jax.tree.leaves_with_path(state) if getattr(path[-1],'name',None)=='count'])
            return jax.tree.map(lambda x:x+1e-6,q),state,dict(finite=healthy,loss=1.,ce=1.,kl=0.,stages={'gradients':{'finite':healthy}}),{}
        return step
    monkeypatch.setattr(tpu_module,'make_step',make)
    monkeypatch.setattr(tpu_module,'token_nll',lambda cfg,base,q,t,*a,**k:jnp.zeros((t.shape[0],t.shape[1]-1)))
    class S:
        def __init__(self,args,contract,hub):
            self.args=args;self.wall=1e30;self.deadline=1e29;self.result=dict(runtime=runtime,status='preflight',branches={},pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0));self.store=SimpleNamespace(restore=lambda *a:None)
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=2 and self.deadline!=self.wall-180:raise TimeoutError('test work limit')
        def persist(self,**kw):
            seen.append(('persist',self.result['cost'].get('accepted_updates',0),any('first_update_health' in v for v in self.result['branches'].values())))
            return True
        def emit(self,*a):pass
        def checkpoint(self,name,payload,step):seen.append(('save',step))
    monkeypatch.setattr(runner,'session',S)
    args=runner.parser().parse_args(['--experiment','142','--output-dir',str(tmp_path)])
    args.max_wall_hours=8.5;args.max_new_hf_gb=4
    result=runner.run(args);row=result['branches']['RESET130-FULL-DECODER']
    assert 4 in arities and 6 in arities
    assert row['step']==(2 if healthy else 0) and row['source_step']==0
    assert source_counts and all(all(v==0 for v in counts) for counts in source_counts)
    assert ('persist',0,True) in seen # first proposal inspected before accepting any update
    assert ('save',2 if healthy else 0) in seen and row['finalization_status']=='completed'
    assert result['status']==('deadline_partial' if healthy else 'arm_failed')
    assert result['cost'].get('accepted_updates',0)==(2 if healthy else 0)
    if not healthy:assert row['failed']['failed_stages']==['gradients']



def test_newly_unlocked_backbone_rates_do_not_change_shared139_rates():
    spec=recipes(142)[0]
    q={'0':{'dense_layer':{'mamba':{'norm':{'scale':jnp.ones(1)}},'mlp':{'kernel':jnp.ones(1)},'input_layernorm':{'scale':jnp.ones(1)}},'pregate_weight':jnp.ones(1)},'_global':{'embed_tokens':{'embedding':jnp.ones(1)}}}
    tx=optimizer(q,spec);updates,_=tx.update(jax.tree.map(jnp.ones_like,q),tx.init(q),q)
    checks=[(updates['0']['dense_layer']['mamba']['norm']['scale'],3e-4),
        (updates['0']['dense_layer']['mlp']['kernel'],3e-5),
        (updates['0']['dense_layer']['input_layernorm']['scale'],1e-5),
        (updates['0']['pregate_weight'],1e-4),
        (updates['_global']['embed_tokens']['embedding'],3e-5)]
    for actual,peak in checks:np.testing.assert_allclose(actual,-peak/256,rtol=2e-5)


def test_real_bf16_full_decoder_gradient_and_head_evaluation_agree():
    p,t=patient();p.cfg=replace(p.cfg,param_dtype='bfloat16',compute_dtype='bfloat16');p.tc=replace(p.tc,param_dtype='bfloat16',compute_dtype='bfloat16')
    p.base=jax.tree.map(lambda v:v.astype(jnp.bfloat16),p.base);p.current=p.base
    source={'parameters':initialize(p,arm_for(130,1))};spec=recipes(142)[0];base,q,_=prepare(p,spec,source);tx=optimizer(q,spec)
    proposed,after,h=jax.jit(make_step(p.cfg,p.tc,p.columns,spec,tx))(q,tx.init(q),base,{},t,jnp.asarray(0.))
    assert bool(h['finite'])
    assert all(float(v['grad_max_abs'])>0 for v in h['groups'].values())
    nll=jax.jit(lambda q:token_nll(p.cfg,base,q,t,p.columns,spec,jnp.asarray(0.)))(q)
    np.testing.assert_allclose(float(h['ce']),float(jnp.mean(nll)),atol=.01)
