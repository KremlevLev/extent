"""Recipe/source invariants and compiled mathematical checks, no remote jobs."""
from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from extent import HybridForCausalLM,tiny_config
from extent.reset_followup import recipes,prepare,folded130,block_forward,source130
from extent.competitor_recovery import arm_for,initialize,optimizer,with_reset_targets,block_forward as legacy_block
from extent.competitor_tpu import make_step,token_nll
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding
from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer
from extent.model import _attention_source_config


def patient():
    cfg=replace(tiny_config(),mla=replace(tiny_config().mla,implementation='qwen3_gqa'),param_dtype='float32',compute_dtype='float32',logits_dtype='float32',tie_word_embeddings=True,remat_policy='none')
    t=jnp.arange(8,dtype=jnp.int32)[None]%cfg.vocab_size
    base=HybridForCausalLM(cfg).init(jax.random.key(4),t)['params']
    cols=2*int(cfg.hidden_size*cfg.mamba.expand)+2*cfg.mamba.mimo_rank*cfg.mamba.groups*cfg.mamba.d_state
    old=initialize_corrections(base,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=8,head_dim=cfg.mamba.head_dim)
    p=SimpleNamespace(cfg=cfg,base=base,current=apply_corrections(base,old,protected_input_columns=cols),coordinates=old,columns=cols,tc=_attention_source_config(cfg))
    return p,t


def test_seven_plans_match_allocated_hardware_and_sources():
    from scripts.m3q_reset_followup_campaign import contract_for
    for e in range(135,142):
        c=contract_for(e);assert c['horizon']==131072 and c['first_goal_step']==32768 and c['batch_size']==1
        assert len(c['recipes'])==(1 if e==135 else 2)
    assert recipes(135)[0]['start']=='130-exact'
    assert [s['reset_parts'] for s in recipes(136)]==['all','projections']
    assert [s['reset_parts'] for s in recipes(137)]==['dt','mimo_x']
    for a,b in zip(recipes(140),recipes(141)):
        assert a['start']==b['start'] and a['initial_reset'] and b['initial_reset']
        assert not a['reset'] and not b['reset'] # dense dynamics remain trainable
        assert a['ramp_steps']==b['ramp_steps']==0
    ce,kd=recipes(138);assert {k:v for k,v in ce.items() if k not in ('name','alpha')}=={k:v for k,v in kd.items() if k not in ('name','alpha')}


def test_partial_surgery_changes_only_selected_parameter_groups():
    p,t=patient();i=p.cfg.mamba_layer_indices[0];old=arm_for(130,1)
    layer=with_reset_targets(p.base[f'layers_{i}'],p.cfg,i)
    q=initialize(p,old)[str(i)];x=jax.random.normal(jax.random.key(7),(1,8,p.cfg.hidden_size))
    for e in (136,137):
        for s in recipes(e):
            got=block_forward(p.cfg,i,layer,q,x,p.columns,s,jnp.asarray(0.))[0]
            ref=legacy_block(p.cfg,i,layer,q,x,p.columns,old,jnp.asarray(0.))[0]
            np.testing.assert_allclose(got,ref,rtol=1e-5,atol=1e-6)
    s=recipes(136)[0]
    got=block_forward(p.cfg,i,layer,q,x,p.columns,s,jnp.asarray(1.))[0]
    ref=legacy_block(p.cfg,i,layer,q,x,p.columns,old,jnp.asarray(1.))[0]
    np.testing.assert_array_equal(got,ref)


def test_matrix_rank_and_dense_have_identical_initial_effective_functions():
    p,t=patient()
    for index in range(2):
        # Tiny synthetic clean tree is enough to test source branching; real QKV mapper is separately tested.
        a,b=recipes(140)[index],recipes(141)[index]
        ba,qa,_=prepare(p,a,clean=p.base);bb,qb,_=prepare(p,b,clean=p.base)
        def score(base,q,s):return token_nll(p.cfg,base,q,t,p.columns,s,jnp.asarray(0.),block_fn=block_forward)
        va=jax.jit(lambda:score(ba,qa,a))();vb=jax.jit(lambda:score(bb,qb,b))()
        np.testing.assert_allclose(va,vb,rtol=2e-5,atol=2e-6)
        assert all('lora' not in qb[str(i)] for i in p.cfg.mamba_layer_indices)


def test_two_device_compiled_step_matches_single_device_and_shards_frozen_weights():
    p,t=patient();spec=recipes(136)[0];base,q,_=prepare(p,spec)
    mesh=create_v5e_mesh(jax.devices('cpu')[:2]);layout=coordinate_layout(q,mesh);q=jax.device_put(q,layout)
    base=jax.device_put(base,named_sharding_tree(base,mesh));tx=optimizer(q,spec);state,sl=sharded_optimizer(tx,q,layout,mesh)
    step=jax.jit(make_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=block_forward))
    a=jax.block_until_ready(step(q,state,base,{},jax.device_put(t,batch_sharding(mesh)),jnp.asarray(2048.)))
    assert bool(a[2]['finite'])
    device=jax.devices('cpu')[0]
    b=jax.jit(make_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=block_forward))(
        jax.device_put(q,device),jax.device_put(state,device),jax.device_put(base,device),{},jax.device_put(t,device),jnp.asarray(2048.))
    assert any(not v.sharding.is_fully_replicated for v in jax.tree.leaves(base) if v.ndim>=2)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y,rtol=4e-4,atol=3e-6)


def test_folded_reset_source_matches_original_function():
    p,t=patient();q=initialize(p,arm_for(130,1));payload=dict(parameters=q)
    base=folded130(p,payload);s=recipes(138)[0];b,new,_=prepare(p,s,source_payload=payload)
    original=p.base
    original={k:with_reset_targets(v,p.cfg,int(k.split('_')[1])) if k.startswith('layers_') and int(k.split('_')[1]) in p.cfg.mamba_layer_indices else v for k,v in original.items()}
    a=jax.jit(lambda:token_nll(p.cfg,original,q,t,p.columns,arm_for(130,1),jnp.asarray(3434.)))()
    z=jax.jit(lambda:token_nll(p.cfg,b,new,t,p.columns,s,jnp.asarray(0.),block_fn=block_forward))()
    np.testing.assert_allclose(a,z,rtol=2e-5,atol=2e-6)


def test_bf16_full_chain_reset_update_and_dense_dynamics_live_gradients():
    p,t=patient();p.cfg=replace(p.cfg,param_dtype='bfloat16',compute_dtype='bfloat16')
    p.base=jax.tree.map(lambda v:v.astype(jnp.bfloat16),p.base);p.current=p.base;p.tc=replace(p.tc,param_dtype='bfloat16',compute_dtype='bfloat16')
    spec=recipes(141)[1];base,q,_=prepare(p,spec)
    from extent.competitor_tpu import objective
    fun=lambda r:objective(p.cfg,base,r,t,p.columns,spec,jnp.asarray(0.),block_fn=block_forward)[0]
    loss,g=jax.jit(jax.value_and_grad(fun))(q)
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(v)) for v in jax.tree.leaves(g))
    i=str(p.cfg.mamba_layer_indices[0]);dyn=np.asarray(g[i]['dense_mamba']['in_proj']['kernel'])[:,p.columns:]
    assert np.any(dyn!=0), 'Dense dynamics must not be overwritten by a permanent reset mask'
    tx=optimizer(q,spec);step=jax.jit(make_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=block_forward))
    _,_,health=step(q,tx.init(q),base,{},t,jnp.asarray(0.));assert bool(health['finite'])


@pytest.mark.parametrize('healthy',[True,False])
def test_actual_runner_continuation_keeps_source_adam_cursor_and_saves_before_final_probes(monkeypatch,tmp_path,healthy):
    # Run the real orchestration with deterministic device kernels/transport doubles.
    import scripts.m3q_reset_followup_campaign as runner
    import extent.config as config_module
    import extent.reset_followup as math_module
    import extent.competitor_tpu as tpu_module
    import extent.plateau_forensics as diagnostics_module
    import extent.t4_patient as patient_module
    import extent.t4_scale_data as data_module
    import extent.hf_artifact_sync as hub_module
    p,t=patient();p.teacher=p.base
    sourceq=initialize(p,arm_for(130,1));tx=optimizer(sourceq,arm_for(130,1));source_state=tx.init(sourceq)
    source_state=jax.tree.map(lambda x:x+jnp.asarray(7,x.dtype) if jnp.issubdtype(x.dtype,jnp.integer) else x,source_state)
    payload=dict(parameters=sourceq,optimizer=source_state,best=sourceq,best_step=jnp.asarray(3072,jnp.int32))
    reference={k:dict(nll=0.,windows=[0.]*(32 if k=='B' else 64)) for k in 'ABC'}
    runtime=dict(jax='x',jaxlib='x',flax='x',optax='x',matmul_precision=None,prng_impl='threefry2x32')
    pin=dict(reference=reference,source_runtime=dict(versions={k:'x' for k in ('jax','jaxlib','flax','optax')}))
    monkeypatch.setattr(config_module,'load_config',lambda *a:(p.cfg,None))
    monkeypatch.setattr(patient_module,'Patient',lambda *a:p)
    monkeypatch.setattr(math_module,'source130',lambda *a:(payload,pin))
    monkeypatch.setattr(math_module,'Inspector',lambda *a:SimpleNamespace(profile=lambda *a:dict(layers=[])))
    monkeypatch.setattr(math_module,'ReferenceProbe',lambda *a:SimpleNamespace(compare=lambda *a:dict(passed=True)))
    monkeypatch.setattr(hub_module,'artifact_config_from_env',lambda:object())
    monkeypatch.setattr(data_module,'load_data',lambda *a:({k:np.zeros((n,256),np.int32) for k,n in [('A',64),('B',32),('C',64),('FIT',16384)]},{}))
    actual_devices=jax.devices
    monkeypatch.setattr(jax,'devices',lambda backend=None:actual_devices('cpu')[:2] if backend=='gpu' else actual_devices(backend))
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
    args=runner.parser().parse_args(['--experiment','135','--worker','0','--output-dir',str(tmp_path)])
    args.max_wall_hours=11.5;args.max_new_hf_gb=4
    result=runner.run(args);row=result['branches']['RESET130-CONTINUE']
    assert 4 in arities and 6 in arities
    assert row['step']==(3436 if healthy else 3434) and row['source_step']==3434
    assert source_counts and all(all(v==7 for v in counts) for counts in source_counts)
    assert ('persist',0,True) in seen # first proposal inspected before accepting any update
    assert ('save',3436 if healthy else 3434) in seen and row['finalization_status']=='completed'
    assert result['status']==('deadline_partial' if healthy else 'arm_failed')
    assert result['cost'].get('accepted_updates',0)==(2 if healthy else 0)
    if not healthy:assert row['failed']['failed_stages']==['gradients']


def test_clean_qkvo_path_uses_teacher_arrays_no_local_training_and_bf16_storage(monkeypatch):
    from extent.reset_followup import clean_transplant,Inspector
    from extent.qwen3_teacher import Qwen3ForCausalLM
    import extent.qwen3_parity as parity
    p,t=patient();teacher=Qwen3ForCausalLM(p.tc).init(jax.random.key(99),t)['params'];p.teacher=teacher;p.check=lambda:None;p.reader=object()
    def arrays(reader,tc,layer):
        a=teacher[f'layers_{layer}']['self_attn'];prefix=f'model.layers.{layer}.self_attn.'
        return {prefix+name+'.weight':np.asarray(a[name]['kernel']).T for name in ('q_proj','k_proj','v_proj','o_proj')} | {prefix+name+'.weight':np.asarray(a[name]['scale']) for name in ('q_norm','k_norm')}
    monkeypatch.setattr(parity,'load_mixer_arrays',arrays)
    base,reports=clean_transplant(p)
    assert len(reports)==len(p.cfg.mamba_layer_indices)
    for i in p.cfg.mamba_layer_indices:
        assert 'self_attn' not in base[f'layers_{i}']
        assert all(str(v.dtype)=='bfloat16' for v in jax.tree.leaves(base[f'layers_{i}']['mamba']))
    for i in p.cfg.attention_layer_indices:
        for a,b in zip(jax.tree.leaves(base[f'layers_{i}']),jax.tree.leaves(teacher[f'layers_{i}'])):np.testing.assert_array_equal(a,b)
    # The partial-reset profiler must inspect the same ordinary block function.
    spec=recipes(137)[0];b,q,_=prepare(p,spec)
    profile=Inspector(p.cfg,b,p.columns,spec).profile(q,np.asarray(t),2048)
    assert all(row['accepted'] for row in profile['layers'])


def test_gradient_replay_detects_direction_changes_and_handles_large_values():
    from extent.reset_followup import compare_gradients
    g={'x':np.asarray([1e30,-2e30],np.float32)}
    assert compare_gradients(g,g)['passed']
    assert not compare_gradients(g,{'x':-g['x']})['passed']
    assert not compare_gradients(g,{'x':g['x']*2})['passed']
    assert compare_gradients({'x':np.zeros(2)},{'x':np.zeros(2)})['passed']


def test_optional_gradient_output_matches_gradient_used_for_update():
    from extent.competitor_tpu import objective
    p,t=patient();spec=recipes(136)[0];base,q,_=prepare(p,spec);tx=optimizer(q,spec)
    out=jax.jit(make_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=block_forward,return_gradients=True))(q,tx.init(q),base,{},t,jnp.asarray(2048.))
    expected=jax.jit(jax.grad(lambda r:objective(p.cfg,base,r,t,p.columns,spec,jnp.asarray(2048.),block_fn=block_forward)[0]))(q)
    for a,b in zip(jax.tree.leaves(out[3]),jax.tree.leaves(expected)):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-7)


def test_real_legacy_streamed_gradient_probe_matches_full_compiled_step():
    from scripts.m3q_t4_preflight import tiny_patient
    from extent.reset_followup import ReferenceProbe
    p,_,params,t=tiny_patient(jax.devices('cpu')[0])
    p.cfg=replace(p.cfg,param_dtype='float32',compute_dtype='float32');p.tc=replace(p.tc,param_dtype='float32',compute_dtype='float32')
    p.base=jax.tree.map(lambda v:v.astype(jnp.float32),params);p.current=p.base
    p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(p.base,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=8,head_dim=p.cfg.mamba.head_dim)
    q=initialize(p,arm_for(130,1));spec=recipes(135)[0];base,q,_=prepare(p,spec,source_payload={'parameters':q})
    t=np.asarray(t[:,:8]);tx=optimizer(q,spec)
    out=jax.jit(make_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=block_forward,return_gradients=True))(q,tx.init(q),base,{},jnp.asarray(t),jnp.asarray(3434.))
    check=ReferenceProbe(p,jax.devices('cpu')[0]).compare(q,t,3434,float(out[2]['loss']),out[3])
    assert check['passed'],check


@pytest.mark.parametrize('experiment',range(135,142))
def test_actual_entrypoint_accepts_documented_session_budgets(experiment,capsys):
    from scripts.m3q_reset_followup_campaign import main,session_budget,recipes,GPU_EXPERIMENTS
    wall=11.5 if experiment in GPU_EXPERIMENTS else 8.5
    assert session_budget(experiment,wall,60)==((wall-1)*3600,60)
    # Exercise real parent and child CLI validation, before any remote work.
    workers=[None]+(list(range(len(recipes(experiment)))) if experiment in GPU_EXPERIMENTS else [])
    for worker in workers:
        argv=['--experiment',str(experiment),'--max-wall-hours',str(wall),'--save-reserve-minutes','60','--plan-only']
        if worker is not None:argv+=['--worker',str(worker)]
        assert main(argv)['experiment']==experiment
    capsys.readouterr()


@pytest.mark.parametrize('experiment,wall,reserve',[
    (135,12.1,60),(139,9.1,60),(135,float('nan'),60),
    (135,float('inf'),60),(139,0,60),(135,11.5,float('nan')),
    (135,11.5,29),(135,1,60),
])
def test_session_budget_rejects_invalid_limits(experiment,wall,reserve):
    from scripts.m3q_reset_followup_campaign import session_budget
    with pytest.raises(ValueError):session_budget(experiment,wall,reserve)


@pytest.mark.parametrize('experiment,wall',[(135,11.5),(139,8.5)])
def test_real_session_constructor_uses_hardware_budget(experiment,wall,tmp_path,monkeypatch):
    import scripts.m3q_reset_followup_campaign as runner
    import extent.recovery_durability as durability
    import extent.hf_artifact_sync as sync
    monkeypatch.setattr(sync,'restore_artifact',lambda *a:False)
    monkeypatch.setattr(durability,'RecoveryCheckpointStore',lambda *a:object())
    args=runner.parser().parse_args(['--experiment',str(experiment),'--output-dir',str(tmp_path),'--max-wall-hours',str(wall)])
    s=runner.session(args,{'test':True},None)
    assert s.deadline-s.begun==pytest.approx((wall-1)*3600)
    assert s.wall-s.begun==pytest.approx(wall*3600)
