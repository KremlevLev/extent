"""EXP125--127 causal bounds, real chain rule, durability: tiny CPU science only."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM
from scripts.m3q_t4_preflight import tiny_patient
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.dynamic_recovery import (Engine,initialize,optimizer,arm_for,block_forward,
    relative_factor,limit_absolute,token_log_rms,RAMP_STEPS)
from extent.t4_scale_recovery import head_tile,update
from extent.chunked_checkpoint import ChunkedCheckpointStore


def fixture(spec):
    p,_,_,t=tiny_patient(jax.devices('cpu')[0])
    p.cfg=replace(p.cfg,param_dtype='float32',compute_dtype='float32')
    t=np.asarray(t[:,:16]);base=HybridForCausalLM(p.cfg).init(jax.random.key(125),t)['params']
    p.base=base;p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(base,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=4,head_dim=p.cfg.mamba.head_dim)
    p.coordinates=jax.tree.map(lambda x:x+.0001,p.coordinates)
    p.current=jax.device_get(apply_corrections(base,p.coordinates,protected_input_columns=p.columns))
    q=initialize(p,spec)
    return p,q,t


def test_token_bounds_preserve_small_vectors_and_are_causal():
    x=jnp.ones((1,3,4))*100
    m=jnp.array([[[10.,10,10,10],[100.,100,100,100],[1e20,1e20,1e20,1e20]]])
    factor=relative_factor(x,m,.5,jnp.array(1.))
    np.testing.assert_allclose(factor[:,:2,0],[[1,.5]],rtol=1e-6)
    bounded=m*factor
    assert np.max(np.asarray(jnp.exp(token_log_rms(bounded))))<=50.001
    changed=m.at[:,2].set(1e30)
    np.testing.assert_array_equal(relative_factor(x,m,.5,1.)[:,:2],relative_factor(x,changed,.5,1.)[:,:2])
    out,_=limit_absolute(m,10.,1.)
    assert np.max(np.asarray(jnp.exp(token_log_rms(out))))<=10.001
    # The geometric transition has no final enormous coefficient jump.
    before=relative_factor(x,m,.5,(RAMP_STEPS-1)/RAMP_STEPS)
    assert float(before[0,2,0]/factor[0,2,0])<1.1


def test_bound_true_derivative_and_zero_finite():
    x=jnp.array([[[1.,2.,3.]]]);m=jnp.array([[[1e4,2e4,3e4]]])
    fn=lambda x,m:jnp.sum(m*relative_factor(x,m,.5,jnp.array(1.)))
    gx,gm=jax.grad(fn,argnums=(0,1))(x,m)
    assert float(jnp.max(jnp.abs(gx)))>0 and jnp.all(jnp.isfinite(gm))
    eps=.001
    finite=(fn(x.at[0,0,0].add(eps),m)-fn(x.at[0,0,0].add(-eps),m))/(2*eps)
    np.testing.assert_allclose(gx[0,0,0],finite,rtol=.01,atol=.001)
    zero=jnp.zeros((1,1,3))
    for a in jax.grad(fn,argnums=(0,1))(zero,zero):assert bool(jnp.all(jnp.isfinite(a)))


@pytest.mark.parametrize('experiment,worker',[(125,0),(125,1),(126,0),(126,1),(127,0),(127,1)])
def test_streamed_gradients_match_monolithic_full_autodiff(experiment,worker):
    spec=arm_for(experiment,worker)
    # Exercise absolute limiting at the small fixture's scale, same equations.
    if spec['absolute'] is not None:spec=dict(spec,absolute=.01)
    if spec['relative'] is not None:spec=dict(spec,relative=.001)
    p,q,t=fixture(spec);e=Engine(p,spec);e.set_step(RAMP_STEPS)
    def reference(q):
        h=p.embed(t,e.cfg.compute_dtype)
        for layer in range(p.cfg.num_layers):h,_=block_forward(e.cfg,layer,e.parameters(layer),q[str(layer)],h,p.columns,spec,e.phase)
        value,_=head_tile(e.cfg,h[:,:-1],e.embedding,e.final_scale,p.placed(t[:,1:]),h[:,:-1],0.,t.shape[1]-1)
        return value
    expected,gradient=jax.value_and_grad(reference)(q)
    health,actual=e.loss_and_grad(q,t)
    np.testing.assert_allclose(health['loss'],expected,rtol=2e-5,atol=2e-6)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(gradient)):
        np.testing.assert_allclose(a,b,rtol=.005,atol=3e-6)
    if spec['mlp']:assert any(float(jnp.max(jnp.abs(actual[k]['mlp_lora']['b'])))>1e-8 for k in q)


def test_zero_ramp_zero_effect_mlp_and_checkpoint_exact_next_step(tmp_path):
    spec=arm_for(127,1);p,q,t=fixture(spec);e=Engine(p,spec);e.set_step(0)
    h,_,_=e.forward(q,t);expected=p.forward_hidden(t,compute='float32')
    np.testing.assert_allclose(h,expected,rtol=1e-5,atol=2e-6)
    e.set_step(RAMP_STEPS);health,g=e.loss_and_grad(q,t);tx=optimizer(q,spec);state=tx.init(q)
    q,state=update(tx,q,state,g)
    payload=dict(parameters=q,optimizer=state,best=q,best_step=jnp.array(RAMP_STEPS))
    store=ChunkedCheckpointStore(tmp_path,'unused');store.save('MAIN',{},payload,RAMP_STEPS)
    restored,meta=store.restore('MAIN',{},payload)
    actual=update(tx,restored['parameters'],restored['optimizer'],g);expected=update(tx,q,state,g)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):np.testing.assert_array_equal(a,b)
    assert meta['step']==RAMP_STEPS and int(restored['best_step'])==RAMP_STEPS
    bad=jax.tree.map(lambda x:jnp.full_like(x,jnp.nan),g)
    with pytest.raises(FloatingPointError,match='rejected'):update(tx,q,state,bad)
    report=e.inspect(q,t)
    assert all(r['accepted'] for r in report['layers'])



def test_clean_deadline_and_resume_preserve_optimizer_cursor(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    from scripts.m3q_t4_dynamic_campaign import run_training
    from extent.recovery_durability import RecoveryCheckpointStore
    spec=arm_for(125,1);p,q,t=fixture(spec)
    monkeypatch.setattr(RecoveryCheckpointStore,'sync',lambda *a,**kw:True)
    monkeypatch.setattr('scripts.m3q_t4_dynamic_campaign.background',lambda *a:None)
    monkeypatch.setattr('scripts.m3q_t4_dynamic_campaign.context_report',lambda *a:{})
    class TinySession:
        def __init__(self):
            self.args=SimpleNamespace(cache_dir=str(tmp_path),experiment=125,worker=1,max_new_hf_gb=1,telegram=False)
            self.hard_deadline=time.time()+1000;self.hub=None;self.prefix='tiny';self.uploaded=0;self.checks=0
            self.result=dict(contract=dict(arm=spec),fits={},tasks={},cost=dict(attempted_updates=0,training_input_tokens=0,teacher_input_tokens=0,rejected_updates=0))
        def check(self):
            self.checks+=1
            if self.checks>2:raise TimeoutError('controlled training deadline')
        def persist(self,*a,**kw):pass
        def finalize_budget(self):pass
        def task(self,key,call):
            if key.startswith('endpoint-'):raise TimeoutError('controlled finalization deadline')
            if key not in self.result['tasks']:self.result['tasks'][key]=dict(value=call())
            return self.result['tasks'][key]['value']
    s=TinySession();tokens=dict(A=t,B=t,C=t,FIT=np.repeat(t,16384,axis=0))
    assert run_training(s,p,tokens)=='deadline_partial'
    row=s.result['fits']['MAIN'];assert row['step']==row['durable_step']==2 and row['best_step']==-1
    s.checks=0
    assert run_training(s,p,tokens)=='deadline_partial'
    assert row['step']==row['durable_step']==4 and s.result['cost']['accepted_updates']==4
    assert row['best_step']==-1 and row['best_nll'] is None



def test_bf16_identity_and_actual_token_bound_profile():
    spec=dict(arm_for(127,1),absolute=.01,relative=.001)
    p,_,base,t=tiny_patient(jax.devices('cpu')[0]);t=np.asarray(t[:,:16]);p.base=base
    p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(base,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=4,head_dim=p.cfg.mamba.head_dim)
    p.current=jax.device_get(apply_corrections(base,p.coordinates,protected_input_columns=p.columns))
    q=initialize(p,spec);e=Engine(p,spec)
    h,_,_=e.forward(q,t);source=p.forward_hidden(t)
    error=np.max(np.abs(p.token_losses(h,t)-p.token_losses(source,t)))
    assert error<=.01
    e.set_step(RAMP_STEPS);report=e.inspect(q,t)
    assert all(v['accepted'] and v['relative_bound_ok'] and v['absolute_bound_ok'] for v in report['layers'])
    assert max(v['max_output_token_rms'] for v in report['layers'])<=.0102
    health,g=e.loss_and_grad(q,t)
    assert np.isfinite(health['loss']) and all(np.all(np.isfinite(a)) for a in jax.tree.leaves(g))
