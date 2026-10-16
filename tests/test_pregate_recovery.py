"""EXP129: normalization placement, independent recurrence views, full gradients and resume."""
import time
from types import SimpleNamespace
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM
from scripts.m3q_t4_preflight import tiny_patient
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.pregate_recovery import (Engine,arm_for,initialize,optimizer,block_forward,
    normalize_readout,pregate_scan,pregate_dual,normalized_mixer,RAMP_STEPS)
from extent.t4_scale_recovery import head_tile,make_update
from extent.layers.mamba3 import mamba3_reference_scan


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



def test_norm_axes_eps_weight_and_extremes():
    x=jnp.array([[[[3.,4.],[30.,40.]],[[1.,2.],[.001,.002]]]])
    w=jnp.array([[1.,2.],[3.,4.]])
    expected=x/jnp.sqrt(jnp.mean(x*x,axis=-1,keepdims=True)+1e-5)*w[None,None]
    np.testing.assert_allclose(normalize_readout(x,w[None,None],1.),expected,rtol=2e-6)
    np.testing.assert_array_equal(normalize_readout(x,w[None,None],0.),x)
    extreme=jnp.array([[[1e30,-1e30,0.,1e-20]]])
    for phase in (0.,.5,1.):
        value,grad=jax.value_and_grad(lambda v:jnp.sum(normalize_readout(v,jnp.ones((1,1,4)),phase)))(extreme)
        assert np.isfinite(value) and np.all(np.isfinite(grad))
    zero=jnp.zeros((1,1,4))
    expected_gradient=np.ones((1,1,4))/np.sqrt(1e-5)
    np.testing.assert_allclose(jax.grad(lambda v:jnp.sum(normalize_readout(v,jnp.ones_like(v),1.)))(zero),expected_gradient,rtol=2e-6)


def scan_inputs():
    rng=np.random.default_rng(129);B,T,H,P,R,N=2,5,2,3,2,4
    def rand(shape):return jnp.asarray(rng.normal(size=shape)*.2,jnp.float32)
    return (rand((B,T,H,P)),rand((B,T,H,P)),rand((B,T,R,H,N)),rand((B,T,R,H,N)),
        jnp.full((B,T,H),.1),-jnp.ones((B,T,H)),rand((B,T,H)),rand((B,T,N//2)),
        rand((H,R,P)),rand((H,R,P)),rand((H,R,P)),jnp.ones(H),N//2)


def test_scan_dual_and_gradients_and_legacy_identity():
    values=scan_inputs();w=jnp.array([[1.,2.,3.],[4.,5.,6.]])
    np.testing.assert_array_equal(pregate_scan(*values,w,0.),mamba3_reference_scan(*values))
    for phase in (.3,1.):
        a=pregate_scan(*values,w,phase);b=pregate_dual(*values,w,phase)
        np.testing.assert_allclose(a,b,rtol=3e-5,atol=2e-7)
        ga=jax.grad(lambda x,weight:jnp.sum(pregate_scan(x,*values[1:],weight,phase)**2),argnums=(0,1))(values[0],w)
        gb=jax.grad(lambda x,weight:jnp.sum(pregate_dual(x,*values[1:],weight,phase)**2),argnums=(0,1))(values[0],w)
        for x,y in zip(ga,gb):np.testing.assert_allclose(x,y,rtol=3e-4,atol=2e-7)
    ordinary=pregate_scan(*values,w,1.)
    diagnostic,metrics=pregate_scan(*values,w,1.,details=True)
    np.testing.assert_array_equal(ordinary,diagnostic)
    assert float(metrics['readout_post_rms_max'])<=6.001


@pytest.mark.parametrize('worker',[0,1])
def test_full_streamed_gradient_and_exact_optimizer_restore(worker):
    from flax import serialization
    spec=arm_for(129,worker);p,q,t=fixture(spec);e=Engine(p,spec);e.set_step(RAMP_STEPS)
    def reference(q):
        h=p.embed(t,e.cfg.compute_dtype)
        for layer in range(p.cfg.num_layers):h,_=block_forward(e.cfg,layer,e.parameters(layer),q[str(layer)],h,p.columns,spec,e.phase)
        value,_=head_tile(e.cfg,h[:,:-1],e.embedding,e.final_scale,p.placed(t[:,1:]),h[:,:-1],0.,t.shape[1]-1)
        return value
    expected,gradient=jax.value_and_grad(reference)(q);health,actual=e.loss_and_grad(q,t)
    np.testing.assert_allclose(health['loss'],expected,rtol=2e-5,atol=2e-6)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(gradient)):np.testing.assert_allclose(a,b,rtol=.006,atol=4e-6)
    if worker:
        assert any(float(jnp.max(jnp.abs(v['pregate_weight'])))>0 for v in actual.values() if 'pregate_weight' in v)
    report=e.inspect(q,t);assert all(r['accepted'] for r in report['layers'])
    update=make_update(optimizer(q,spec));state=optimizer(q,spec).init(q)
    q,state,_=update(q,state,actual)
    restored=serialization.from_bytes(dict(q=q,state=state),serialization.to_bytes(dict(q=q,state=state)))
    a=update(q,state,actual);b=update(restored['q'],restored['state'],actual)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('worker',[0,1])
def test_bf16_source_parity_profiles_and_finite_gradient(worker):
    spec=arm_for(129,worker)
    p,_,base,t=tiny_patient(jax.devices('cpu')[0]);t=np.asarray(t[:,:16]);p.base=base
    p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(base,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=4,head_dim=p.cfg.mamba.head_dim)
    p.current=jax.device_get(apply_corrections(base,p.coordinates,protected_input_columns=p.columns))
    q=initialize(p,spec);e=Engine(p,spec)
    h,_,_=e.forward(q,t);source=p.forward_hidden(t)
    assert np.max(np.abs(p.token_losses(h,t)-p.token_losses(source,t)))<=.01
    e.set_step(RAMP_STEPS);report=e.inspect(q,t)
    assert all(v['accepted'] for v in report['layers'])
    health,g=e.loss_and_grad(q,t)
    assert np.isfinite(health['loss']) and all(np.all(np.isfinite(a)) for a in jax.tree.leaves(g))


def test_pregate_runner_clean_deadline_restore(tmp_path,monkeypatch):
    from scripts.m3q_t4_pregate_129_campaign import run_training
    from extent.recovery_durability import RecoveryCheckpointStore
    spec=arm_for(129,0);p,q,t=fixture(spec)
    monkeypatch.setattr(RecoveryCheckpointStore,'sync',lambda *a,**kw:True)
    monkeypatch.setattr('scripts.m3q_t4_pregate_129_campaign.background',lambda *a:None)
    monkeypatch.setattr('scripts.m3q_t4_pregate_129_campaign.context_report',lambda *a:{})
    class Session:
        def __init__(self):
            self.args=SimpleNamespace(cache_dir=str(tmp_path),experiment=129,worker=0,max_new_hf_gb=1,telegram=False)
            self.hard_deadline=time.time()+1000;self.hub=None;self.prefix='tiny';self.uploaded=0;self.checks=0
            self.result=dict(contract=dict(arm=spec),fits={},tasks={},cost=dict(attempted_updates=0,training_input_tokens=0,teacher_input_tokens=0,rejected_updates=0))
        def check(self):
            self.checks+=1
            if self.checks>2:raise TimeoutError('test training deadline')
        def persist(self,*a,**kw):pass
        def finalize_budget(self):pass
        def task(self,key,call):
            if key.startswith('endpoint-'):raise TimeoutError('test reserve deadline')
            if key not in self.result['tasks']:self.result['tasks'][key]=dict(value=call())
            return self.result['tasks'][key]['value']
    s=Session();tokens=dict(A=t,B=t,C=t,FIT=np.repeat(t,16384,axis=0))
    assert run_training(s,p,tokens)=='deadline_partial'
    assert s.result['fits']['MAIN']['step']==s.result['fits']['MAIN']['durable_step']==2
    s.checks=0;assert run_training(s,p,tokens)=='deadline_partial'
    assert s.result['fits']['MAIN']['step']==s.result['fits']['MAIN']['durable_step']==4
    assert s.result['cost']['accepted_updates']==4
