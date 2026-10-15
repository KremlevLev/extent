"""EXP128 engineering: per-component preservation, exact chain rule and resume."""
import math
import time
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace
from extent import HybridForCausalLM
from scripts.m3q_t4_preflight import tiny_patient
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.channel_bound_recovery import initialize
from extent.channel_bound_recovery import Engine,arm_for,channel_factor,block_forward,RAMP_STEPS
from extent.t4_scale_recovery import head_tile


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


def test_hard_preserves_small_components_in_large_vector():
    x=jnp.array([[[1e12,.1,-.2,0.]]]);bound=math.log(1.)
    result=x*channel_factor(x,bound,1.,'hard')
    np.testing.assert_allclose(result,[[[1.,.1,-.2,0.]]],rtol=3e-6,atol=1e-7)
    for mode in ('hard','tanh'):
        np.testing.assert_array_equal(x*channel_factor(x,bound,0.,mode),x)
        out=x*channel_factor(x,bound,1.,mode)
        assert float(jnp.max(jnp.abs(out)))<=1.00001
        assert bool(jnp.all(jnp.isfinite(jax.grad(lambda v:jnp.sum(v*channel_factor(v,bound,1.,mode)))(jnp.zeros((1,1,4))))))
    soft=x*channel_factor(x,bound,1.,'tanh')
    np.testing.assert_allclose(soft[0,0,1:3],jnp.tanh(x[0,0,1:3]),rtol=2e-6)


@pytest.mark.parametrize('worker',[0,1])
def test_streamed_component_bounds_gradients_match_full_chain(worker):
    spec=dict(arm_for(128,worker),absolute=.01,relative=.001)
    p,q,t=fixture(spec);e=Engine(p,spec);e.set_step(RAMP_STEPS)
    def reference(q):
        h=p.embed(t,e.cfg.compute_dtype)
        for layer in range(p.cfg.num_layers):h,_=block_forward(e.cfg,layer,e.parameters(layer),q[str(layer)],h,p.columns,spec,e.phase)
        value,_=head_tile(e.cfg,h[:,:-1],e.embedding,e.final_scale,p.placed(t[:,1:]),h[:,:-1],0.,t.shape[1]-1)
        return value
    expected,gradient=jax.value_and_grad(reference)(q);health,actual=e.loss_and_grad(q,t)
    np.testing.assert_allclose(health['loss'],expected,rtol=2e-5,atol=2e-6)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(gradient)):np.testing.assert_allclose(a,b,rtol=.005,atol=3e-6)
    report=e.inspect(q,t)
    assert report['active_fraction_scope']=='components'
    assert all(r['accepted'] and r['relative_bound_ok'] and r['absolute_bound_ok'] and r['component_peak_bound_ok'] for r in report['layers'])
    assert max(r['output_max'] for r in report['layers'])<=.0102


def test_channel_runner_clean_deadline_restore(tmp_path,monkeypatch):
    from scripts.m3q_t4_channel_bound_128_campaign import run_training
    from extent.recovery_durability import RecoveryCheckpointStore
    spec=arm_for(128,0);p,q,t=fixture(spec)
    monkeypatch.setattr(RecoveryCheckpointStore,'sync',lambda *a,**kw:True)
    monkeypatch.setattr('scripts.m3q_t4_channel_bound_128_campaign.background',lambda *a:None)
    monkeypatch.setattr('scripts.m3q_t4_channel_bound_128_campaign.context_report',lambda *a:{})
    class Session:
        def __init__(self):
            self.args=SimpleNamespace(cache_dir=str(tmp_path),experiment=128,worker=0,max_new_hf_gb=1,telegram=False)
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



@pytest.mark.parametrize('worker',[0,1])
def test_bf16_component_bounds_and_identity(worker):
    spec=dict(arm_for(128,worker),absolute=.01,relative=.001)
    p,_,base,t=tiny_patient(jax.devices('cpu')[0]);t=np.asarray(t[:,:16]);p.base=base
    p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(base,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=4,head_dim=p.cfg.mamba.head_dim)
    p.current=jax.device_get(apply_corrections(base,p.coordinates,protected_input_columns=p.columns))
    q=initialize(p,spec);e=Engine(p,spec)
    h,_,_=e.forward(q,t);source=p.forward_hidden(t)
    assert np.max(np.abs(p.token_losses(h,t)-p.token_losses(source,t)))<=.01
    e.set_step(RAMP_STEPS);report=e.inspect(q,t)
    assert all(v['accepted'] and v['relative_bound_ok'] and v['absolute_bound_ok'] and v['component_peak_bound_ok'] for v in report['layers'])
    health,g=e.loss_and_grad(q,t)
    assert np.isfinite(health['loss']) and all(np.all(np.isfinite(a)) for a in jax.tree.leaves(g))
