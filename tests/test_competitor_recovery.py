"""Peer recipe adaptations: real full-chain gradients and persistence."""
import time
from dataclasses import replace
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from extent import HybridForCausalLM
from scripts.m3q_t4_preflight import tiny_patient
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.competitor_recovery import (Engine,arm_for,initialize,optimizer,block_forward,head_tile,with_reset_targets,effective_layer)
from extent.competitor_tpu import objective,hidden,make_step
from extent.t4_scale_recovery import make_update


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




CASES=[(e,w) for e in (130,131,132,134) for w in range(2)]+[(133,w) for w in range(3)]

@pytest.mark.parametrize('experiment,worker',CASES)
def test_streamed_full_chain_matches_tiled_tpu_math(experiment,worker):
    spec=arm_for(experiment,worker);p,q,t=fixture(spec);e=Engine(p,spec);e.set_step(2048)
    fixed=jax.random.normal(jax.random.key(3),(1,t.shape[1],p.cfg.hidden_size))*.1
    p.forward_hidden=lambda tokens,**kw:fixed[:,:tokens.shape[1]]
    base={f'layers_{i}':e.parameters(i) for i in range(p.cfg.num_layers)}
    base.update(embed_tokens=p.current['embed_tokens'],norm=p.current['norm'])
    def fn(q):return objective(p.cfg,base,q,t,p.columns,spec,jnp.array(2048.),fixed if spec['alpha'] else None,tile=4)[0]
    expected,g=jax.value_and_grad(fn)(q);health,actual=e.loss_and_grad(q,t)
    np.testing.assert_allclose(health['loss'],expected,rtol=3e-5,atol=3e-6)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(g)):np.testing.assert_allclose(a,b,rtol=.01,atol=5e-6)
    assert all(r['accepted'] for r in e.inspect(q,t)['layers'])
    if spec['unprotect']:
        assert any(float(jnp.max(jnp.abs(row['lora']['in_proj']['b'][:,p.columns:])))>0 for row in actual.values() if 'lora' in row)
    if spec['dense']:
        assert all('dense_mamba' in q[str(i)] for i in p.cfg.mamba_layer_indices)
    update=make_update(optimizer(q,spec));state=optimizer(q,spec).init(q)
    q,state,_=update(q,state,actual)
    saved=serialization.from_bytes(dict(q=q,state=state),serialization.to_bytes(dict(q=q,state=state)))
    for a,b in zip(jax.tree.leaves(update(q,state,actual)),jax.tree.leaves(update(saved['q'],saved['state'],actual))):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('experiment,worker',[(130,1),(131,1),(132,1),(134,1)])
def test_bf16_source_identity_and_full_strength_health(experiment,worker):
    spec=arm_for(experiment,worker);p,_,base,t=tiny_patient(jax.devices('cpu')[0]);t=t[:,:12];p.base=base
    p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(base,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=4,head_dim=p.cfg.mamba.head_dim)
    p.current=jax.device_get(apply_corrections(base,p.coordinates,protected_input_columns=p.columns))
    q=initialize(p,spec);e=Engine(p,spec)
    h,_,_=e.forward(q,t);source=p.forward_hidden(t)
    assert np.max(np.abs(p.token_losses(h,t)-p.token_losses(source,t)))<=.01
    p.forward_hidden=lambda tokens,**kw:jnp.zeros((1,tokens.shape[1],p.cfg.hidden_size),jnp.bfloat16)
    e.set_step(2048);health,g=e.loss_and_grad(q,t)
    assert np.isfinite(health['loss']) and all(np.all(np.isfinite(v)) for v in jax.tree.leaves(g))


def test_reset_exact_targets_and_identity():
    spec=arm_for(130,1);p,q,t=fixture(spec);i=p.cfg.mamba_layer_indices[0]
    params=with_reset_targets(p.base[f'layers_{i}'],p.cfg,i)
    before=effective_layer(params,q[str(i)],p.columns,spec,0.)
    full=effective_layer(params,q[str(i)],p.columns,spec,1.)
    np.testing.assert_array_equal(full['mamba']['in_proj']['kernel'][:,p.columns:],0.)
    np.testing.assert_allclose(jnp.sum(full['mamba']['mimo_x'],axis=1),1.,atol=2e-7)
    np.testing.assert_allclose(jax.nn.softplus(full['mamba']['dt_bias']),jax.nn.softplus(params['_reset']['dt_bias']),rtol=1e-6)
    assert jax.tree.structure(before)==jax.tree.structure(full)


def test_temperature1_kl_formula():
    from extent.plateau_forensics import norm
    spec=arm_for(132,1);p,q,t=fixture(spec);rng=np.random.default_rng(5)
    h=jnp.asarray(rng.normal(size=(1,4,p.cfg.hidden_size)),jnp.float32)
    teacher=h*.7+1;emb=p.current['embed_tokens']['embedding'];scale=p.current['norm']['scale'];labels=t[:,1:5]
    val,(ce,kl)=head_tile(p.cfg,h,emb,scale,labels,teacher,1.,4)
    sl=jax.nn.log_softmax(norm(h,{'scale':scale},p.cfg)@emb.T,-1)
    tl=jax.nn.log_softmax(norm(teacher,{'scale':scale},p.cfg)@emb.T,-1)
    np.testing.assert_allclose(val,jnp.mean(jnp.sum(jnp.exp(tl)*(tl-sl),-1)),atol=1e-6)
    np.testing.assert_array_equal(val,kl)


def test_competitor_runner_clean_deadline_restore(tmp_path,monkeypatch):
    from scripts.m3q_t4_competitor_campaign import run_training
    from extent.recovery_durability import RecoveryCheckpointStore
    spec=arm_for(132,1);p,q,t=fixture(spec)
    p.forward_hidden=lambda tokens,**kw:jnp.zeros((1,tokens.shape[1],p.cfg.hidden_size),jnp.float32)
    monkeypatch.setattr(RecoveryCheckpointStore,'sync',lambda *a,**kw:True)
    monkeypatch.setattr('scripts.m3q_t4_competitor_campaign.background',lambda *a:None)
    monkeypatch.setattr('scripts.m3q_t4_competitor_campaign.context_report',lambda *a:{})
    class Session:
        def __init__(self):
            self.args=SimpleNamespace(cache_dir=str(tmp_path),experiment=132,worker=1,max_new_hf_gb=1,telegram=False)
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
    assert s.result['cost']['teacher_input_tokens']==4*t.size


def test_tanh_angle_is_float32_before_recurrence(monkeypatch):
    import extent.competitor_recovery as module
    from extent.plateau_forensics import project_mamba
    spec=arm_for(130,0);p,q,t=fixture(spec)
    cfg=replace(p.cfg,compute_dtype='bfloat16')
    i=cfg.mamba_layer_indices[0];params=p.base[f'layers_{i}']['mamba']
    inputs=jnp.ones((1,4,cfg.hidden_size),jnp.bfloat16)
    raw=project_mamba(inputs,params,cfg)[0][-1].astype(jnp.float32)
    actual=module.pregate_scan;seen=[]
    def spy(*args,**kw):
        seen.append(args[7]);return actual(*args,**kw)
    monkeypatch.setattr(module,'pregate_scan',spy)
    H=int(cfg.hidden_size*cfg.mamba.expand)//cfg.mamba.head_dim
    module.normalized_mixer(inputs,params,cfg,jnp.ones((H,cfg.mamba.head_dim)),1.,spec)
    assert seen[0].dtype==jnp.float32
    np.testing.assert_array_equal(seen[0],jnp.tanh(raw))
