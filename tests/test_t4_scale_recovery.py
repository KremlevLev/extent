"""Meaningful gradient/continuation checks, not physical1.7B GPU results."""
from dataclasses import replace
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.t4_scale_recovery import (Engine,initialize,optimizer,update,block_forward,head_tile,arm_for)
from extent.t4_scale_data import training_index
from extent.chunked_checkpoint import ChunkedCheckpointStore
from scripts.m3q_t4_preflight import tiny_patient
from scripts.m3q_t4_scale_campaign import probe_jobs,reconcile,contract_for


def fixture(fp32=True):
    p,model,params,t=tiny_patient(jax.devices('cpu')[0])
    if fp32:
        p.cfg=replace(p.cfg,param_dtype='float32',compute_dtype='float32')
        model=HybridForCausalLM(p.cfg)
        params=model.init(jax.random.key(120),t)['params']
    p.base=params
    p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(params,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=4,head_dim=p.cfg.mamba.head_dim)
    p.coordinates=jax.tree.map(lambda x:x+.0001,p.coordinates)
    p.current=jax.device_get(apply_corrections(params,p.coordinates,protected_input_columns=p.columns))
    # Teacher final features are a fixed tensor for an exact KL gradient test.
    fixed=jax.random.normal(jax.random.key(2),(1,40,p.cfg.hidden_size))*.1
    p.forward_hidden=lambda tokens,**kw:fixed[:,:tokens.shape[1]].astype(getattr(jnp,p.cfg.compute_dtype))
    return p,np.asarray(t)


def full_reference(e,q,t):
    h=e.p.embed(t,e.cfg.compute_dtype);penalties=[]
    for layer in range(e.cfg.num_layers):
        h,penalty=block_forward(e.cfg,layer,e.p.placed(e.parameters(layer)),q[str(layer)],h,e.p.columns)
        if layer in e.cfg.mamba_layer_indices:penalties.append(penalty)
    teacher=e.p.forward_hidden(t,teacher=True) if e.spec['alpha'] else h
    value,_=head_tile(e.cfg,h[:,:-1],e.embedding,e.final_scale,e.p.placed(t[:,1:]),
        teacher[:,:-1],e.spec['alpha'],t.shape[0]*(t.shape[1]-1))
    return value+e.spec['penalty']*jnp.mean(jnp.stack(penalties))


@pytest.mark.parametrize('experiment,worker',[(121,0),(121,1),(122,0),(122,1),(123,0),(123,1)])
def test_streamed_backward_matches_full_autodiff(experiment,worker):
    p,t=fixture();spec=arm_for(experiment,worker);e=Engine(p,spec);q=initialize(p,spec)
    # Nonzero gains exercise penalty, chain rule and both gradient branches.
    for row in q.values():
        if 'gain' in row:row['gain']=jnp.array(.1,jnp.float32)
    expected,gradient=jax.value_and_grad(lambda q:full_reference(e,q,t))(q)
    health,actual=e.loss_and_grad(q,t)
    assert sum(float(jnp.sum(jnp.abs(a))) for a in jax.tree.leaves(actual))>1e-10
    np.testing.assert_allclose(health['loss'],float(expected),rtol=2e-5,atol=2e-6)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(gradient)):
        np.testing.assert_allclose(a,b,rtol=.003,atol=2e-6)


def test_gain1_hook_matches_streamed_source_bf16():
    p,t=fixture(False);spec=arm_for(121,0);e=Engine(p,spec);q=initialize(p,spec)
    h,_,_=e.forward(q,t)
    source=p.embed(t)
    for layer in range(p.cfg.num_layers):source=p.layer(layer,source)
    got=p.token_losses(h,t);expected=p.token_losses(source,t)
    assert np.max(np.abs(got-expected))<=.01


def test_probe_zero_gain_and_inspection_use_actual_block_outputs():
    p,t=fixture(False);spec=arm_for(121,0);e=Engine(p,spec,direct=True)
    q=initialize(p,spec)
    for row in q.values():
        if 'gain' in row:row['gain']=jnp.array(0.,jnp.float32)
    report=e.inspect(q,t)
    assert all(r['accepted'] for r in report['layers'])
    for r in report['layers']:
        if r['layer'] in p.cfg.mamba_layer_indices:assert r['scaled_mixer_rms']==0
    shorter=e.inspect(q,t[:,:16])
    assert all(r['accepted'] for r in shorter['layers'])


def test_nonfinite_optimizer_proposal_is_rejected():
    p,t=fixture();spec=arm_for(122,1);q=initialize(p,spec);tx=optimizer(q,spec);state=tx.init(q)
    bad=jax.tree.map(lambda x:jnp.full_like(x,jnp.nan),q)
    with pytest.raises(FloatingPointError,match='rejected'):update(tx,q,state,bad)


def test_optimizer_checkpoint_next_step_exact_and_best_preserved(tmp_path):
    p,t=fixture();spec=arm_for(122,1);q=initialize(p,spec);tx=optimizer(q,spec);state=tx.init(q)
    gradient=jax.tree.map(lambda x:jnp.ones_like(x)*.01,q)
    q,state=update(tx,q,state,gradient)
    payload=dict(parameters=q,optimizer=state,best=initialize(p,spec),best_step=jnp.array(0))
    store=ChunkedCheckpointStore(tmp_path,'unused');store.save('MAIN',{},payload,1)
    restored,meta=store.restore('MAIN',{},payload)
    expected=update(tx,q,state,gradient);actual=update(tx,restored['parameters'],restored['optimizer'],gradient)
    assert meta['step']==1 and int(restored['best_step'])==0
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(a,b)


def test_permutation_coverage_probes_and_future_cursor_rollback():
    assert len({training_index(i) for i in range(16384)})==16384
    assert training_index(0)==training_index(16384)
    jobs=probe_jobs([j for j in range(28) if j not in (6,13,20,27)])
    assert len(jobs)>400 and len({r['key'] for r in jobs})==len(jobs)
    assert all(0<=r['gain']<=1 and not set(r['layers'])&{6,13,20,27} for r in jobs)
    row=dict(step=600,history=[dict(step=128),dict(step=600)],evaluations=[dict(step=0),dict(step=512)],final={})
    reconcile(row,128,0)
    assert row['step']==128 and len(row['history'])==1 and len(row['evaluations'])==1 and 'final' not in row


def test_training_contracts_are_distinct_and_positive_horizon():
    for experiment in (120,121,122,123):
        for worker in (0,1):
            c=contract_for(experiment,worker,2)
            assert c['steps']==65536
            if experiment!=120:assert c['arm']['name']==arm_for(experiment,worker)['name']


def test_clean_deadline_saves_training_cursor_and_resumes_without_repeating(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from extent.recovery_durability import RecoveryCheckpointStore
    from scripts.m3q_t4_scale_campaign import run_training
    p,t=fixture();spec=arm_for(122,1)
    monkeypatch.setattr(RecoveryCheckpointStore,'sync',lambda *a,**kw:True)
    class TinySession:
        def __init__(self):
            self.args=SimpleNamespace(cache_dir=str(tmp_path),experiment=122,worker=1,max_new_hf_gb=1)
            self.hard_deadline=time.time()+1000;self.hub=None;self.prefix='tiny';self.uploaded=0;self.checks=0
            self.result=dict(contract=dict(arm=spec),fits={},tasks={},cost=dict(attempted_updates=0,training_input_tokens=0,
                teacher_input_tokens=0,rejected_updates=0))
        def check(self):
            self.checks+=1
            if self.checks>2:raise TimeoutError('tiny controlled deadline')
        def persist(self,*a,**kw):pass
        def finalize_budget(self):pass
        def task(self,*a,**kw):raise TimeoutError('reserve exhausted in tiny test')
    # Source/hook equivalence is covered above; here exercise real optimization,
    # snapshots and cursor restore with a controlled two-step deadline.
    monkeypatch.setattr('scripts.m3q_t4_scale_campaign.background',lambda *a:None)
    s=TinySession();tokens=dict(A=t,B=t,C=t,FIT=np.repeat(t,16384,axis=0))
    assert run_training(s,p,tokens)=='deadline_partial'
    assert s.result['fits']['MAIN']['step']==s.result['fits']['MAIN']['durable_step']==2
    assert s.result['cost']['attempted_updates']==2
    s.checks=0
    assert run_training(s,p,tokens)=='deadline_partial'
    assert s.result['fits']['MAIN']['step']==s.result['fits']['MAIN']['durable_step']==4
    assert s.result['cost']['attempted_updates']==4
