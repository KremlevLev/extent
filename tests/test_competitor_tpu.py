"""Real compiled updates on virtual CPU shards; not physical TPU validation."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from types import SimpleNamespace
from extent import HybridForCausalLM
from scripts.m3q_t4_preflight import tiny_patient
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.competitor_recovery import initialize
from extent.competitor_recovery import arm_for
from extent.competitor_tpu import make_step,teacher_hidden,reconcile
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding
from extent.chunked_checkpoint import ChunkedCheckpointStore
from scripts.m3q_peer_tpu_campaign import initial_payload,record_endpoint


def fixture(spec):
    p,_,_,t=tiny_patient(jax.devices('cpu')[0])
    p.cfg=replace(p.cfg,param_dtype='float32',compute_dtype='float32')
    t=np.asarray(t[:,:16]);base=HybridForCausalLM(p.cfg).init(jax.random.key(125),t)['params']
    p.base=base;p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(base,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=8,head_dim=p.cfg.mamba.head_dim)
    p.coordinates=jax.tree.map(lambda x:x+.0001,p.coordinates)
    p.current=jax.device_get(apply_corrections(base,p.coordinates,protected_input_columns=p.columns))
    q=initialize(p,spec)
    return p,q,t




@pytest.mark.parametrize('experiment,worker,length',[(133,0,16),(133,2,16),(134,0,16),(134,1,1024)])
def test_sharded_actual_step_and_exact_checkpoint_next_update(tmp_path,experiment,worker,length):
    spec=arm_for(experiment,worker);p,q,t=fixture(spec)
    t=np.tile(t,(1,(length+t.shape[1]-1)//t.shape[1]))[:,:length]
    tc=replace(p.tc,param_dtype='float32',compute_dtype='float32',remat_policy='none')
    teacher=Qwen3ForCausalLM(tc);tp=teacher.init(jax.random.key(134),t)['params']
    tp=dict(tp,embed_tokens=p.base['embed_tokens'],norm=p.base['norm'])
    _,states=teacher.apply({'params':tp},t,return_hidden_states=True)
    np.testing.assert_allclose(teacher_hidden(p.cfg,tc,tp,jnp.asarray(t)),states[-1],rtol=1e-5,atol=3e-6)
    mesh=create_v5e_mesh();payload,layouts,tx=initial_payload(p.cfg,p.base,p.current,p.coordinates,p.columns,spec,mesh)
    for path,leaf in jax.tree.leaves_with_path(payload['optimizer']):
        if leaf.ndim>=2 and any(getattr(k,'name',None) in ('mu','nu') for k in path):assert not leaf.sharding.is_fully_replicated
    base=jax.device_put(p.base,named_sharding_tree(p.base,mesh));tp=jax.device_put(tp,named_sharding_tree(tp,mesh))
    t=jax.device_put(t,batch_sharding(mesh));step=jax.jit(make_step(p.cfg,tc,p.columns,spec,tx))
    q,state,health=jax.block_until_ready(step(payload['parameters'],payload['optimizer'],base,tp,t,jnp.asarray(2048.,jnp.float32)))
    assert bool(health['finite']) and np.isfinite(float(health['loss']))
    payload.update(parameters=jax.device_put(q,layouts['parameters']),optimizer=jax.device_put(state,layouts['optimizer']))
    store=ChunkedCheckpointStore(tmp_path,'unused');store.save('MAIN',{},payload,1)
    saved,meta=store.restore('MAIN',{},payload,layouts)
    expected=step(payload['parameters'],payload['optimizer'],base,tp,t,jnp.asarray(2049.,jnp.float32))
    actual=step(saved['parameters'],saved['optimizer'],base,tp,t,jnp.asarray(2049.,jnp.float32))
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(a,b)
    assert meta['step']==1


def test_resume_invalidates_future_and_completes_interrupted_endpoint():
    row=dict(step=4096,history=[dict(step=2048),dict(step=4096)],evaluations=[],best_step=4096,
        best_nll=None,final=dict(current_step=4096))
    reconcile(row,2048,-1)
    assert row['step']==2048 and len(row['history'])==1 and 'final' not in row
    class D:
        def __init__(self):self.fail=True
        def evaluate(self,q,t,step):
            if self.fail:self.fail=False;raise TimeoutError()
            return dict(nll=6.5)
        def profile(self,*a):return dict(accepted=True)
    d=D();s=SimpleNamespace(persist=lambda:None);payload={};spec=arm_for(134,0);tokens=dict(A=np.zeros((64,16)),B=np.zeros((32,16)),C=np.zeros((64,16)))
    with pytest.raises(TimeoutError):record_endpoint(s,row,payload,d,{},tokens,2048,spec)
    assert not row['evaluations']
    record_endpoint(s,row,payload,d,{},tokens,2048,spec)
    assert row['evaluations'][0]['selection_eligible'] and int(payload['best_step'])==2048


def test_tpu_checkpoint_retries_failed_save_and_same_step_best_change():
    from scripts.m3q_peer_tpu_campaign import Session
    class Store:
        def __init__(self):self.fail=True;self.calls=0
        def directory(self,arm):
            from pathlib import Path
            return Path('unused')
        def save(self,*a):
            self.calls+=1
            if self.fail:self.fail=False;raise OSError('test disk full')
    s=Session.__new__(Session);s.checkpoint_signatures={};s.store=Store();s.contract={};s.result=dict(pending_slots=[])
    with pytest.raises(OSError):s.checkpoint('A',dict(best_step=-1),128)
    assert not s.checkpoint_signatures
    s.checkpoint('A',dict(best_step=-1),128);s.checkpoint('A',dict(best_step=-1),128)
    assert s.store.calls==2
    s.checkpoint('A',dict(best_step=128),128);assert s.store.calls==3


def test_long_training_pool_permutation_and_boundary_targets():
    from extent.competitor_tpu import training_batch
    from extent.t4_scale_data import training_index
    pool=np.arange(4194304,dtype=np.int32).reshape(16384,256)
    indices=[]
    for step in range(4096):
        t=training_batch(pool,step,134)
        assert t.shape==(1,1024) and t[:,1:].size==1023
        indices.append(int(t[0,0])//1024)
    assert len(set(indices))==4096
    np.testing.assert_array_equal(training_batch(pool,0,133,4096),pool[training_index(4096)][None])


@pytest.mark.parametrize('bad_stage',['objective','parameters_before','optimizer_before','gradients','updates','parameters_after','optimizer_after'])
def test_proposal_health_identifies_exact_nonfinite_stage(bad_stage):
    from extent.competitor_tpu import proposal_health
    names=['objective','parameters_before','optimizer_before','gradients','updates','parameters_after','optimizer_after']
    values=[{'loss':jnp.asarray(1.)}]+[{'layer':{'kernel':jnp.asarray([1.,2.])}} for _ in range(6)]
    index=names.index(bad_stage)
    if index==0:values[0]['loss']=jnp.asarray(jnp.nan)
    else:values[index]['layer']['kernel']=jnp.asarray([jnp.inf,jnp.nan])
    def run():return proposal_health(values[0]['loss'],{},*values[1:],detailed=True)
    h=jax.jit(run)()
    assert not bool(h['finite'])
    assert [name for name,v in h['stages'].items() if not bool(v['finite'])]==[bad_stage]
    leaves=h['stages'][bad_stage]['leaves']
    assert sum(int(v['nan_count']) for v in leaves.values())==1
    if index:assert sum(int(v['inf_count']) for v in leaves.values())==1


def test_first_update_diagnostic_never_accepts_proposal_and_persists_each_arm():
    from scripts.m3q_peer_tpu_campaign import diagnose_first_updates,Session
    from extent.competitor_tpu import training_batch
    from extent.sharding import create_v5e_mesh
    import hashlib
    row=dict(step=0);q={'weight':np.array([1.],np.float32)};state={'count':np.array(0)}
    payload=dict(parameters=q,optimizer=state)
    pool=np.arange(4194304,dtype=np.int32).reshape(16384,256)
    calls=[]
    s=SimpleNamespace(args=SimpleNamespace(experiment=133),check=lambda:None,
        result=dict(cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0)),
        persist=lambda **kw:calls.append(('persist',kw)),emit=lambda v:calls.append(('emit',v)))
    def kernel(*args):return {'weight':np.array([np.nan])},{'count':np.array(1)},dict(finite=False,loss=1.,stages={'gradients':{'finite':False}})
    diagnose_first_updates(s,{'A':row},{'A':payload},{'A':kernel},{},{},{'FIT':pool},create_v5e_mesh(),{'A':dict(train_offset=4096,alpha=0)})
    assert row['step']==0 and payload['parameters'] is q and payload['optimizer'] is state
    assert row['first_update_diagnostic']['failed_stages']==['gradients']
    assert row['first_update_diagnostic']['training_batch_sha256']==hashlib.sha256(training_batch(pool,0,133,4096).tobytes()).hexdigest()
    assert s.result['status']=='diagnostic_failed' and calls[0][0]=='persist'
    assert 'accepted_updates' not in s.result['cost']
    # A diagnostic session never writes full source/checkpoint copies.
    session=Session.__new__(Session);session.args=SimpleNamespace(diagnostic_only=True)
    session.checkpoint('A',payload,0)
