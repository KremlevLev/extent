"""EXP124 engineering controls; tiny model, no physical TPU claims."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM
from extent.context_recovery import (training_batch,initial_coordinates,objective,hidden,tiled_nll,
    optimizer,SPEC,make_step,reconcile)
from extent.t4_scale_recovery import corrected_layer
from extent.recovery_subspace import initialize_corrections
from extent.model import causal_lm_loss
from extent.sharding import create_v5e_mesh,batch_sharding,named_sharding_tree
from extent.initialization import initialize_sharded_optimizer_state
from extent.chunked_checkpoint import ChunkedCheckpointStore
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout
from scripts.m3q_t4_preflight import tiny_patient


def fixture():
    p,_,_,t=tiny_patient(jax.devices()[0])
    cfg=replace(p.cfg,param_dtype='float32',compute_dtype='float32')
    model=HybridForCausalLM(cfg);base=model.init(jax.random.key(124),t[:,:16])['params']
    old=initialize_corrections(base,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=8,head_dim=cfg.mamba.head_dim)
    old=jax.tree.map(lambda x:x+.0001,old)
    q=initial_coordinates(cfg,old)
    columns=2*int(cfg.hidden_size*cfg.mamba.expand)+2*cfg.mamba.mimo_rank*cfg.mamba.groups*cfg.mamba.d_state
    return cfg,base,q,t[:,:16],columns


def test_same_inputs_and_exact_same_target_ids_boundaries():
    pool=np.arange(4194304,dtype=np.int32).reshape(16384,256)
    seen=[]
    for step in range(4096):
        short,sm=training_batch(pool,step,'SHORT');long,lm=training_batch(pool,step,'LONG')
        np.testing.assert_array_equal(short.ravel(),long.ravel())
        np.testing.assert_array_equal(short[:,1:].ravel(),long[:,1:][lm.astype(bool)])
        assert sm.sum()==lm.sum()==1020
        seen.append(int(long[0,0])//1024)
    assert len(set(seen))==4096
    np.testing.assert_array_equal(training_batch(pool,0,'LONG')[0],training_batch(pool,4096,'LONG')[0])


def test_tiled_head_and_full_model_gradients_agree():
    cfg,base,q,t,columns=fixture();mask=np.ones((1,15),np.float32);mask[0,7]=0
    def reference(q):
        corrected=dict(base)
        for i in range(cfg.num_layers):corrected[f'layers_{i}']=corrected_layer(base[f'layers_{i}'],q.get(str(i),{}),columns)
        # gain1 source control, gradients wrt LoRA only compared with monolithic model
        return causal_lm_loss(HybridForCausalLM(cfg).apply({'params':corrected},t),t,jnp.pad(mask,((0,0),(1,0))))
    actual,gradient=jax.value_and_grad(lambda q:objective(cfg,base,q,t,jnp.asarray(mask),columns))(q)
    expected,refgrad=jax.value_and_grad(reference)(q)
    np.testing.assert_allclose(actual,expected,rtol=2e-5)
    for layer in q:
        for a,b in zip(jax.tree.leaves(gradient[layer]['lora']),jax.tree.leaves(refgrad[layer]['lora'])):
            np.testing.assert_allclose(a,b,rtol=.003,atol=2e-6)
    # A learned gain must receive a true end-to-end gradient too.
    assert any(abs(float(gradient[k]['gain']))>1e-8 for k in q)


def test_tiled_padding_and_zero_gain_instrumentation():
    cfg,base,q,t,columns=fixture()
    h=hidden(cfg,base,q,t,columns)
    a=tiled_nll(cfg,base,h,t,tile=4);b=tiled_nll(cfg,base,h,t,tile=32)
    np.testing.assert_allclose(a,b,atol=2e-6)
    for row in q.values():row['gain']=jnp.array(-2.)
    assert float(objective(cfg,base,q,t,jnp.ones((1,15)),columns))!=float(a.mean())


def test_sharded_update_and_exact_checkpoint_continuation(tmp_path):
    cfg,base,q,t,columns=fixture();mesh=create_v5e_mesh()
    layout=trainable_layout(q,mesh);q=jax.device_put(q,layout)
    base=jax.device_put(base,named_sharding_tree(base,mesh));tx=optimizer(q,SPEC)
    initial=initialize_sharded_optimizer_state(tx,q,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),q),layout,mesh)
    state=initial.opt_state;step=jax.jit(make_step(cfg,columns,tx))
    t=jax.device_put(t,batch_sharding(mesh));mask=jax.device_put(np.ones((1,15),np.float32),batch_sharding(mesh))
    q,state,health=jax.block_until_ready(step(q,state,base,t,mask));assert bool(health['finite'])
    q=jax.device_put(q,layout);state=jax.device_put(state,initial.layout)
    store=ChunkedCheckpointStore(tmp_path,'unused');payload=dict(parameters=q,optimizer=state)
    store.save('LONG',{},payload,1);saved,meta=store.restore('LONG',{},payload,dict(parameters=layout,optimizer=initial.layout))
    expected=step(q,state,base,t,mask);actual=step(saved['parameters'],saved['optimizer'],base,t,mask)
    assert meta['step']==1
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(a,b)
    bad=jax.tree.map(lambda x:jnp.full_like(x,jnp.nan),q)
    _,_,health=step(bad,state,base,t,mask);assert not bool(health['finite'])


def test_resume_invalidates_future_diagnostics():
    row=dict(step=512,history=[dict(step=128),dict(step=512)],evaluations=[dict(step=256),dict(step=512)],final=dict(step=512))
    reconcile(row,256)
    assert row['step']==256 and len(row['history'])==1 and len(row['evaluations'])==1 and 'final' not in row


@pytest.mark.parametrize('arm',['SHORT','LONG'])
def test_production_batch_shapes_compile_and_update_on_eight_shards(arm):
    cfg,base,q,_,columns=fixture();mesh=create_v5e_mesh()
    layout=trainable_layout(q,mesh);q=jax.device_put(q,layout)
    base=jax.device_put(base,named_sharding_tree(base,mesh));tx=optimizer(q,SPEC)
    init=initialize_sharded_optimizer_state(tx,q,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),q),layout,mesh)
    pool=(np.arange(4194304,dtype=np.int32)%cfg.vocab_size).reshape(-1,256)
    t,mask=training_batch(pool,0,arm)
    step=jax.jit(make_step(cfg,columns,tx))
    _,_,health=jax.block_until_ready(step(q,init.opt_state,base,
        jax.device_put(t,batch_sharding(mesh)),jax.device_put(mask,batch_sharding(mesh))))
    assert bool(health['finite']) and np.isfinite(float(health['loss']))


def test_finalization_evaluates_common_pair_and_invalidates_cached_other_state():
    from types import SimpleNamespace
    from scripts.m3q_context_recovery_124_campaign import finalize
    cfg,base,q,t,columns=fixture()
    class Diagnostic:
        def context(self,q,windows):return {'128':{'nll':3.}}
        def profile(self,q,window):return {'accepted':True}
    s=SimpleNamespace(initial_q=q,result={'branches':{
        'SHORT':{'step':512,'final':{'state_sha256':'stale','validation_context':{'stale':True}}},
        'LONG':{'step':256}}},persist=lambda:None)
    tokens={p:np.zeros((4,256),np.int32) for p in ('validation','locked_wiki','locked_pg19')}
    finalize(s,{a:{'parameters':q} for a in ('SHORT','LONG')},Diagnostic(),tokens,{'SHORT':256,'LONG':256})
    assert s.result['comparison']['matched_updates']
    assert s.result['comparison']['latest_steps']['SHORT']==512
    assert s.result['branches']['SHORT']['final']['step']==256
    assert 'stale' not in s.result['branches']['SHORT']['final']['validation_context']
