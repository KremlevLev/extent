"""CPU/virtual-device evidence only; no TPU or live remote notifications."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM, tiny_config
from extent.batch_recovery import averaged_gradient, optimizer, make_step, schedule, TOTAL_STEPS
from extent.decoder_recovery import split_decoder
from extent.stable_gradient_clip import stable_clip_by_global_norm
from extent.chunked_checkpoint import ChunkedCheckpointStore
from extent.recovery_durability import RecoveryCheckpointStore, cloud_roundtrip
from extent.initialization import initialize_sharded_optimizer_state
from extent.sharding import create_v5e_mesh, named_sharding_tree, batch_sharding, replicated_sharding
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout


from jax.sharding import NamedSharding, PartitionSpec as P
@pytest.mark.parametrize("accumulation,adaptive",[(1,False),(4,False),(4,True)])
def test_compiled_sharded_step_and_binary_next_step_resume(tmp_path,accumulation,adaptive):
    config=replace(tiny_config(),param_dtype="bfloat16",compute_dtype="bfloat16")
    model=HybridForCausalLM(config)
    mesh=create_v5e_mesh()
    tokens=jax.device_put(jnp.arange(4,dtype=jnp.int32)[None],batch_sharding(mesh))
    initial=model.init(jax.random.key(106),tokens)["params"]
    p,f=split_decoder(initial)
    pl,fl=trainable_layout(p,mesh),named_sharding_tree(f,mesh)
    p,f=jax.device_put(p,pl),jax.device_put(f,fl)
    tx=optimizer(p,adaptive,accumulation)
    abstract=jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),p)
    sharded=initialize_sharded_optimizer_state(tx,p,abstract,pl,mesh)
    for x,layout in zip(jax.tree.leaves(sharded.opt_state[1].mu),jax.tree.leaves(pl)):
        assert x.sharding==layout
    micro_layout=NamedSharding(mesh,P("data",None,None))
    tokens=jax.device_put(jnp.stack([tokens]*accumulation),micro_layout)
    step=make_step(model,tx,"DECODER-CE",0,config.mamba.head_dim,
        (pl,sharded.layout,fl,micro_layout,replicated_sharding(mesh)),adaptive=adaptive,accumulation=accumulation)
    state=sharded.opt_state
    for _ in range(3):
        p,state,h=step(p,state,f,tokens)
        assert bool(h["finite"])
    assert int(state[1].count)==3
    assert h["layers"] and all("log10_relative_update" in row for row in h["layers"].values())
    for key in ("embed_tokens","lm_head"):
        if key in initial:
            for a,b in zip(jax.tree.leaves(initial[key]),jax.tree.leaves(f[key])):
                np.testing.assert_array_equal(a,b)
    store=ChunkedCheckpointStore(tmp_path,"test",chunk_bytes=4096)
    template=dict(parameters=p,optimizer=state)
    contract=dict(adaptive=adaptive,accumulation=accumulation)
    store.save("arm",contract,template,3*accumulation)
    restored,meta=store.restore("arm",contract,template,dict(parameters=pl,optimizer=sharded.layout))
    assert meta["step"]==3*accumulation
    for a,b in zip(jax.tree.leaves(step(p,state,f,tokens)),
                   jax.tree.leaves(step(restored["parameters"],restored["optimizer"],f,tokens))):
        np.testing.assert_array_equal(a,b)



def test_unclipped_mean_and_micro_finite_guard():
    p={"layers_0":{"w":jnp.array([2.,-1.])}}
    tokens=jnp.array([[1.,4.],[-3.,2.],[5.,-6.],[-2.,3.]])
    def loss(p,t):return jnp.sum((p["layers_0"]["w"]-t)**2),jnp.array(True)
    value,g,finite,noise=averaged_gradient(loss,p,tokens)
    expected=jax.grad(lambda p:jnp.mean(jax.vmap(lambda t:loss(p,t)[0])(tokens)))(p)
    np.testing.assert_allclose(g["layers_0"]["w"],expected["layers_0"]["w"],rtol=1e-6)
    assert bool(finite) and float(noise["micro_loss_std"])>0
    def bad(p,t):return loss(p,t)[0],jnp.all(t>=0)
    assert not bool(averaged_gradient(bad,p,tokens)[2])


def test_token_clock_budget_and_contracts():
    from scripts.m3q_batch_recovery_campaign import session_budget,contract_for,compare_campaigns,ROOT
    from extent.config import load_config
    assert session_budget(9)==(27000.,90.)
    assert session_budget(8.8)[0]==pytest.approx(26280.)
    with pytest.raises(ValueError):session_budget(9.1)
    assert float(schedule(1024))==pytest.approx(1e-5)
    assert float(schedule(65536))==pytest.approx(1e-6)
    p={"layers_0":{"w":jnp.ones((2,2))}}
    for accumulation in (1,4):
        tx=optimizer(p,accumulation=accumulation); state=tx.init(p)
        for n in range(3):
            updates,state=tx.update(jax.tree.map(jnp.zeros_like,p),state,p)
            np.testing.assert_allclose(updates["layers_0"]["w"],-.1*float(schedule(n*accumulation)),atol=1e-12)
    config,_=load_config(ROOT/"config/hybrid_1_7b_gqa_v5e8.yaml")
    contracts=[contract_for(config,n) for n in (108,109,110)]
    assert [c["optimizer"]["accumulation"] for c in contracts]==[1,4,4]
    assert compare_campaigns({"contract":contracts[0]},{"contract":contracts[1]})["primary_gate"] is None
    assert compare_campaigns({"contract":contracts[1]},{"contract":contracts[2]})["primary_gate"] is None
    with pytest.raises(ValueError):compare_campaigns({"contract":contracts[2]},{"contract":contracts[0]})


def test_paired_gate_and_mismatch_rejection():
    from scripts.m3q_batch_recovery_campaign import contract_for,compare_campaigns,ROOT
    from extent.config import load_config
    config,_=load_config(ROOT/"config/hybrid_1_7b_gqa_v5e8.yaml")
    def result(exp,end):
        metric=lambda n:{name:{"nll":n,"windows":[n,n]} for name in ("locked_test","pg19_test")}
        return {"contract":contract_for(config,exp),"branches":{str(seed):{"DECODER-CE":
            {"step":65536,"start":metric(8.),"final":metric(end)}} for seed in (123,456)}}
    control,candidate=result(108,7.9),result(109,7.7)
    assert compare_campaigns(control,candidate)["primary_gate"] is True
    candidate["branches"]["456"]["DECODER-CE"]["final"]["pg19_test"]={"nll":7.85,"windows":[7.85,7.85]}
    assert compare_campaigns(control,candidate)["primary_gate"] is False
    candidate["contract"]["data"][0][2]+=256
    with pytest.raises(ValueError,match="unmatched"):compare_campaigns(control,candidate)
