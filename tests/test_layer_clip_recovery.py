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
from extent.layer_clip_recovery import clip_gradients, optimizer, make_step, schedule, TOTAL_STEPS
from extent.decoder_recovery import split_decoder
from extent.stable_gradient_clip import stable_clip_by_global_norm
from extent.chunked_checkpoint import ChunkedCheckpointStore
from extent.recovery_durability import RecoveryCheckpointStore, cloud_roundtrip
from extent.initialization import initialize_sharded_optimizer_state
from extent.sharding import create_v5e_mesh, named_sharding_tree, batch_sharding, replicated_sharding
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout


def test_layer_isolation_extreme_norms_and_global_identity():
    p={"layers_0":{"w":jnp.ones((8,))},"layers_1":{"w":jnp.ones((8,))}}
    g={"layers_0":{"w":jnp.full((8,),1e30)},"layers_1":{"w":jnp.full((8,),.001)}}
    global_grad,global_stats=clip_gradients(g,p,False)
    expected=stable_clip_by_global_norm(1).update(g,())[0]
    for a,b in zip(jax.tree.leaves(global_grad),jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a,b)
    adaptive,stats=clip_gradients(g,p,True)
    assert all(np.all(np.isfinite(x)) for x in jax.tree.leaves(adaptive))
    np.testing.assert_allclose(adaptive["layers_0"]["w"],.01,rtol=1e-5)
    np.testing.assert_array_equal(adaptive["layers_1"]["w"],g["layers_1"]["w"])
    assert float(global_stats["layers_1"]["gradient_retained_fraction"])<1e-20
    assert float(stats["layers_1"]["gradient_retained_fraction"])==1.
    # Zero-weight groups use a registered positive floor; NaN is not repaired.
    zero={"layers_0":{"w":jnp.zeros((8,))}}
    clipped,_=clip_gradients({"layers_0":{"w":jnp.ones((8,))}},zero,True)
    assert np.linalg.norm(np.asarray(clipped["layers_0"]["w"]))==pytest.approx(1e-5)
    bad,_=clip_gradients({"layers_0":{"w":jnp.full((8,),jnp.nan)}},zero,True)
    assert np.any(np.isnan(bad["layers_0"]["w"]))


def test_long_schedule_does_not_decay_at_old_endpoint():
    assert float(schedule(0))==0
    assert float(schedule(512))==pytest.approx(1e-5)
    assert float(schedule(2048))>9e-6
    assert float(schedule(TOTAL_STEPS))==pytest.approx(1e-6)


@pytest.mark.parametrize("adaptive",[False,True])
def test_compiled_sharded_step_and_binary_next_step_resume(tmp_path,adaptive):
    config=replace(tiny_config(),param_dtype="bfloat16",compute_dtype="bfloat16")
    model=HybridForCausalLM(config)
    mesh=create_v5e_mesh()
    tokens=jax.device_put(jnp.arange(4,dtype=jnp.int32)[None],batch_sharding(mesh))
    initial=model.init(jax.random.key(106),tokens)["params"]
    p,f=split_decoder(initial)
    pl,fl=trainable_layout(p,mesh),named_sharding_tree(f,mesh)
    p,f=jax.device_put(p,pl),jax.device_put(f,fl)
    tx=optimizer(p,adaptive)
    abstract=jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),p)
    sharded=initialize_sharded_optimizer_state(tx,p,abstract,pl,mesh)
    for x,layout in zip(jax.tree.leaves(sharded.opt_state[1].mu),jax.tree.leaves(pl)):
        assert x.sharding==layout
    step=make_step(model,tx,"DECODER-CE",0,config.mamba.head_dim,
        (pl,sharded.layout,fl,batch_sharding(mesh),replicated_sharding(mesh)),adaptive=adaptive)
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
    contract=dict(adaptive=adaptive)
    store.save("arm",contract,template,3)
    restored,meta=store.restore("arm",contract,template,dict(parameters=pl,optimizer=sharded.layout))
    assert meta["step"]==3
    for a,b in zip(jax.tree.leaves(step(p,state,f,tokens)),
                   jax.tree.leaves(step(restored["parameters"],restored["optimizer"],f,tokens))):
        np.testing.assert_array_equal(a,b)


def mock_cloud(monkeypatch,tmp_path):
    import extent.chunked_checkpoint as chunks
    import extent.recovery_durability as remote
    cloud=tmp_path/"cloud"
    def upload(files,*args,**kwargs):
        for source,name in files:
            target=cloud/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    def download(target,name,*args,**kwargs):
        source=cloud/name
        if not source.exists():return False
        Path(target).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        return True
    for module in (chunks,remote):
        monkeypatch.setattr(module,"upload_artifacts_together",upload)
        monkeypatch.setattr(module,"restore_artifact",download)
    return upload,download,cloud


def test_remote_roundtrip_and_sample_verification(tmp_path,monkeypatch):
    mock_cloud(monkeypatch,tmp_path)
    assert cloud_roundtrip(tmp_path/"local","experiment",object(),float("inf"))["verified"]
    assert list((tmp_path/"cloud").rglob("checkpoint.json"))


@pytest.mark.parametrize("status",[429,401])
def test_upload_failure_retains_cursor_and_respects_cooldown(tmp_path,monkeypatch,status):
    import extent.recovery_durability as remote
    store=RecoveryCheckpointStore(tmp_path,"test",object(),chunk_bytes=16)
    store.save("arm",{},dict(p=np.arange(10,dtype=np.float32)),1)
    calls=[]
    def fail(*a,**k):
        calls.append(1)
        exc=RuntimeError("mock HTTP failure")
        exc.response=SimpleNamespace(status_code=status)
        raise exc
    monkeypatch.setattr(remote,"upload_artifacts_together",fail)
    if status==401:
        with pytest.raises(RuntimeError):store.sync("arm",force=True)
    else:
        assert not store.sync("arm",force=True)
        assert not store.sync("arm",force=True)
        assert len(calls)==1
    assert store.sync_errors["arm"]["http_status"]==status
    assert not (store.directory("arm")/"synced.json").exists()


def test_remote_corruption_rejected(tmp_path,monkeypatch):
    import extent.recovery_durability as remote
    upload,download,cloud=mock_cloud(monkeypatch,tmp_path)
    def corrupt(target,name,*args,**kwargs):
        value=download(target,name,*args,**kwargs)
        if str(name).endswith(".npy") and value:Path(target).write_bytes(b"bad")
        return value
    monkeypatch.setattr(remote,"restore_artifact",corrupt)
    store=RecoveryCheckpointStore(tmp_path/"local","test",object())
    store.save("arm",{},dict(p=np.arange(10,dtype=np.float32)),1)
    with pytest.raises(ValueError,match="SHA"):store.sync("arm")


def test_matched_comparison_partial_and_contract_rejection():
    from scripts.m3q_layer_clip_campaign import compare_campaigns
    def metric(n):return dict(nll=n,windows=[n]*2)
    def result(exp,n):
        return dict(contract=dict(experiment=exp,optimizer=dict(clipping=str(exp),agc_ratio=None,min_weight_norm=None)),
            branches={str(s):{"DECODER-CE":dict(step=TOTAL_STEPS,
                start={"locked_test":metric(7.3),"pg19_test":metric(8)},
                final={"locked_test":metric(7),"pg19_test":metric(n)})} for s in (123,456)})
    control,adaptive=result(106,7.9),result(107,7.7)
    assert compare_campaigns(control,adaptive)["primary_gate"]
    adaptive["branches"]["456"]["DECODER-CE"]["step"]=4096
    assert compare_campaigns(control,adaptive)["primary_gate"] is None
    adaptive["contract"]["optimizer"]["lr"]=.01
    with pytest.raises(ValueError,match="unmatched"):compare_campaigns(control,adaptive)


@pytest.mark.parametrize("adaptive",[False,True])
@pytest.mark.parametrize("bad_proposal",[False,True])
def test_campaign_cloud_deadline_and_fresh_runtime_resume(tmp_path,monkeypatch,adaptive,bad_proposal):
    import scripts.m3q_layer_clip_campaign as run
    mesh=create_v5e_mesh()
    clock=[0.]
    tokens={name:np.tile(np.arange(4,dtype=np.int32),(n,1)) for name,n in
        (("train",TOTAL_STEPS),("validation",2),("locked_test",2),("pg19_test",2))}
    base={"embed_tokens":{"embedding":jnp.eye(8,dtype=jnp.bfloat16)},
          "layers_0":{"mamba":{"in_proj":{"kernel":jnp.eye(8,dtype=jnp.bfloat16)}}}}
    coordinates={"layers_0/mamba/in_proj/kernel":{"a":jnp.zeros((8,32),jnp.float32),"b":jnp.zeros((32,8),jnp.float32)}}
    class Model:
        def apply(self,variables,t):
            p=variables["params"]
            return p["embed_tokens"]["embedding"][t].astype(jnp.float32)@p["layers_0"]["mamba"]["in_proj"]["kernel"].astype(jnp.float32)
    monkeypatch.setattr(run,"load_config",lambda p:(tiny_config(),None))
    monkeypatch.setattr(run,"contract_for",lambda c,e:dict(experiment=e,data_manifest=run.token_manifest(tokens)))
    monkeypatch.setattr(run,"load_tokens",lambda c:tokens)
    monkeypatch.setattr(run,"load_sources",lambda *a:(Model(),(0,),{123:(base,coordinates),456:(base,coordinates)}))
    monkeypatch.setattr(run,"HORIZONS",(2,))
    monkeypatch.setattr(run.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(run.jax,"devices",lambda:[SimpleNamespace(platform="tpu",memory_stats=lambda:{"bytes_limit":32<<30})]*8)
    monkeypatch.setattr(run,"create_v5e_mesh",lambda d:mesh)
    monkeypatch.setattr(run,"artifact_config_from_env",lambda:object())
    monkeypatch.setattr(run,"cloud_roundtrip",lambda *a:dict(verified=True))
    messages=[]
    monkeypatch.setattr(run,"_safe_notify",lambda enabled,message:(messages.append(message) or dict(sent=True)))
    upload,download,cloud=mock_cloud(monkeypatch,tmp_path)
    monkeypatch.setattr(run,"upload_artifacts_together",upload)
    monkeypatch.setattr(run,"restore_artifact",download)
    def factory(model,tx,arm,*args,**kwargs):
        assert kwargs["adaptive"]==adaptive
        class Compiled:
            def lower(self,*a):return self
            def compile(self):return self
            def memory_analysis(self):return None
            def __call__(self,p,state,f,t):
                clock[0]=24000.
                gradients=jax.tree.map(jnp.ones_like,p)
                updates,state=tx.update(gradients,state,p)
                import optax
                return optax.apply_updates(p,updates),state,dict(finite=jnp.array(not bad_proposal),loss=jnp.array(1.),layers={})
        return Compiled()
    monkeypatch.setattr(run,"make_step",factory)
    argv=["--output-dir",str(tmp_path/"output"),"--state-dir",str(tmp_path/"states")]
    exp=107 if adaptive else 106
    if bad_proposal:
        with pytest.raises(FloatingPointError,match="proposal rejected"):
            run.main(argv,experiment=exp)
        first=json.loads((tmp_path/"output"/f"extent-m3q-layer-clip-{exp}.json").read_text())
        assert first["status"]=="failed"
    else:
        first=run.main(argv,experiment=exp)
        assert first["status"]=="deadline_partial"
    assert not first["pending_slots"]
    assert any(("failed" if bad_proposal else "finished and saved") in message for message in messages)
    slot=tmp_path/"states/seed-456/decoder-ce"
    if bad_proposal:
        row=first["branches"]["456"]["DECODER-CE"]
        assert row["failed"] and row["step"]==0
        assert json.loads((slot/"checkpoint.json").read_text())["step"]==0
        return
    assert json.loads((slot/"checkpoint.json").read_text())["step"]==1
    # Delete only verified test-local runtime files: emulate a destroyed VM.
    shutil.rmtree(tmp_path/"states")
    shutil.rmtree(tmp_path/"output")
    clock[0]=0.
    second=run.main(argv,experiment=exp)
    assert second["branches"]["456"]["DECODER-CE"]["step"]==2
    meta=json.loads((slot/"checkpoint.json").read_text())
    counts=[r for r in meta["leaves"] if r["path"][-1]=="count"]
    assert all(int(np.load(slot/r["chunks"][0]["file"]).item())==2 for r in counts)


def test_transient_retry_uses_cooldown_without_restarting_payload(tmp_path,monkeypatch):
    import extent.recovery_durability as remote
    clock=[0.]
    monkeypatch.setattr(remote.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(remote.time,"sleep",lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    upload,download,cloud=mock_cloud(monkeypatch,tmp_path)
    calls=[0]
    def transient(files,*args,**kwargs):
        calls[0]+=1
        if calls[0]==1:raise OSError("mock connection lost")
        return upload(files,*args,**kwargs)
    monkeypatch.setattr(remote,"upload_artifacts_together",transient)
    store=RecoveryCheckpointStore(tmp_path/"local","test",object())
    store.save("arm",{},dict(p=np.arange(10,dtype=np.float32)),1)
    assert store.sync_until("arm",300)
    assert clock[0]==120
    assert not store.sync_errors
