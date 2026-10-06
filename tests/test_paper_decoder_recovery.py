"""EXP104 executable numerical/resume tests; these are CPU checks, not TPU evidence."""
from dataclasses import replace
import json
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM, tiny_config
from extent.decoder_recovery import split_decoder, decoder_parameters, optimizer, make_step, schedule
from extent.initialization import initialize_sharded_optimizer_state
from extent.sharding import create_v5e_mesh, batch_sharding, replicated_sharding, named_sharding_tree
from extent.chunked_checkpoint import ChunkedCheckpointStore
from extent.recovery_subspace import initialize_corrections, apply_corrections
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout, aggregate


@pytest.mark.parametrize("arm", ["ADAPTER-CE", "DECODER-CE"])
def test_compiled_updates_frozen_vocabulary_and_exact_resume(arm, tmp_path):
    config = replace(tiny_config(), param_dtype="bfloat16", compute_dtype="bfloat16")
    model = HybridForCausalLM(config)
    mesh = create_v5e_mesh()
    tokens = jax.device_put(jnp.arange(4, dtype=jnp.int32)[None], batch_sharding(mesh))
    base = model.init(jax.random.key(104), tokens)["params"]
    c = initialize_corrections(base, (0, 2), "INOUT-LORA", seed=123, rank=32, head_dim=config.mamba.head_dim)
    columns = 2*int(config.hidden_size*config.mamba.expand)+2*config.mamba.mimo_rank*config.mamba.groups*config.mamba.d_state
    for name, factors in c.items():
        factors["b"] = jnp.full_like(factors["b"], .01)
        if "/in_proj/" in name:
            factors["b"] = factors["b"].at[:, columns:].set(0)
    recovered = apply_corrections(base, c, protected_input_columns=columns, head_dim=config.mamba.head_dim)
    p, f = split_decoder(recovered) if arm == "DECODER-CE" else (c, base)
    if arm == "DECODER-CE":
        for a,b in zip(jax.tree.leaves(recovered),jax.tree.leaves(decoder_parameters(p,f))):
            np.testing.assert_array_equal(a,b)
        assert "embed_tokens" not in p and "lm_head" not in p
    layout, fixed_layout = trainable_layout(p,mesh), named_sharding_tree(f,mesh)
    p,f = jax.device_put(p,layout),jax.device_put(f,fixed_layout)
    tx = optimizer(p)
    abstract = jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),p)
    opt = initialize_sharded_optimizer_state(tx,p,abstract,layout,mesh)
    assert jax.tree.structure(opt.opt_state[1].mu) == jax.tree.structure(p)
    for value, intended in zip(jax.tree.leaves(opt.opt_state[1].mu),jax.tree.leaves(layout)):
        assert value.sharding == intended
    step = make_step(model,tx,arm,columns,config.mamba.head_dim,
                     (layout,opt.layout,fixed_layout,batch_sharding(mesh),replicated_sharding(mesh)))
    start = p
    state = opt.opt_state
    for _ in range(8):
        p,state,health = step(p,state,f,tokens)
        assert bool(health["finite"])
    assert any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(start),jax.tree.leaves(p)))
    if arm == "DECODER-CE":
        assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(p))
        assert any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(decoder_parameters(start,f)),jax.tree.leaves(decoder_parameters(p,f))))
        for key in f:
            for a,b in zip(jax.tree.leaves(f[key]),jax.tree.leaves(recovered[key])):
                np.testing.assert_array_equal(a,b)
    template = dict(parameters=p,optimizer=state)
    store = ChunkedCheckpointStore(tmp_path,"unit",chunk_bytes=4096)
    store.save("arm",{"arm":arm},template,8)
    restored,meta = store.restore("arm",{"arm":arm},template,dict(parameters=layout,optimizer=opt.layout))
    assert meta["step"]==8
    expected = step(p,state,f,tokens)
    actual = step(restored["parameters"],restored["optimizer"],f,tokens)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        np.testing.assert_array_equal(a,b)


def test_schedule_exact():
    assert float(schedule(0)) == 0
    assert float(schedule(64)) == pytest.approx(1e-5)
    assert float(schedule(2048)) == pytest.approx(1e-6)


def test_decay_excludes_multidimensional_mamba_biases():
    import optax
    p={"layers_0":{"mamba":{"b_bias":jnp.ones((8,4,8)),"mimo_x":jnp.ones((8,4,8)),
                             "dt_bias":jnp.ones((8,))}}}
    tx=optimizer(p)
    state=tx.init(p)
    zero=jax.tree.map(jnp.zeros_like,p)
    # Advance warmup so zero-gradient updates expose weight decay alone.
    for _ in range(65):
        u,state=tx.update(zero,state,p)
    np.testing.assert_array_equal(u["layers_0"]["mamba"]["b_bias"],0)
    np.testing.assert_array_equal(u["layers_0"]["mamba"]["dt_bias"],0)
    assert np.all(np.asarray(u["layers_0"]["mamba"]["mimo_x"])<0)


def test_chunk_corruption_and_interrupted_generation(tmp_path, monkeypatch):
    import extent.chunked_checkpoint as module
    store=ChunkedCheckpointStore(tmp_path,"unit",chunk_bytes=16)
    payload={"parameters":{"kernel":jnp.arange(32,dtype=jnp.float32)},"optimizer":{"count":jnp.array(7)}}
    meta=store.save("arm",{},payload,7)
    save=module.np.save
    monkeypatch.setattr(module.np,"save",lambda *a,**k: (_ for _ in ()).throw(OSError("disk interrupted")))
    with pytest.raises(OSError):
        store.save("arm",{},payload,8)
    monkeypatch.setattr(module.np,"save",save)
    assert store.restore("arm",{},payload)[1]["step"]==7
    file=store.directory("arm")/meta["leaves"][0]["chunks"][0]["file"]
    file.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="SHA"):
        store.restore("arm",{},payload)


@pytest.mark.parametrize("status",[429,401])
def test_remote_manifest_not_advanced_after_failed_chunk(tmp_path,monkeypatch,status):
    import extent.chunked_checkpoint as module
    store=ChunkedCheckpointStore(tmp_path,"unit",hub=object(),chunk_bytes=16)
    store.save("arm",{}, {"parameters":jnp.arange(16,dtype=jnp.float32)},1)
    calls=[]
    def upload(files,*a,**k):
        calls.append(files[0][1])
        exc=RuntimeError("test HTTP failure")
        exc.response=SimpleNamespace(status_code=status)
        raise exc
    monkeypatch.setattr(module,"upload_artifacts_together",upload)
    if status==401:
        with pytest.raises(RuntimeError):store.sync("arm")
    else:
        assert store.sync("arm") is False
        assert store.sync("arm") is False
        assert len(calls)==1
    assert not any(p.endswith("checkpoint.json") for p in calls)


def test_gate_partial_and_paired():
    assert aggregate({})["primary_gate"] is None
    def metric(n):return dict(nll=n,windows=[n]*24)
    result=dict(original_qwen={"locked_test":metric(3)},branches={})
    for seed in (123,456):
        result["branches"][str(seed)]={}
        for arm,n in (("ADAPTER-CE",7.2),("DECODER-CE",7.0)):
            result["branches"][str(seed)][arm]=dict(step=2048,start={"locked_test":metric(7.3),"pg19_test":metric(8)},
                final={"locked_test":metric(n),"pg19_test":metric(n+.5)})
    assert aggregate(result)["primary_gate"] is True
    assert aggregate(result)["transfer_gate"] is True
    result["branches"]["456"]["DECODER-CE"]["failed"]=True
    assert aggregate(result)["primary_gate"] is None


@pytest.mark.parametrize("bad_proposal", [False, True])
def test_runner_deadline_resume_keeps_optimizer_cursor(tmp_path, monkeypatch, bad_proposal):
    import shutil
    import scripts.m3q_paper_decoder_recovery_campaign as run
    import extent.chunked_checkpoint as chunks
    mesh = create_v5e_mesh()
    clock = [0.0]
    config = tiny_config()
    tokens = {name: np.tile(np.arange(4,dtype=np.int32), (n,1)) for name,n in
              (("train",2048),("validation",2),("locked_test",2),("pg19_test",2))}
    base = {"embed_tokens":{"embedding":jnp.eye(8,dtype=jnp.bfloat16)},
            "layers_0":{"mamba":{"in_proj":{"kernel":jnp.eye(8,dtype=jnp.bfloat16)}}}}
    coordinates={"layers_0/mamba/in_proj/kernel":{"a":jnp.zeros((8,32),jnp.float32),
                                                "b":jnp.zeros((32,8),jnp.float32)}}
    class Model:
        def apply(self, variables, t):
            p=variables["params"]
            return p["embed_tokens"]["embedding"][t].astype(jnp.float32) @ p["layers_0"]["mamba"]["in_proj"]["kernel"].astype(jnp.float32)
    monkeypatch.setattr(run,"load_config",lambda p:(config,None))
    monkeypatch.setattr(run,"contract_for",lambda c:{"data_manifest":run.token_manifest(tokens)})
    monkeypatch.setattr(run,"load_tokens",lambda c:tokens)
    monkeypatch.setattr(run,"load_sources",lambda *a:(Model(),(0,),{123:(base,coordinates),456:(base,coordinates)}))
    monkeypatch.setattr(run,"HORIZONS",(2,))
    monkeypatch.setattr(run.time,"monotonic",lambda:clock[0])
    real_devices=jax.devices()
    monkeypatch.setattr(run.jax,"devices",lambda:[SimpleNamespace(platform="tpu",memory_stats=lambda:{"bytes_limit":32<<30})]*8)
    monkeypatch.setattr(run,"create_v5e_mesh",lambda d:mesh)
    monkeypatch.setattr(run,"artifact_config_from_env",lambda:object())
    cloud=tmp_path/"cloud"
    def upload(files,*a,**k):
        for source,name in files:
            target=cloud/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    def download(target,name,*a,**k):
        source=cloud/name
        if not source.exists():return False
        Path(target).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        return True
    from pathlib import Path
    monkeypatch.setattr(run,"upload_artifacts_together",upload)
    monkeypatch.setattr(chunks,"upload_artifacts_together",upload)
    monkeypatch.setattr(run,"restore_artifact",download)
    monkeypatch.setattr(chunks,"restore_artifact",download)
    def factory(model,tx,arm,*a):
        class Compiled:
            def lower(self,*a):return self
            def compile(self):return self
            def memory_analysis(self):return None
            def __call__(self,p,state,f,t):
                clock[0]=24000.0
                updates,state=tx.update(jax.tree.map(jnp.ones_like,p),state,p)
                import optax
                return optax.apply_updates(p,updates),state,{"finite":jnp.array(not bad_proposal),"loss":jnp.array(1.)}
        return Compiled()
    monkeypatch.setattr(run,"make_step",factory)
    argv=["--output-dir",str(tmp_path/"output"),"--state-dir",str(tmp_path/"states")]
    first=run.main(argv)
    assert first["status"]=="deadline_partial"
    if bad_proposal:
        assert first["branches"]["123"]["ADAPTER-CE"]["failed"]
        assert first["branches"]["123"]["ADAPTER-CE"]["step"]==0
        meta=json.loads((tmp_path/"states/seed-123/adapter-ce/checkpoint.json").read_text())
        assert meta["step"]==0
        assert first["aggregate"]["primary_gate"] is None
        return
    assert first["branches"]["123"]["ADAPTER-CE"]["step"]==1
    meta=json.loads((tmp_path/"states/seed-123/adapter-ce/checkpoint.json").read_text())
    counts=[r for r in meta["leaves"] if r["path"][-1]=="count"]
    assert all(int(np.load(tmp_path/"states/seed-123/adapter-ce"/r["chunks"][0]["file"]).item())==1 for r in counts)
    clock[0]=0
    second=run.main(argv)
    assert second["branches"]["123"]["ADAPTER-CE"]["step"]==2
    meta=json.loads((tmp_path/"states/seed-123/adapter-ce/checkpoint.json").read_text())
    counts=[r for r in meta["leaves"] if r["path"][-1]=="count"]
    assert all(int(np.load(tmp_path/"states/seed-123/adapter-ce"/r["chunks"][0]["file"]).item())==2 for r in counts)
    monkeypatch.setattr(run.jax,"devices",lambda:real_devices)
