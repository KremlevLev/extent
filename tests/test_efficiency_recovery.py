"""Tiny compiled CPU evidence and mocked cloud, never a TPU result."""
from dataclasses import replace
import json
import shutil
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding,PartitionSpec as P
from extent import HybridForCausalLM,tiny_config
from extent.efficiency_recovery import optimizer,make_step,choose_lr,crossing_intervals,cost_totals,pilot_parity,schedule
from extent.efficiency_diagnostics import Probes,update_metrics,relative,json_safe
from extent.efficiency_sources import tree_digest
from extent.decoder_recovery import split_decoder,forward_parameters
from extent.recovery_subspace import initialize_corrections
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding,replicated_sharding
from extent.initialization import initialize_sharded_optimizer_state
from extent.chunked_checkpoint import ChunkedCheckpointStore
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout

def test_selection_crossing_and_charged_replays():
    rows={str(s):{a:dict(step=1024,pilot_validation=dict(nll=n),attempted_updates=1024)
        for a,n in [('LR-1e-5',9),('LR-1e-4',8),('LR-3e-4',7)]} for s in (123,456)}
    rows['456']['LR-3e-4']['failed']=True
    assert choose_lr(rows)[0]=='LR-1e-4'
    rows['456']['LR-1e-4']['step']=512
    assert choose_lr(rows)[0]=='LR-1e-5'
    rows['456']['LR-1e-5']['failed']=True
    with pytest.raises(ValueError,match='BOTH'):choose_lr(rows)
    d=[dict(step=i*1024,reproduction=dict(nll=n)) for i,n in enumerate((12,7.4,7.6,7.3,7.2))]
    assert crossing_intervals(d,7.5)==dict(reached=True,last_above=2048,first_below=3072,confirmed_at=4096)
    assert not crossing_intervals(d,7)['reached']
    assert cost_totals(rows)['input_tokens']==6*1024*256
    assert pilot_parity(dict(validation=dict(nll=8)),dict(pilot_validation=dict(nll=8)))['validation_absolute_error']==0
    with pytest.raises(ValueError,match='replay'):pilot_parity(dict(validation=dict(nll=8.1)),dict(pilot_validation=dict(nll=8)))
    assert float(schedule(0,3e-4))==0
    assert float(schedule(128,3e-4))==pytest.approx(3e-4)
    assert float(schedule(0,3e-4,True))==pytest.approx(3e-4)

@pytest.mark.parametrize('adapter',[True,False])
def test_compiled_step_exact_next_step_resume_and_frozen_vocabulary(tmp_path,adapter):
    cfg=replace(tiny_config(),param_dtype='bfloat16',compute_dtype='bfloat16')
    model=HybridForCausalLM(cfg);mesh=create_v5e_mesh()
    t=jax.device_put(jnp.arange(4,dtype=jnp.int32)[None],batch_sharding(mesh))
    initial=model.init(jax.random.key(111),t)['params']
    columns=2*64+2*cfg.mamba.mimo_rank*cfg.mamba.groups*cfg.mamba.d_state
    if adapter:
        p=initialize_corrections(initial,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=32,head_dim=cfg.mamba.head_dim)
        fixed=initial;kind='ADAPTER-CE'
    else:p,fixed=split_decoder(initial);kind='DECODER-CE'
    pl,fl=trainable_layout(p,mesh),named_sharding_tree(fixed,mesh)
    p,fixed=jax.device_put(p,pl),jax.device_put(fixed,fl)
    tx=optimizer(p,peak=3e-4,adapter=adapter)
    sharded=initialize_sharded_optimizer_state(tx,p,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),p),pl,mesh)
    state=sharded.opt_state;micro=NamedSharding(mesh,P('data',None,None));t=jax.device_put(t[None],micro)
    step=make_step(model,tx,kind,columns,cfg.mamba.head_dim,(pl,sharded.layout,fl,micro,replicated_sharding(mesh)),accumulation=1)
    for _ in range(3):
        p,state,h=step(p,state,fixed,t)
        assert bool(h['finite'])
    assert int(state[1].count)==int(state[3].count)==3
    effective=forward_parameters(kind,p,fixed,columns,cfg.mamba.head_dim)
    for key in ('embed_tokens','lm_head'):
        if key in initial:
            for a,b in zip(jax.tree.leaves(initial[key]),jax.tree.leaves(effective[key])):np.testing.assert_array_equal(a,b)
    if adapter:
        for j in cfg.mamba_layer_indices:
            np.testing.assert_array_equal(initial[f'layers_{j}']['mamba']['in_proj']['kernel'][:,columns:],effective[f'layers_{j}']['mamba']['in_proj']['kernel'][:,columns:])
    store=ChunkedCheckpointStore(tmp_path,'test',chunk_bytes=4096)
    template=dict(parameters=p,optimizer=state);store.save('MAIN',{},template,3)
    restored,meta=store.restore('MAIN',{},template,dict(parameters=pl,optimizer=sharded.layout))
    assert meta['step']==3
    for a,b in zip(jax.tree.leaves(step(p,state,fixed,t)),jax.tree.leaves(step(restored['parameters'],restored['optimizer'],fixed,t))):np.testing.assert_array_equal(a,b)

def test_actual_updates_count_bf16_rounding_without_mutation():
    cfg=tiny_config();p={'layers_0':{'norm':{'scale':jnp.ones((2,))}}}
    proposed=jax.tree.map(lambda x:x+1e-5,p)
    eb=jax.tree.map(lambda x:x.astype(jnp.bfloat16),p);ea=jax.tree.map(lambda x:x.astype(jnp.bfloat16),proposed)
    before=tree_digest(jax.device_get(p))
    report=json_safe(jax.jit(lambda a,b,c,d:update_metrics(a,b,c,d,cfg))(p,proposed,eb,ea))
    assert report['trainable']['layers_0/norm/scale']['update']['max_abs']>0
    assert report['effective']['layers_0/norm/scale']['bf16_changed_fraction']==0
    assert tree_digest(jax.device_get(p))==before
    assert float(relative(jnp.zeros(2),jnp.zeros(2))['relative_l2'])==0

def test_ce_probes_local_stream_and_rollback_are_read_only():
    from extent.qwen3_teacher import Qwen3ForCausalLM,tiny_qwen3_teacher_config
    cfg=replace(tiny_config(),num_layers=2,param_dtype='bfloat16',compute_dtype='bfloat16',
        mla=replace(tiny_config().mla,implementation='qwen3_gqa'))
    tc=tiny_qwen3_teacher_config();model=HybridForCausalLM(cfg);teacher=Qwen3ForCausalLM(tc)
    mesh=create_v5e_mesh();batch=batch_sharding(mesh)
    t=jax.device_put(jnp.arange(4,dtype=jnp.int32)[None],batch)
    p=model.init(jax.random.key(12),t)['params'];tp=teacher.init(jax.random.key(13),t)['params']
    before=tree_digest(jax.device_get(p))
    probe=Probes(model,teacher,cfg,tc,mesh,batch)
    report=probe.run(p,p,jax.device_get(tp),np.asarray(t),rollback=True,fp32=True)
    assert tree_digest(jax.device_get(p))==before
    assert report['rollback_identity_nll_error']==0
    assert all(r['nll_change']==0 for r in report['rollback'].values())
    assert len(probe.local)==2 and len(report['matched_input'][0])==2
    assert np.isfinite(report['windows'][0]['loss'])
    from scripts.m3q_paper_decoder_recovery_campaign import evaluate
    assert report['canonical_nll']['nll']==evaluate(model,p,np.asarray(t),batch)['nll']
    for row in report['windows'][0]['layers'].values():
        assert row['contribution_error']==row['initial_contribution_error']
    assert np.isfinite(report['fp32_forward']['nll'])
    for row in report['windows'][0]['components'].values():assert row['bf16_changed_fraction']==0

def test_upload_budget_and_missing_runtime(tmp_path):
    from scripts.m3q_efficiency_campaign import upload_requirement,completed,expected_tasks,summarize
    store=ChunkedCheckpointStore(tmp_path,'test');m=store.save('123/MAIN',{},dict(p=np.ones(4,np.float32)),1)
    assert upload_requirement(store,['123/MAIN'])==m['payload_bytes']
    with pytest.raises(FileNotFoundError,match='original runtime'):upload_requirement(store,['456/MAIN'])
    assert not completed({'branches':{}})
    assert len(expected_tasks({},112))==6
    assert summarize({})['gate'] is None

@pytest.mark.parametrize('experiment,cap',[(111,2),(111,1e-9),(112,35)])
def test_campaign_deadline_cloud_only_resume_and_budget(tmp_path,monkeypatch,experiment,cap):
    import scripts.m3q_efficiency_campaign as run
    import extent.chunked_checkpoint as chunks
    import extent.recovery_durability as remote
    import psutil
    mesh=create_v5e_mesh();clock=[0.];cloud=tmp_path/'cloud'
    def upload(files,*a,**k):
        for source,name in files:
            target=cloud/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    def download(target,name,*a,**k):
        source=cloud/name
        if not source.exists():return False
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);return True
    for module in (run,chunks,remote):
        monkeypatch.setattr(module,'upload_artifacts_together',upload);monkeypatch.setattr(module,'restore_artifact',download)
    tokens={n:np.tile(np.arange(4,dtype=np.int32),(2,1)) for n in ('train','validation','locked_test','pg19_test','reproduction')}
    base={'embed_tokens':{'embedding':jnp.eye(8,dtype=jnp.bfloat16)},
        'layers_0':{'mamba':{'in_proj':{'kernel':jnp.eye(8,dtype=jnp.bfloat16)}}}}
    c={'layers_0/mamba/in_proj/kernel':{'a':jnp.ones((8,32),jnp.float32),'b':jnp.zeros((32,8),jnp.float32)}}
    class Model:
        def apply(self,v,t):return v['params']['embed_tokens']['embedding'][t].astype(jnp.float32)@v['params']['layers_0']['mamba']['in_proj']['kernel'].astype(jnp.float32)
    class Probe:
        def __init__(self,*a):pass
        def run(self,*a,**k):return dict(read_only=True)
    monkeypatch.setattr(run,'TOTAL_STEPS',2);monkeypatch.setattr(run,'PROBE_STEPS',(0,2))
    monkeypatch.setattr(run,'PILOT_STEPS',2)
    def tasks(result,exp):
        if exp==112 and not result.get('selected_lr'):return [(s,a,2,True) for a in run.PEAKS for s in (123,456)]
        return [(123,'MAIN',2,False),(456,'MAIN',2,False)]
    monkeypatch.setattr(run,'expected_tasks',tasks)
    # Only shorten the pilot horizon; keep real two-seed selection/tie handling.
    if experiment==112:
        def choose(rows):
            normalized=json.loads(json.dumps(rows))
            for arms in normalized.values():
                for row in arms.values():row['step']=1024
            return choose_lr(normalized)
        monkeypatch.setattr(run,'choose_lr',choose)
    monkeypatch.setattr(run,'load_config',lambda p:(tiny_config(),None))
    monkeypatch.setattr(run,'contract_for',lambda c,e:dict(experiment=e,data_manifest=run.token_manifest(tokens)))
    monkeypatch.setattr(run,'load_tokens',lambda p:tokens)
    monkeypatch.setattr(run,'load_sources',lambda *a,**k:(Model(),(0,),{s:(base,c) for s in (123,456)},(None,None,None)))
    monkeypatch.setattr(run,'Probes',Probe)
    monkeypatch.setattr(run.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(run.jax,'devices',lambda:[SimpleNamespace(platform='tpu',memory_stats=lambda:{'bytes_limit':32<<30})]*8)
    monkeypatch.setattr(run,'create_v5e_mesh',lambda d:mesh)
    monkeypatch.setattr(run,'artifact_config_from_env',lambda:object())
    monkeypatch.setattr(run,'cloud_roundtrip',lambda *a:dict(verified=True))
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=1<<40))
    messages=[]
    monkeypatch.setattr(run,'_safe_notify',lambda enabled,message:(messages.append(message) or dict(sent=True)))
    def factory(model,tx,arm,*a,**k):
        class Compiled:
            def lower(self,*a):return self
            def compile(self):return self
            def memory_analysis(self):return None
            def __call__(self,p,state,f,t):
                import optax
                clock[0]+=27000 if experiment==111 else 1
                u,state=tx.update(jax.tree.map(jnp.ones_like,p),state,p)
                return optax.apply_updates(p,u),state,dict(finite=jnp.array(True))
        return Compiled()
    monkeypatch.setattr(run,'make_step',factory)
    argv=['--output-dir',str(tmp_path/'output'),'--checkpoint-dir',str(tmp_path/'states'),
        '--state-dir',str(tmp_path/'source'),'--max-new-hf-gb',str(cap)]
    first=run.main(argv,experiment=experiment)
    if experiment==112:
        assert first['status']=='completed' and first['selected_lr']=='LR-1e-5'
        assert first['aggregate']['cost']['attempted_updates']==16
        assert first['aggregate']['cost']['input_tokens']==4096
        assert not first['pending_slots']
        for rows in first['branches'].values():
            assert all(rows[a]['metrics_only_pilot'] for a in run.PEAKS)
            assert rows['MAIN']['durable_step']==2
        assert all('/MAIN/' in p.as_posix() for p in (cloud/'experiments/exp112-recovery-efficiency/checkpoints').rglob('checkpoint.json'))
        return
    assert first['status']=='deadline_partial'
    assert first['branches']['123']['MAIN']['step']==1
    if cap<1:
        assert first['pending_slots']==['123/MAIN'] and first['storage_error']['required_bytes']>1
        assert not (cloud/'experiments/exp111-recovery-efficiency/checkpoints/123/MAIN/checkpoint.json').exists()
        assert any('HF states pending' in m for m in messages)
        return
    assert not first['pending_slots'] and first['branches']['123']['MAIN']['durable_step']==1
    latest=cloud/'experiments/exp111-recovery-efficiency/latest.json'
    ahead=json.loads(latest.read_text());ahead['branches']['123']['MAIN']['step']=2
    ahead['branches']['123']['MAIN']['diagnostics'].append(dict(step=2,validation=dict(nll=1000)))
    latest.write_text(json.dumps(ahead))
    # Destroy only test temporary local files, emulating a cloud-only continuation.
    shutil.rmtree(tmp_path/'states');shutil.rmtree(tmp_path/'output');clock[0]=0
    second=run.main(argv,experiment=111)
    assert second['branches']['123']['MAIN']['step']==2
    assert second['aggregate']['cost']['attempted_updates']==2
    assert second['branches']['123']['MAIN']['rollbacks']==[dict(reported=2,restored=1)]
    assert not any(d.get('validation',{}).get('nll')==1000 for d in second['branches']['123']['MAIN']['diagnostics'])
    assert '2' not in second['branches']['123']['MAIN']['probes']  # deadline deferred
    assert not second['pending_slots']
    meta=json.loads((tmp_path/'states/123/MAIN/checkpoint.json').read_text())
    for r in meta['leaves']:
        if r['path'][-1]=='count':assert int(np.load(tmp_path/'states/123/MAIN'/r['chunks'][0]['file']).item())==2
    clock[0]=0
    third=run.main(argv,experiment=111)
    assert '2' in third['branches']['123']['MAIN']['probes']
    assert 'final' in third['branches']['123']['MAIN']
