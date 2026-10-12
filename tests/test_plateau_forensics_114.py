"""Local engineering evidence only; no TPU/model quality claims."""
from dataclasses import replace
from types import SimpleNamespace
import json,shutil
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM
from extent.config import tiny_config
from extent.qwen3_teacher import Qwen3ForCausalLM
from extent.plateau_forensics import (ForensicLM,GuardedJit,teacher_config_for,
    mamba_trace,readout_init,readout_logits,readout_step,readout_optimizer,
    select_layers,paired_summary,token_metrics)
from extent.layers.mamba3 import Mamba3MIMO
from extent.layers.common import RMSNorm
from extent.sharding import create_v5e_mesh,batch_sharding,named_sharding_tree
from extent.chunked_checkpoint import ChunkedCheckpointStore
from extent.plateau_data import pack_panels

def fixture(dtype='bfloat16'):
    base=tiny_config();cfg=replace(base,num_layers=2,attention_layer_indices=(1,),
        mla=replace(base.mla,implementation='qwen3_gqa'),param_dtype=dtype,compute_dtype=dtype,
        tie_word_embeddings=True)
    tc=teacher_config_for(cfg);model=HybridForCausalLM(cfg);teacher=Qwen3ForCausalLM(tc)
    mesh=create_v5e_mesh();tokens=jax.device_put(jnp.arange(8,dtype=jnp.int32)[None],batch_sharding(mesh))
    p=model.init(jax.random.key(114),tokens)['params'];q=teacher.init(jax.random.key(115),tokens)['params']
    p=jax.device_put(p,named_sharding_tree(p,mesh));q=jax.device_put(q,named_sharding_tree(q,mesh))
    return cfg,tc,model,teacher,p,q,tokens

@pytest.mark.parametrize('dtype',['bfloat16','float32'])
def test_hook_identity_rollback_boundary_teacher_and_reset_scopes(dtype):
    cfg,tc,model,teacher,p,q,t=fixture(dtype)
    expected=jax.jit(lambda p,t:model.apply({'params':p},t))(p,t)
    _,qh=teacher.apply({'params':q},t,return_hidden_states=True)
    empty=jnp.zeros(cfg.num_layers,bool)
    for mode in ('identity','rollback','teacher','boundary','reset'):
        hook=ForensicLM(cfg,tc,mode,2 if mode=='reset' else 0)
        call=GuardedJit(lambda p,r,m,h,t:hook.apply({'params':p},t,r,m,h))
        actual=call(p,q if mode=='teacher' else p,empty,qh,t)
        if dtype=='float32':np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=0)
        else:
            from extent.plateau_forensics import losses
            assert abs(float(losses(actual,t)-losses(expected,t)))<.01
    start={k:jax.tree.map(lambda x:x*.99 if x.ndim==2 else x,v) if k.startswith('layers_') else v for k,v in p.items()}
    hook=ForensicLM(cfg,tc,'rollback')
    actual=jax.jit(lambda p,r,t:hook.apply({'params':p},t,r,jnp.ones(cfg.num_layers,bool),qh))(p,start,t)
    np.testing.assert_allclose(actual,model.apply({'params':start},t),atol=.02 if dtype=='bfloat16' else 1e-4,rtol=0)
    # Replace final boundary with identical current boundary = identity.
    _,h=model.apply({'params':p},t,return_hidden_states=True)
    boundary=ForensicLM(cfg,tc,'boundary')
    np.testing.assert_allclose(boundary.apply({'params':p},t,p,jnp.array([False,True]),h),expected,atol=.02 if dtype=='bfloat16' else 1e-4,rtol=0)
    # Retained attention teacher-mixer neighbor is explicitly a null control.
    teacher_hook=ForensicLM(cfg,tc,'teacher')
    np.testing.assert_allclose(teacher_hook.apply({'params':p},t,q,jnp.array([False,True]),h),expected,atol=.02 if dtype=='bfloat16' else 1e-4,rtol=0)
    # Real Mamba->teacher attention replacement changes the function.
    altered=teacher_hook.apply({'params':p},t,q,jnp.array([True,False]),h)
    assert np.max(np.abs(np.asarray(altered)-np.asarray(expected)))>1e-4
    reset=ForensicLM(cfg,tc,'reset',0)
    np.testing.assert_allclose(reset.apply({'params':p},t,p,jnp.array([True,False]),h),expected,atol=.02 if dtype=='bfloat16' else 1e-4,rtol=0)

@pytest.mark.parametrize('dtype',['bfloat16','float32'])
def test_recurrence_trace_parity_and_explicit_all_state_reset(dtype):
    cfg,_,_,_,p,_,t=fixture(dtype)
    x=p['embed_tokens']['embedding'][t]
    module=Mamba3MIMO(cfg.hidden_size,cfg.mamba,getattr(jnp,dtype),getattr(jnp,dtype))
    mp=p['layers_0']['mamba']
    expected=module.apply({'params':mp},x)
    output,report=jax.jit(lambda p,x:mamba_trace(x,p,cfg))(mp,x)
    np.testing.assert_allclose(output,expected,atol=1e-5,rtol=0)
    reset=jax.jit(lambda p,x:mamba_trace(x,p,cfg,2)[0])(mp,x)
    chunked=jnp.concatenate([module.apply({'params':mp},x[:,i:i+2]) for i in range(0,x.shape[1],2)],1)
    np.testing.assert_allclose(reset,chunked,atol=1e-5,rtol=0)
    assert np.asarray(report['time']['state']['rms']).shape==(8,)
    assert bool(jnp.all(jnp.isfinite(report['time']['rank_participation'])))

def test_readout_zero_identity_checkpoint_exact_next_update_and_frozen_patient(tmp_path):
    cfg,_,model,_,p,_,t=fixture()
    from extent.efficiency_sources import tree_digest
    before=tree_digest(jax.device_get(p));logits,h=model.apply({'params':p},t,return_hidden_states=True)
    features=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps,jnp.bfloat16).apply({'params':p['norm']},h[-1])
    vocabulary={'embed_tokens':p['embed_tokens']};c=readout_init(cfg.hidden_size,114123)
    np.testing.assert_array_equal(readout_logits(features,c,vocabulary,cfg),logits)
    from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout
    from extent.initialization import initialize_sharded_optimizer_state
    mesh=create_v5e_mesh();pl=trainable_layout(c,mesh);c=jax.device_put(c,pl)
    tx=readout_optimizer();initial=initialize_sharded_optimizer_state(tx,c,
        jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),c),pl,mesh)
    state=initial.opt_state;step=GuardedJit(readout_step(cfg))
    for _ in range(3):
        c,state,health=step(c,state,features,vocabulary,t);assert bool(health['finite'])
        c=jax.device_put(c,pl);state=jax.device_put(state,initial.layout)
    assert int(state[1].count)==int(state[2].count)==3
    store=ChunkedCheckpointStore(tmp_path,'test',chunk_bytes=4096)
    payload=dict(parameters=c,optimizer=state);store.save('READOUT',{},payload,3)
    restored,_=store.restore('READOUT',{},payload,dict(parameters=pl,optimizer=initial.layout))
    a=step(c,state,features,vocabulary,t);b=step(restored['parameters'],restored['optimizer'],features,vocabulary,t)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
    assert tree_digest(jax.device_get(p))==before

def test_selection_and_paired_signs_target_indices_and_provenance():
    rows=[dict(family=f,layer=j,delta_nll=n) for f in ('rollback','teacher','boundary') for j,n in ((2,-1),(0,-1),(1,.5))]
    selection=select_layers(rows);assert selection['single']==dict(rollback=0,teacher=0,boundary=0)
    assert selection['pair']==[0,2]
    result=paired_summary([3,4],[2,5]);assert result['delta_nll']==0 and result['fraction_improved']==.5
    with pytest.raises(FloatingPointError):paired_summary([3],[np.nan])
    z=jnp.eye(4)[jnp.array([[0,1,2]])];tokens=jnp.array([[0,1,2]])
    metrics=token_metrics(z,tokens)
    assert metrics['nll'].shape==(1,2) and metrics['rank'].tolist()==[[2,2]]
    class Tokenizer:
        eos_token_id=99
        def __call__(self,text,**kw):return {'input_ids':list(range(len(text)))}
    tokens,meta=pack_panels(Tokenizer(),[' = One = ',' abc ',' def '],{'A':(0,2),'B':(8,2)},length=4)
    expected=np.asarray(list(range(9))+[99]+list(range(5))+[99]+list(range(5))+[99],np.int32)
    np.testing.assert_array_equal(tokens['A'].ravel(),expected[:8]);np.testing.assert_array_equal(tokens['B'].ravel(),expected[8:16])
    assert meta['articles']['A'][0]==['article-row-0']
    with pytest.raises(ValueError,match='capacity'):pack_panels(Tokenizer(),['a'],{'A':(0,4)},length=4)

def test_session_cloud_cursor_upload_cap_and_429_cooldown(tmp_path,monkeypatch):
    import scripts.m3q_plateau_forensics_114_campaign as run
    import extent.chunked_checkpoint as chunks
    import extent.recovery_durability as durable
    from extent.hf_artifact_sync import HubArtifactConfig
    cloud=tmp_path/'cloud';clock=[100.];calls=[]
    def upload(files,*a,**kw):
        calls.append(list(files))
        for src,name in files:
            dest=cloud/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    def download(dest,name,*a,**kw):
        src=cloud/name
        if not src.exists():return False
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);return True
    for mod in (run,chunks,durable):
        monkeypatch.setattr(mod,'upload_artifacts_together',upload);monkeypatch.setattr(mod,'restore_artifact',download)
    monkeypatch.setattr(run.time,'monotonic',lambda:clock[0])
    args=SimpleNamespace(output_dir=str(tmp_path/'output'),checkpoint_dir=str(tmp_path/'checkpoints'),
        max_wall_hours=1.,save_reserve_minutes=10,max_new_hf_gb=1.,telegram=False)
    hub=HubArtifactConfig('test','dataset','none');s=run.Session(args,{'test':True},hub)
    s.result['readout']['123']=dict(step=7);payload=dict(x=np.arange(8,dtype=np.float32))
    s.checkpoint('123/READOUT',payload,7);assert s.persist(force=True)
    assert s.result['readout']['123']['durable_step']==7 and not s.result['pending_slots']
    assert s.task('done',lambda:4)==4
    # Destroy all local runtime files; restore from cloud, never rely on RAM.
    shutil.rmtree(tmp_path/'output');shutil.rmtree(tmp_path/'checkpoints')
    s2=run.Session(args,{'test':True},hub)
    restored=s2.store.restore('123/READOUT',dict(s2.contract,slot='123/READOUT'),payload)
    np.testing.assert_array_equal(restored[0]['x'],payload['x'])
    class RateLimit(Exception):response=SimpleNamespace(status_code=429)
    monkeypatch.setattr(run,'upload_artifacts_together',lambda *a,**k:(_ for _ in ()).throw(RateLimit()))
    assert not s2.persist(force=True);after=s2.next_sync
    assert after==clock[0]+3600
    # Force never bypasses a known429 cooldown.
    assert not s2.persist(force=True) and s2.next_sync==after
    s2.args.max_new_hf_gb=1e-12;s2.next_sync=0
    assert not s2.persist(force=True) and s2.result['report_sync_error']=='invocation upload cap'

def test_aggregation_does_not_select_using_B_or_C_or_impute_missing():
    from scripts.m3q_plateau_forensics_114_campaign import choose_readout,analyze_report
    r=dict(contract={'data_manifest':{n:dict(shape=[2,256]) for n in ('A','B','C_wiki','C_pg19')}},tasks={},
        choices={},readout={str(s):{'curve':{'1024':{'nll':7},'4096':{'nll':8}}} for s in (123,456)})
    for seed in (123,456):
        for temp in (.5,1.,2.,4.,8.):
            for i in range(2):r['tasks'][f'readout/{seed}/A/T-{temp}/{i}']=dict(value=abs(temp-2)+7)
        r['tasks'][f'readout/{seed}/C_wiki/fit-4096/0']=dict(value=-100)
    choose_readout(r)
    assert r['choices']['readout_endpoint']==1024 and r['choices']['temperature']==2
    assert analyze_report(r)['intervention_signals']['teacher']['status']=='INCOMPLETE'
    del r['tasks']['readout/456/A/T-0.5/1']
    with pytest.raises(ValueError,match='incomplete'):choose_readout(r)

def test_readout_interrupted_at_endpoint_cloud_resume_then_full_patient_modules(tmp_path,monkeypatch):
    import scripts.m3q_plateau_forensics_114_campaign as run
    import extent.chunked_checkpoint as chunks
    import extent.recovery_durability as remote
    from extent.hf_artifact_sync import HubArtifactConfig
    from extent.recovery_subspace import initialize_corrections
    from extent.efficiency_recovery import optimizer
    from extent.efficiency_sources import tree_digest
    from extent.plateau_forensics import teacher_config_for
    cloud=tmp_path/'cloud'
    def upload(files,*a,**kw):
        for src,name in files:
            dst=cloud/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    def restore(dst,name,*a,**kw):
        src=cloud/name
        if not src.exists():return False
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);return True
    for mod in (run,chunks,remote):
        monkeypatch.setattr(mod,'upload_artifacts_together',upload);monkeypatch.setattr(mod,'restore_artifact',restore)
    monkeypatch.setattr(run,'DISCOVERY_WINDOWS',1);monkeypatch.setattr(run,'FIT_STEPS',3)
    monkeypatch.setattr(run,'ENDPOINTS',(1,3))
    cfg,tc,model,teacher,p,q,t=fixture()
    # Two Mamba layers permit a genuine pair intervention on the tiny model.
    cfg=replace(cfg,num_layers=3);tc=teacher_config_for(cfg)
    model=HybridForCausalLM(cfg);teacher=Qwen3ForCausalLM(tc)
    p=model.init(jax.random.key(114),t)['params'];q=teacher.init(jax.random.key(115),t)['params']
    mesh=create_v5e_mesh();c=initialize_corrections(p,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=32,head_dim=cfg.mamba.head_dim)
    from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout
    c=jax.device_put(c,trainable_layout(c,mesh));state=optimizer(c,peak=3e-4,adapter=True).init(c)
    tokens={name:np.tile(np.arange(8,dtype=np.int32),(3 if name=='FIT' else 2,1))
        for name in ('A','B','C_wiki','C_pg19','FIT','T_seen','T_new')}
    contract=dict(data_manifest={n:dict(shape=list(v.shape)) for n,v in tokens.items()})
    args=SimpleNamespace(output_dir=str(tmp_path/'output'),checkpoint_dir=str(tmp_path/'states'),
        max_wall_hours=1.,save_reserve_minutes=10.,max_new_hf_gb=1.,telegram=False)
    hub=HubArtifactConfig('test','dataset','none');s=run.Session(args,contract,hub)
    patient=run.Patient(cfg,model,teacher,q,tc,p,c,state,mesh,s,123,tokens)
    from extent.plateau_forensics import state_digest
    frozen=state_digest(jax.device_get({'p':p,'c':c,'state':state}))
    original=s.task
    def interrupt(key,fn):
        if key=='readout/123/A/fit-1/0':raise TimeoutError('simulated cloud interruption')
        return original(key,fn)
    monkeypatch.setattr(s,'task',interrupt)
    with pytest.raises(TimeoutError):patient.fit()
    assert s.result['readout']['123']['step']==1 and s.result['readout']['123']['curve']=={}
    assert s.persist(force=True)
    shutil.rmtree(tmp_path/'output');shutil.rmtree(tmp_path/'states')
    continued=run.Session(args,contract,hub)
    patient=run.Patient(cfg,model,teacher,q,tc,p,c,state,mesh,continued,123,tokens)
    patient.fit()
    assert continued.result['readout']['123']['step']==3
    assert continued.result['readout']['123']['attempted_updates']==3
    assert set(continued.result['readout']['123']['curve'])=={'1','3'}
    assert state_digest(jax.device_get({'p':p,'c':c,'state':state}))==frozen
    # Exercise actual module methods, not a reimplementation of their formulae.
    patient.profile();patient.sweep();patient.temperatures();patient.optimizer_probe();patient.dynamics();patient.precision()
    continued.result['choices']['temperature']=1.
    continued.result['choices']['readout_endpoint']=3
    patient.locked(dict(sweep='completed',readout='completed',optimizer='completed',dynamics='completed'))
    assert state_digest(jax.device_get({'p':p,'c':c,'state':state}))==frozen
    assert len(continued.result['tables']['interventions'])>10
