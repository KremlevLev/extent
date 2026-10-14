"""Tiny CPU/virtual8 engineering checks, not a full-model TPU result."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json,time
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM,tiny_config
from extent.rank_growth import ARMS,initialize,materialize,make_step,optimizers,select
from extent.efficiency_recovery import optimizer,make_step as legacy_step
from extent.recovery_subspace import initialize_corrections
from extent.plateau_forensics import GuardedJit,state_digest
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding
from extent.chunked_checkpoint import ChunkedCheckpointStore
from scripts.m3q_rank_growth_115_campaign import init_payload,run_branch,Session,summarize
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout

def fixture():
    cfg=replace(tiny_config(),num_layers=2,attention_layer_indices=(1,),param_dtype='bfloat16',compute_dtype='bfloat16',logits_dtype='float32')
    model=HybridForCausalLM(cfg);mesh=create_v5e_mesh();t=jax.device_put(jnp.arange(8,dtype=jnp.int32)[None],batch_sharding(mesh))
    base=model.init(jax.random.key(115),t)['params'];base=jax.device_put(base,named_sharding_tree(base,mesh))
    old=initialize_corrections(base,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=32,head_dim=cfg.mamba.head_dim)
    old=jax.tree.map(lambda x:x+jax.random.normal(jax.random.key(3),x.shape)*.001,old)
    old=jax.device_put(old,trainable_layout(old,mesh));tx=optimizer(old,peak=3e-4,adapter=True)
    state=tx.init(old);state=list(state);state[1]=state[1]._replace(count=jnp.array(32768));state[3]=state[3]._replace(count=jnp.array(32768));state=tuple(state)
    columns=2*int(cfg.hidden_size*cfg.mamba.expand)+2*cfg.mamba.mimo_rank*cfg.mamba.groups*cfg.mamba.d_state
    return cfg,model,mesh,t,base,old,state,columns

def test_all_expansions_start_bitwise_equal_and_share_new_direction_prefixes():
    cfg,model,mesh,t,base,old,state,cols=fixture()
    expected=materialize(base,{'old':old},cols,ARMS['R32'],0)
    for arm in ARMS:
        q=initialize(old,arm);actual=materialize(base,q,cols,ARMS[arm],0)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):np.testing.assert_array_equal(a,b)
    small=initialize(old,'G64');large=initialize(old,'G256');ladder=initialize(old,'LADDER256')
    for name in old:
        np.testing.assert_array_equal(small['new0'][name]['a'],large['new0'][name]['a'][:,:32])
        np.testing.assert_array_equal(ladder['new1'][name]['a'],large['new0'][name]['a'][:,32:96])

def test_control_matches_original_adam_next_update():
    cfg,model,mesh,t,base,old,state,cols=fixture()
    tx=optimizer(old,peak=3e-4,adapter=True)
    expected=legacy_step(model,tx,'ADAPTER-CE',cols,cfg.mamba.head_dim)(old,state,base,t[None])
    actual=GuardedJit(make_step(model,cols,'R32'))({'old':old},{'old':state},base,t,jnp.array(0))
    assert bool(actual[2]['finite'])
    for a,b in zip(jax.tree.leaves(actual[0]['old']),jax.tree.leaves(expected[0])):np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
    for a,b in zip(jax.tree.leaves(actual[1]['old']),jax.tree.leaves(expected[1])):np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)

@pytest.mark.parametrize('arm',['G64','ADD256','LADDER256'])
def test_active_clocks_freeze_protected_columns_and_exact_sharded_resume(tmp_path,arm):
    cfg,model,mesh,t,base,old,state,cols=fixture()
    payload,layouts=init_payload(old,state,arm,mesh);q,s=payload['parameters'],payload['optimizer'];step=GuardedJit(make_step(model,cols,arm))
    cursor=1024 if arm=='LADDER256' else 0
    for _ in range(3):
        q2,s2,h=step(q,s,base,t,jnp.asarray(cursor));assert bool(h['finite'])
        q=jax.device_put(q2,layouts['parameters']);s=jax.device_put(s2,layouts['optimizer']);cursor+=1
    if arm=='ADD256':
        for a,b in zip(jax.tree.leaves(q['old']),jax.tree.leaves(old)):np.testing.assert_array_equal(a,b)
        assert int(s['old'][1].count)==32768
    if arm=='LADDER256':assert int(s['new1'][1].count)==int(s['new2'][1].count)==0
    for group in q.values():
        for name,factors in group.items():
            if '/in_proj/' in name and group is not q['old']:assert bool(jnp.all(factors['b'][:,cols:]==0))
    store=ChunkedCheckpointStore(tmp_path,'test',chunk_bytes=4096)
    store.save('candidate',{},dict(parameters=q,optimizer=s),cursor)
    restored,_=store.restore('candidate',{},dict(parameters=q,optimizer=s),layouts)
    a=step(q,s,base,t,jnp.asarray(cursor));b=step(restored['parameters'],restored['optimizer'],base,t,jnp.asarray(cursor))
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)

def test_validation_only_deterministic_selection_and_failure_filter():
    rows={a:dict(step=4096,evaluations=[dict(step=i,validation={'nll':v}) for i,v in ((3072,7.2),(4096,7.3))],
        final={'locked_wiki':{'nll':-100 if a=='G256' else 100}}) for a in ('R32','G64','G128','G256')}
    chosen,scores=select(rows);assert chosen==['G128','G256']
    rows['G128']['failed']=True;assert select(rows)[0]==['G256','G64']
    assert summarize({'branches':{'G64':dict(step=0,evaluations=[])}})['progress']['G64']['last_validation'] is None

def test_real_branch_runs_and_resumes_without_mutating_source(tmp_path,monkeypatch):
    from scripts import m3q_rank_growth_115_campaign as runner
    cfg,model,mesh,t,base,old,state,cols=fixture();before=state_digest(jax.device_get({'p':old,'s':state,'base':base}))
    monkeypatch.setattr(runner,'EVAL_STEPS',(0,1,3))
    store=ChunkedCheckpointStore(tmp_path/'checkpoints','test',chunk_bytes=4096)
    class LocalSession:
        contract={};result={'branches':{},'cost':{'attempted_updates':0,'input_tokens':0,'ce_targets':0}}
        def check(self):pass
        def persist(self,*args,**kwargs):pass
        def emit(self,*args):pass
        def checkpoint(self,arm,payload,step):store.save(arm,{'arm':arm},payload,step)
    sess=LocalSession();sess.store=store;tokens={'train':np.asarray(t).repeat(4,axis=0),'validation':np.asarray(t)}
    kernels={};run_branch(sess,'G64',1,model,base,old,state,cols,mesh,tokens,kernels)
    run_branch(sess,'G64',3,model,base,old,state,cols,mesh,tokens,kernels)
    assert sess.result['branches']['G64']['step']==3
    assert sess.result['cost']['attempted_updates']==3
    assert [r['step'] for r in sess.result['branches']['G64']['evaluations']]==[0,1,3]
    assert before==state_digest(jax.device_get({'p':old,'s':state,'base':base}))

def test_session_cloud_restore_cap_and_rate_limit(tmp_path,monkeypatch):
    from scripts import m3q_rank_growth_115_campaign as runner
    import extent.chunked_checkpoint as chunks
    import extent.recovery_durability as durable
    cloud={}
    def upload(files,hub,**kwargs):
        for local,remote in files:cloud[remote]=Path(local).read_bytes()
    def restore(local,remote,hub):
        if remote not in cloud:return False
        local=Path(local);local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(cloud[remote]);return True
    for module in (runner,durable):
        monkeypatch.setattr(module,'upload_artifacts_together',upload)
        monkeypatch.setattr(module,'restore_artifact',restore)
    monkeypatch.setattr(chunks,'restore_artifact',restore)
    args=SimpleNamespace(max_wall_hours=.5,save_reserve_minutes=10,max_new_hf_gb=.01,
        output_dir=str(tmp_path/'out'),checkpoint_dir=str(tmp_path/'state'),telegram=False)
    s=Session(args,{'test':1},object());s.result['branches']['G64']=dict(step=3,evaluations=[])
    payload={'parameters':np.arange(32,dtype=np.float32),'optimizer':{'count':np.array(3,np.int32)}}
    s.checkpoint('G64',payload,3);assert s.persist(force=True)
    assert not s.result['pending_slots'] and s.result['branches']['G64']['durable_step']==3
    assert s.invocation['uploaded_bytes']>128
    args2=SimpleNamespace(**dict(vars(args),output_dir=str(tmp_path/'new-out'),checkpoint_dir=str(tmp_path/'new-state')))
    fresh=Session(args2,{'test':1},object())
    restored,meta=fresh.store.restore('G64',{'test':1,'arm':'G64'},payload)
    assert meta['step']==3;np.testing.assert_array_equal(restored['parameters'],payload['parameters'])
    # Force must still honor a known429 report cooldown.
    fresh.result['report_sync_error']={'http_status':429};fresh.next_sync=time.monotonic()+3600
    count=len(cloud);assert not fresh.persist(force=True,checkpoints=False);assert len(cloud)==count
    fresh.next_sync=0;fresh.args.max_new_hf_gb=1e-9
    assert not fresh.persist(force=True,checkpoints=False)
    assert fresh.result['report_sync_error']=='invocation payload cap'

def test_report_final_cursor_cannot_skip_missing_checkpoint_training():
    from scripts.m3q_rank_growth_115_campaign import reconcile_cursor
    row=dict(step=16384,evaluations=[{'step':n} for n in (0,4096,8192,16384)],
        final={'locked_wiki':{'nll':6}},attempted_updates=16384)
    reconcile_cursor(row,4096)
    assert row['step']==4096 and 'final' not in row
    assert [r['step'] for r in row['evaluations']]==[0,4096]
    assert row['attempted_updates']==16384
