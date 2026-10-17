"""Long-session source, clock, notification and initialization invariants."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from extent import HybridForCausalLM,tiny_config
from extent.model import _attention_source_config
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.competitor_recovery import initialize,optimizer,arm_for
from extent.reset_followup import recipes as old_recipes,block_forward
from extent.reset_long_recovery import recipes,prepare
from extent.competitor_tpu import token_nll

def patient():
    cfg=replace(tiny_config(),mla=replace(tiny_config().mla,implementation='qwen3_gqa'),param_dtype='float32',compute_dtype='float32',logits_dtype='float32',tie_word_embeddings=True,remat_policy='none')
    t=jnp.arange(8,dtype=jnp.int32)[None]%cfg.vocab_size
    base=HybridForCausalLM(cfg).init(jax.random.key(4),t)['params']
    cols=2*int(cfg.hidden_size*cfg.mamba.expand)+2*cfg.mamba.mimo_rank*cfg.mamba.groups*cfg.mamba.d_state
    old=initialize_corrections(base,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=8,head_dim=cfg.mamba.head_dim)
    p=SimpleNamespace(cfg=cfg,base=base,current=apply_corrections(base,old,protected_input_columns=cols),coordinates=old,columns=cols,tc=_attention_source_config(cfg))
    return p,t



def test_registered_recipes_restore_or_reset_intended_state_and_rates():
    assert [s['source'] for s in recipes(143)]==['136-all','136-projections']
    a,b=recipes(144)
    assert a['source']==b['source']=='138-ce'
    assert b['mamba_lr']==a['mamba_lr']/3 and b['norm_lr']==a['norm_lr']/3
    assert [s['mode'] for s in recipes(145)]==['fresh-adam','fold-new']
    assert all(s['source_seed']==456 and s['source']=='111-456' for s in recipes(146))
    from scripts.m3q_reset_long_campaign import main,session_budget
    for e in range(143,147):assert main(['--experiment',str(e),'--plan-only'])['horizon']==131072
    assert session_budget(143,11.7,30)[0]==pytest.approx(11.2*3600)


def test_fresh_factor_stage_preserves_source_function_and_shared_norm():
    p,t=patient();source_spec=old_recipes(136)[0];q=initialize(p,source_spec)
    i=p.cfg.mamba_layer_indices[0];q[str(i)]['lora']['out_proj']['b']+=1e-4
    payload={'parameters':q};pin={'metadata':{'step':12586,'contract':{'recipes':[source_spec]}}}
    prepared=[prepare(p,s,payload,pin) for s in recipes(145)]
    losses=[]
    for s,(base,r,cursor) in zip(recipes(145),prepared):
        assert cursor==12586
        losses.append(jax.jit(lambda r:token_nll(p.cfg,base,r,t,p.columns,s,jnp.asarray(cursor),block_fn=block_forward))(r))
    np.testing.assert_allclose(losses[0],losses[1],rtol=1e-5,atol=1e-5)
    assert prepared[1][1][str(i)]['lora']['in_proj']['a'].shape[-1]==32
    for base,_,_ in prepared:
        assert base['embed_tokens']['embedding'] is p.base['embed_tokens']['embedding']
        assert base['layers_0']['mlp']['down_proj']['kernel'] is p.base['layers_0']['mlp']['down_proj']['kernel']
    for s,(base,r,_) in zip(recipes(145),prepared):
        counts=[int(v) for path,v in jax.tree.leaves_with_path(optimizer(r,s).init(r)) if getattr(path[-1],'name',None)=='count']
        assert counts and set(counts)=={0}


def test_source_restore_checks_actual_chunk_payload_adam_clock_and_best(tmp_path,monkeypatch):
    import extent.reset_long_recovery as module
    from extent.hf_artifact_sync import HubArtifactConfig
    from extent.recovery_durability import RecoveryCheckpointStore
    import hashlib
    p,t=patient();spec=old_recipes(136)[0];q=initialize(p,spec);tx=optimizer(q,spec);state=tx.init(q)
    for _ in range(3):
        g=jax.tree.map(jnp.ones_like,q);_,state=tx.update(g,state,q)
    payload=dict(parameters=q,optimizer=state,best=q,best_step=jnp.asarray(1,jnp.int32))
    code={}
    for name in ('extent/reset_streamed_gpu.py','extent/reset_followup.py','extent/competitor_recovery.py'):
        raw=(Path(__file__).resolve().parents[1]/name).read_bytes().replace(b'\r\n',b'\n');destination=tmp_path/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw);code[name]=hashlib.sha256(raw).hexdigest()
    directory=tmp_path/'cache'/'136-all';store=RecoveryCheckpointStore(directory,'prefix');meta=store.save('ARM',dict(recipes=[spec],code=code),payload,3)
    raw=(store.directory('ARM')/'checkpoint.json').read_bytes();pin=dict(repo_id='offline',revision='immutable',prefix='prefix',slot='ARM',path='prefix/ARM/checkpoint.json',manifest_sha256=hashlib.sha256(raw).hexdigest(),metadata=meta,best_step=1)
    (tmp_path/'results').mkdir();(tmp_path/'results/EXP-143-146-source-pin.json').write_text(json.dumps({'136-all':pin}));monkeypatch.setattr(module,'ROOT',tmp_path)
    restored,_=module.restore_source('136-all',HubArtifactConfig('offline','dataset',''),tmp_path/'cache')
    for a,b in zip(jax.tree.leaves(payload),jax.tree.leaves(restored)):np.testing.assert_array_equal(a,b)
    chunk=meta['leaves'][0]['chunks'][0]['file'];(store.directory('ARM')/chunk).write_bytes(b'broken')
    with pytest.raises(ValueError,match='SHA'):module.restore_source('136-all',HubArtifactConfig('offline','dataset',''),tmp_path/'cache')


def test_parent_notification_hard_cap_and_child_free_periodic_policy():
    from extent.reset_quiet import ParentNotifier
    now=[0.];sent=[]
    n=ParentNotifier(True,143,clock=lambda:now[0],sender=lambda e,t:sent.append(t))
    assert n.emit('started','start')
    for minute in range(1,721):
        now[0]=minute*60;n.emit('progress','tick')
        if minute in (300,301,302):n.emit('failure','bad')
    n.emit('finished','done');n.emit('finished','duplicate')
    assert len(sent)==5 and [x['kind'] for x in n.events]==['started','progress','failure','progress','finished']


def test_parent_spawns_quiet_workers_and_treats_arm_rejection_as_failure(tmp_path,monkeypatch,capsys):
    import scripts.m3q_reset_long_campaign as runner
    import extent.reset_quiet as quiet
    original=quiet.ParentNotifier;sent=[]
    monkeypatch.setattr(quiet,'ParentNotifier',lambda e,n:original(e,n,sender=lambda e,t:sent.append(t)))
    commands=[]
    class Child:
        returncode=0
        def __init__(self,cmd,**kw):
            commands.append(cmd);worker=int(cmd[cmd.index('--worker')+1]);assert cmd[-1]=='--no-telegram';assert kw['env']['CUDA_VISIBLE_DEVICES']==str(worker)
            (tmp_path/f'extent-m3q-long-143-worker-{worker}.json').write_text(json.dumps(dict(status='arm_failed' if worker==0 else 'completed',branches={})))
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    with pytest.raises(RuntimeError):runner.main(['--experiment','143','--output-dir',str(tmp_path)])
    assert len(commands)==2 and len(sent)==2
    report=json.loads((tmp_path/'extent-m3q-long-143.json').read_text());assert report['status']=='failed'
    capsys.readouterr()


@pytest.mark.parametrize('healthy,limit,probe_fails',[(True,130,False),(False,2,False),(True,2,True)])
def test_actual_long_runner_keeps_source_clocks_and_checks_before_acceptance(monkeypatch,tmp_path,healthy,limit,probe_fails):
    # Run the real orchestration with deterministic device kernels/transport doubles.
    import scripts.m3q_reset_long_campaign as runner
    import extent.config as config_module
    import extent.reset_followup as math_module
    import extent.reset_streamed_gpu as tpu_module
    import extent.competitor_tpu as eval_module
    import extent.reset_long_recovery as long_module
    import extent.reset_precision_probe as probe_module
    import extent.reset_long_patient as long_patient_module
    import extent.plateau_forensics as diagnostics_module
    import extent.t4_patient as patient_module
    import extent.t4_scale_data as data_module
    import extent.hf_artifact_sync as hub_module
    p,t=patient();p.teacher=p.base
    sourceq=initialize(p,arm_for(130,1));tx=optimizer(sourceq,arm_for(130,1));source_state=tx.init(sourceq)
    source_state=jax.tree.map(lambda x:x+jnp.asarray(7,x.dtype) if jnp.issubdtype(x.dtype,jnp.integer) else x,source_state)
    payload=dict(parameters=sourceq,optimizer=source_state,best=sourceq,best_step=jnp.asarray(3072,jnp.int32))
    reference={k:dict(nll=0.,windows=[0.]*(32 if k=='B' else 64)) for k in 'ABC'}
    runtime=dict(jax='x',jaxlib='x',flax='x',optax='x',matmul_precision=None,prng_impl='threefry2x32')
    pin=dict(reference=reference,source_runtime=dict(versions={k:'x' for k in ('jax','jaxlib','flax','optax')}))
    monkeypatch.setattr(config_module,'load_config',lambda *a:(p.cfg,None))
    monkeypatch.setattr(long_patient_module,'Patient',lambda *a,**kw:p)
    pin.update(runtime=runtime,metadata=dict(step=3434,contract=dict(recipes=[old_recipes(136)[0]])))
    monkeypatch.setattr(long_module,'restore_source',lambda *a:(payload,pin))
    def precision_probe(*a):
        if probe_fails:raise RuntimeError('optional probe unsupported')
        return {'scope':'mock'}
    monkeypatch.setattr(probe_module,'probe',precision_probe)
    monkeypatch.setattr(math_module,'Inspector',lambda *a:SimpleNamespace(profile=lambda *a:dict(layers=[])))
    monkeypatch.setattr(math_module,'ReferenceProbe',lambda *a:SimpleNamespace(compare=lambda *a:dict(passed=True)))
    monkeypatch.setattr(hub_module,'artifact_config_from_env',lambda:object())
    monkeypatch.setattr(data_module,'load_data',lambda *a:({k:np.zeros((n,256),np.int32) for k,n in [('A',64),('B',32),('C',64),('FIT',16384)]},{}))
    actual_devices=jax.devices
    monkeypatch.setattr(jax,'devices',lambda backend=None:actual_devices('cpu')[:1] if backend=='gpu' else actual_devices(backend))
    arities=[]
    def guarded(function):
        def call(*args):
            arities.append(len(args))
            if len(args)==4:assert 'embed_tokens' in args[0] # frozen base is explicit, never a giant closed-over constant
            return function(*args)
        return call
    monkeypatch.setattr(diagnostics_module,'GuardedJit',guarded)
    seen=[];source_counts=[]
    def make(*args,**kwargs):
        def step(q,state,*rest):
            source_counts.append([int(np.asarray(v)) for path,v in jax.tree.leaves_with_path(state) if getattr(path[-1],'name',None)=='count'])
            return jax.tree.map(lambda x:x+1e-6,q),state,dict(finite=healthy,loss=1.,ce=1.,kl=0.,stages={'gradients':{'finite':healthy}}),{}
        return step
    monkeypatch.setattr(tpu_module,'make_step',make)
    monkeypatch.setattr(eval_module,'token_nll',lambda cfg,base,q,t,*a,**k:jnp.zeros((t.shape[0],t.shape[1]-1)))
    class S:
        def __init__(self,args,contract,hub):
            self.args=args;self.wall=1e30;self.deadline=1e29;self.result=dict(runtime=runtime,status='preflight',branches={},pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0));self.store=SimpleNamespace(restore=lambda *a:None)
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=limit and self.deadline!=self.wall-180:raise TimeoutError('test work limit')
        def persist(self,**kw):
            seen.append(('persist',self.result['cost'].get('accepted_updates',0),any('first_update_health' in v for v in self.result['branches'].values())))
            return True
        def emit(self,*a):pass
        def checkpoint(self,name,payload,step):seen.append(('save',step))
    monkeypatch.setattr(runner,'session',S)
    args=runner.parser().parse_args(['--experiment','143','--worker','0','--output-dir',str(tmp_path)])
    args.max_wall_hours=11.5;args.max_new_hf_gb=4
    result=runner.run(args);row=result['branches']['ALL-CONTINUE']
    assert 4 in arities and source_counts
    assert row['step']==(3434+limit if healthy else 3434) and row['source_step']==3434
    assert source_counts and all(all(v==7 for v in counts) for counts in source_counts)
    assert ('persist',0,True) in seen # first proposal inspected before accepting any update
    assert ('save',3434+limit if healthy else 3434) in seen and row['finalization_status']=='completed'
    assert result['status']==('deadline_partial' if healthy else 'arm_failed')
    assert result['cost'].get('accepted_updates',0)==(limit if healthy else 0)
    if healthy and limit>=128:assert any(e['new_steps']==128 and e['step']==3562 for e in row['evaluations'])
    if probe_fails:assert result['precision_probe']['error_type']=='RuntimeError'
    if not healthy:assert row['failed']['failed_stages']==['gradients']



def test_seed456_patient_routes_all_preparation_and_main_sources_to456(tmp_path,monkeypatch):
    import extent.reset_long_patient as module
    from extent.hf_artifact_sync import HubArtifactConfig
    from extent.qwen3_teacher import Qwen3ForCausalLM
    from flax import traverse_util
    from scripts import m3q_subspace_engine as engine
    p,t=patient();teacher=Qwen3ForCausalLM(p.tc).init(jax.random.key(21),t)['params']
    flat=traverse_util.flatten_dict(teacher);arrays={'/'.join(k):np.asarray(v) for k,v in flat.items()}
    entries=[SimpleNamespace(source=k,target=k,transform='none') for k in arrays]
    pin=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-114-source-pin.json').read_text());calls=[]
    monkeypatch.setattr(module,'patient_config',lambda *a:p.cfg)
    monkeypatch.setattr(module,'teacher_config_from_spec',lambda *a,**kw:p.tc)
    monkeypatch.setattr(module,'ensure_qwen',lambda *a:SimpleNamespace(read=lambda key:arrays[key]))
    monkeypatch.setattr(module,'teacher_qwen_mappings',lambda *a:entries)
    monkeypatch.setattr(module,'expected_qwen_shape',lambda e,*a:arrays[e.source].shape)
    monkeypatch.setattr(module,'tree_digest',lambda *a:pin['source_models']['456']['sha256'])
    monkeypatch.setattr(module,'read_parameter_chunks',lambda *a:p.coordinates)
    monkeypatch.setattr(engine,'configured_contract',lambda *a:dict(replacement_order=p.cfg.mamba_layer_indices,source_contract={}))
    def endpoint(seq,seed,mode,layer,depth,sha):assert seed==456;return dict(seed=seed)
    monkeypatch.setattr(engine.sequential,'endpoint_contract',endpoint)
    class Store:
        def __init__(self,*a):pass
        def metadata(self,path,contract):assert 'seed-456/' in path and contract['seed']==456;return {'checkpoint_sha256':'prep'}
        def restore(self,path,contract):
            assert 'seed-456/' in path and contract['seed']==456
            i=int(path.rsplit('-',1)[1]);calls.append(i)
            return {'params':p.base[f'layers_{i}']['mamba']},{'checkpoint_sha256':json.loads((module.ROOT/'results/EXP-111-113-source-pin.json').read_text())['seeds']['456'][str(i)]}
    monkeypatch.setattr(module,'ReadOnlySourceStore',Store)
    def restore(path,remote,hub):
        assert '/456/MAIN/' in remote;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(pin['manifests']['456']['metadata']))
    monkeypatch.setattr(module,'restore_artifact',restore)
    result=module.Patient(tmp_path,HubArtifactConfig('offline','dataset',''),jax.devices('cpu')[0],seed=456)
    assert calls==list(p.cfg.mamba_layer_indices) and result.seed==456


def test_precision_probe_labels_its_scope_and_times_ready_results():
    from extent.reset_precision_probe import probe
    result=probe(jax.devices('cpu')[0],shapes=((8,16,32),),repeats=2)
    assert result['platform']=='cpu' and 'GEMM forward ONLY' in result['scope']
    assert set(result['rows'][0]['timings'])=={'bf16','fp16','fp32'}
    assert all(v['finite'] and v['median_seconds']>0 for v in result['rows'][0]['timings'].values())


def test_lightweight_source130_fold_is_array_identical_to_original():
    from extent.reset_long_recovery import fold_weights
    from extent.reset_followup import folded130
    p,t=patient();s=arm_for(130,1);q=initialize(p,s)
    expected=folded130(p,{'parameters':q})
    actual=fold_weights(p.base,q,p.cfg,p.columns,s)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('worker',[0,1])
def test_seed_replication_prepare_matches_original_recipe(worker):
    from extent.reset_followup import prepare as original
    p,t=patient();s=recipes(146)[worker]
    a,q,at=prepare(p,s);b,r,bt=original(p,s)
    for step in (0,2048):
        x=jax.jit(lambda q:token_nll(p.cfg,a,q,t,p.columns,s,jnp.asarray(step),block_fn=block_forward))(q)
        y=jax.jit(lambda r:token_nll(p.cfg,b,r,t,p.columns,s,jnp.asarray(step),block_fn=block_forward))(r)
        np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-5)


def test_direct_worker_also_forces_telegram_off(monkeypatch):
    import scripts.m3q_reset_long_campaign as runner
    monkeypatch.setattr(runner,'run',lambda args:dict(telegram=args.telegram))
    assert runner.main(['--experiment','143','--worker','0','--telegram'])==dict(telegram=False)


def test_lr_div3_changes_updates_without_resetting_adam_or_schedule():
    p,t=patient();a,b=recipes(144);q=initialize(p,a);ta,tb=optimizer(q,a),optimizer(q,b)
    state=ta.init(q);state=jax.tree.map(lambda v:v+jnp.asarray(300,v.dtype) if jnp.issubdtype(v.dtype,jnp.integer) else v,state)
    g=jax.tree.map(jnp.ones_like,q)
    ua,sa=ta.update(g,state,q);ub,sb=tb.update(g,state,q)
    for x,y in zip(jax.tree.leaves(ua),jax.tree.leaves(ub)):np.testing.assert_allclose(y,x/3,rtol=2e-5,atol=1e-10)
    for x,y in zip(jax.tree.leaves(sa),jax.tree.leaves(sb)):np.testing.assert_array_equal(x,y)
    counts=[int(v) for path,v in jax.tree.leaves_with_path(sa) if getattr(path[-1],'name',None)=='count']
    assert counts and set(counts)=={301}
