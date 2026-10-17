"""Bounded-memory streamed GPU math on tiny CPU virtual devices."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from extent import HybridForCausalLM,tiny_config
from extent.model import _attention_source_config
from extent.recovery_subspace import initialize_corrections,apply_corrections
from extent.reset_followup import recipes,prepare,block_forward
from extent.competitor_recovery import initialize,arm_for,optimizer
from extent.competitor_tpu import make_step as compiled_step
from extent.reset_streamed_gpu import make_step as streamed_step
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding
from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer

def patient():
    cfg=replace(tiny_config(),mla=replace(tiny_config().mla,implementation='qwen3_gqa'),param_dtype='float32',compute_dtype='float32',logits_dtype='float32',tie_word_embeddings=True,remat_policy='none')
    t=jnp.arange(8,dtype=jnp.int32)[None]%cfg.vocab_size
    base=HybridForCausalLM(cfg).init(jax.random.key(4),t)['params']
    cols=2*int(cfg.hidden_size*cfg.mamba.expand)+2*cfg.mamba.mimo_rank*cfg.mamba.groups*cfg.mamba.d_state
    old=initialize_corrections(base,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=8,head_dim=cfg.mamba.head_dim)
    p=SimpleNamespace(cfg=cfg,base=base,current=apply_corrections(base,old,protected_input_columns=cols),coordinates=old,columns=cols,tc=_attention_source_config(cfg))
    return p,t


@pytest.mark.parametrize('experiment,worker',[(135,0),(136,0),(136,1),(137,0),(137,1),(138,0),(138,1)])
def test_streamed_full_gradient_matches_compiled_objective(experiment,worker):
    p,t=patient();spec=recipes(experiment)[worker]
    source={'parameters':initialize(p,arm_for(130,1))}
    base,q,cursor=prepare(p,spec,source)
    mesh=create_v5e_mesh(jax.devices('cpu')[:2 if experiment==135 else 1])
    base=jax.device_put(base,named_sharding_tree(base,mesh));q=jax.device_put(q,coordinate_layout(q,mesh));tx=optimizer(q,spec)
    state,_=sharded_optimizer(tx,q,coordinate_layout(q,mesh),mesh)
    tp=jax.device_put(p.base,named_sharding_tree(p.base,mesh)) if spec['alpha'] else {}
    # Tiny teacher must have GQA in every layer, unlike the hybrid base.
    if spec['alpha']:
        from extent.qwen3_teacher import Qwen3ForCausalLM
        tp=Qwen3ForCausalLM(p.tc).init(jax.random.key(99),t)['params']
        tp=jax.device_put(tp,named_sharding_tree(tp,mesh))
    args=(q,state,base,tp,jax.device_put(t,batch_sharding(mesh)),jnp.asarray(2048.))
    expected=jax.jit(compiled_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=block_forward,return_gradients=True))(*args)
    engine=streamed_step(p.cfg,p.tc,p.columns,spec,tx,return_gradients=True)
    actual=engine(*args)
    assert bool(actual[2]['finite'])
    np.testing.assert_allclose(float(actual[2]['loss']),float(expected[2]['loss']),rtol=2e-5,atol=2e-5)
    for a,b in zip(jax.tree.leaves(actual[3]),jax.tree.leaves(expected[3])):np.testing.assert_allclose(a,b,rtol=2e-3,atol=2e-5)
    for a,b in zip(jax.tree.leaves(actual[0]),jax.tree.leaves(expected[0])):np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-6)
    if experiment==135:
        assert engine.forward[False].cache
        assert any(len(e.runtime_executable().local_devices())==2 for e in engine.backward[False].cache.values())


def test_streamed_plan_has_separate_namespace_and_accepts_all_gpu_cells(capsys):
    from scripts.m3q_reset_t4_streamed_campaign import main
    for e in range(135,139):
        c=main(['--experiment',str(e),'--max-wall-hours','11.5','--plan-only'])
        assert c['protocol']=='exp135-138-streamed-v2' and c['horizon']==131072
    capsys.readouterr()


def test_parent_prints_original_worker_traceback(tmp_path,monkeypatch,capsys):
    import scripts.m3q_reset_t4_streamed_campaign as runner
    import scripts.m3q_t4_campaign as transport
    monkeypatch.setattr(transport,'notify',lambda *a:None)
    class Child:
        returncode=1
        def __init__(self,*a,**kw):kw['stdout'].write('Traceback: actual child failure\nMemoryError: example HBM detail\n')
        def wait(self):return 1
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    with pytest.raises(RuntimeError):runner.main(['--experiment','135','--output-dir',str(tmp_path),'--no-telegram'])
    assert 'actual child failure' in capsys.readouterr().out
    import json
    report=json.loads((tmp_path/'extent-m3q-reset-streamed-v2-135.json').read_text())
    assert report['workers'][0]['exit']==1 and 'HBM detail' in report['workers'][0]['log_tail']


def test_per_layer_memory_guard_still_rejects_overflow_and_explains_numbers(monkeypatch):
    import extent.reset_streamed_gpu as m
    dev=SimpleNamespace(id=0,platform='gpu',memory_stats=lambda:dict(bytes_limit=100,bytes_in_use=90))
    stats=SimpleNamespace(argument_size_in_bytes=1,output_size_in_bytes=1,temp_size_in_bytes=20,alias_size_in_bytes=0)
    exe=SimpleNamespace(memory_analysis=lambda:stats,runtime_executable=lambda:SimpleNamespace(local_devices=lambda:[dev]))
    monkeypatch.setattr(m.jax,'jit',lambda f:SimpleNamespace(lower=lambda *a:SimpleNamespace(compile=lambda:exe)))
    with pytest.raises(MemoryError,match='"temp_bytes": 20'):m.MemoryCheckedJit(lambda:None)()


@pytest.mark.parametrize('healthy',[True,False])
def test_streamed_runner_preserves_source_adam_and_rejects_before_acceptance(monkeypatch,tmp_path,healthy):
    # Run the real orchestration with deterministic device kernels/transport doubles.
    import scripts.m3q_reset_t4_streamed_campaign as runner
    import extent.config as config_module
    import extent.reset_followup as math_module
    import extent.reset_streamed_gpu as tpu_module
    import extent.competitor_tpu as evaluation_module
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
    monkeypatch.setattr(patient_module,'Patient',lambda *a:p)
    monkeypatch.setattr(math_module,'source130',lambda *a:(payload,pin))
    monkeypatch.setattr(math_module,'Inspector',lambda *a:SimpleNamespace(profile=lambda *a:dict(layers=[])))
    monkeypatch.setattr(math_module,'ReferenceProbe',lambda *a:SimpleNamespace(compare=lambda *a:dict(passed=True)))
    monkeypatch.setattr(hub_module,'artifact_config_from_env',lambda:object())
    monkeypatch.setattr(data_module,'load_data',lambda *a:({k:np.zeros((n,256),np.int32) for k,n in [('A',64),('B',32),('C',64),('FIT',16384)]},{}))
    actual_devices=jax.devices
    monkeypatch.setattr(jax,'devices',lambda backend=None:actual_devices('cpu')[:2] if backend=='gpu' else actual_devices(backend))
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
    monkeypatch.setattr(evaluation_module,'token_nll',lambda cfg,base,q,t,*a,**k:jnp.zeros((t.shape[0],t.shape[1]-1)))
    class S:
        def __init__(self,args,contract,hub):
            self.args=args;self.wall=1e30;self.deadline=1e29;self.result=dict(runtime=runtime,status='preflight',branches={},pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0));self.store=SimpleNamespace(restore=lambda *a:None)
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=2 and self.deadline!=self.wall-180:raise TimeoutError('test work limit')
        def persist(self,**kw):
            seen.append(('persist',self.result['cost'].get('accepted_updates',0),any('first_update_health' in v for v in self.result['branches'].values())))
            return True
        def emit(self,*a):pass
        def checkpoint(self,name,payload,step):seen.append(('save',step))
    monkeypatch.setattr(runner,'session',S)
    args=runner.parser().parse_args(['--experiment','135','--worker','0','--output-dir',str(tmp_path)])
    args.max_wall_hours=11.5;args.max_new_hf_gb=4
    result=runner.run(args);row=result['branches']['RESET130-CONTINUE']
    assert 4 in arities and source_counts # streamed step is a Python chain, not one outer JIT
    assert row['step']==(3436 if healthy else 3434) and row['source_step']==3434
    assert source_counts and all(all(v==7 for v in counts) for counts in source_counts)
    assert ('persist',0,True) in seen # first proposal inspected before accepting any update
    assert ('save',3436 if healthy else 3434) in seen and row['finalization_status']=='completed'
    assert result['status']==('deadline_partial' if healthy else 'arm_failed')
    assert result['cost'].get('accepted_updates',0)==(2 if healthy else 0)
    if not healthy:assert row['failed']['failed_stages']==['gradients']



def test_bf16_streamed135_passes_actual_legacy_gradient_probe():
    from scripts.m3q_t4_preflight import tiny_patient
    from extent.reset_followup import ReferenceProbe
    p,_,params,t=tiny_patient(jax.devices('cpu')[0]);p.base=params;p.current=params
    p.columns=2*int(p.cfg.hidden_size*p.cfg.mamba.expand)+2*p.cfg.mamba.mimo_rank*p.cfg.mamba.groups*p.cfg.mamba.d_state
    p.coordinates=initialize_corrections(p.base,p.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=8,head_dim=p.cfg.mamba.head_dim)
    source={'parameters':initialize(p,arm_for(130,1))};spec=recipes(135)[0];base,q,_=prepare(p,spec,source);tx=optimizer(q,spec);t=np.asarray(t[:,:8])
    out=streamed_step(p.cfg,p.tc,p.columns,spec,tx,return_gradients=True)(q,tx.init(q),base,{},jnp.asarray(t),jnp.asarray(3434.))
    check=ReferenceProbe(p,jax.devices('cpu')[0]).compare(q,t,3434,float(out[2]['loss']),out[3])
    assert check['passed'],check
