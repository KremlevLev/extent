import json,time
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding,Mesh,PartitionSpec
from tests.test_batch_recovery_161 import initialized_parent
from extent.method_sweep_163 import coordinates,optimizer,adam_counts,MethodReference,MethodStep
from extent import full_stream_166 as model
from extent import stream_batch_planner as planner
from extent.recovery_durability import RecoveryCheckpointStore


def test_actual_whole_batch_streamed_gradient_and_two_updates(tmp_path):
    p,t,current,_,teacher,_,_,_,_=initialized_parent(tmp_path)
    spec=dict(model.recipes(166)[0],align_layers=list(range(p.cfg.num_layers)))
    q=coordinates(p.base,current,p.cfg,p.columns,spec);tx=optimizer(q,spec)
    layout=NamedSharding(Mesh(np.array(jax.devices('cpu')),('data',)),PartitionSpec())
    q=jax.device_put(q,layout);state=jax.device_put(tx.init(q),layout);base=jax.device_put(p.base,layout);teacher=jax.device_put(teacher,layout)
    tokens=jax.device_put(jnp.concatenate([(t+i)%p.cfg.vocab_size for i in range(3)]),layout)
    before=jax.tree.map(np.array,(q,state))
    engine=MethodReference(p.cfg,p.tc,p.columns,spec,tx,return_gradients=True)
    out=engine(q,state,base,teacher,tokens,jnp.asarray(157685.,jnp.float32))
    reference=MethodStep(p.cfg,p.tc,p.columns,spec,tx).accumulated(q,state,base,teacher,tokens,jnp.asarray(157685.,jnp.float32))
    from extent.device_execution import compare_proposals
    assert compare_proposals(reference,out)['passed']
    assert bool(out[2]['finite']) and adam_counts(out[1])==[1]*6
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves((q,state))):np.testing.assert_array_equal(a,b)
    q1=jax.device_put(out[0],layout);s1=jax.device_put(out[1],layout);engine.return_gradients=False
    second=engine(q1,s1,base,teacher,tokens,jnp.asarray(157686.,jnp.float32))
    assert len(second)==3 and bool(second[2]['finite']) and adam_counts(second[1])==[2]*6


def test_planner_records_all_batches_full_errors_and_memory():
    ticks=[0.];messages=[]
    def invoke(engine,batch):
        if batch>=7:raise MemoryError('HBM guard: limit=1234 in_use=1200 requested=900')
        ticks[0]+={1:1,2:1.1,3:1.2,4:2,5:3,6:4}[batch]
    result=planner.benchmark(lambda b:b,invoke,lambda:None,clock=lambda:ticks[0],snapshot=lambda:[{'limit':1234}],record=messages.append)
    assert result['selected_batch']==3 and len(result['trials'])==8 and result['accepted_updates']==0
    assert result['trials'][-1]['error_message'].startswith('HBM guard') and result['trials'][-1]['memory_before']
    with pytest.raises(RuntimeError,match='No streamed'):
        planner.benchmark(lambda b:b,lambda *a:(_ for _ in ()).throw(MemoryError('actual detail')),lambda:None,snapshot=lambda:[])


def small_contract(tmp_path,monkeypatch):
    import extent.broad_corpus
    from scripts.prepare_edu_corpus import REVISION
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    c4=SimpleNamespace(identity='fixed-c4',manifest=expected)
    edu=SimpleNamespace(identity='fixed-edu',arrays={'train':np.zeros(20000*256,np.int32)},manifest=dict(contract=dict(
        dataset='HuggingFaceFW/fineweb-edu',revision=REVISION,excluded_c4_identity='fixed-c4',sequence_length=256,
        tokenizer_revision=expected['contract']['tokenizer_revision'])))
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda path:c4 if path=='c4' else edu)
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    source=dict(step=157685,windows=16993,adam_count=2833,runtime=runtime,manifest_file='full-stream/parent-manifest.json',manifest_sha='parent')
    parent=dict(format='methods-parent-v1',corpus_identity='fixed-c4',runtime=runtime,branches={'FULL-KD':dict(step=154852,parent_windows=0)},full_source=source)
    plan=tmp_path/'source.json';plan.write_text(json.dumps(parent))
    return plan,model.contract(166,None,1000000000,'c4','server-balanced',plan,'edu')


def test_preflight_zero_updates_never_copies_or_queues_a_full_checkpoint(tmp_path,monkeypatch):
    import extent.hf_artifact_sync
    from scripts import m3q_full_stream_166 as runner
    plan,c=small_contract(tmp_path,monkeypatch)
    assert c['experiment']==166 and len(json.dumps(c))<1000000
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',lambda *a:False)
    args=runner.parser().parse_args(['--source-plan',str(plan),'--corpus-dir','c4','--edu-dir','edu','--output-dir',str(tmp_path/'results'),
        '--checkpoint-dir',str(tmp_path/'states'),'--work-root',str(tmp_path),'--max-wall-hours','12','--no-telegram'])
    session=model.make_session(args,c,None);writes=[]
    session.store.save=lambda name,contract,payload,step:writes.append(step)
    payload={'best_step':jnp.asarray(157685)}
    session.checkpoint('FULL-KD',payload,157685);session.finalizing=True;session.checkpoint('FULL-KD',payload,157685)
    assert not writes and not session.result['pending_slots']
    session.checkpoint('FULL-KD',payload,157686);assert not writes
    session.result['branches']['FULL-KD']['invocation_updates']=1
    session.checkpoint('FULL-KD',payload,157686);assert writes==[157686]


def test_no_bootstrap_state_copy_and_external_manifest_fork(tmp_path,monkeypatch):
    q={'w':jnp.arange(4,dtype=jnp.float32)};state=tuple(jnp.asarray(2833,jnp.int32) for _ in range(6))
    payload=dict(parameters=q,optimizer=state,best=q,best_step=jnp.asarray(157685,jnp.int32),data_windows=jnp.asarray(16993,jnp.int32))
    parent=RecoveryCheckpointStore(tmp_path/'parent','old');meta=parent.save('FULL-KD',{'parent':165},payload,157685)
    pointer=parent.directory('FULL-KD')/'checkpoint.json';manifest=tmp_path/'parent-manifest.json';manifest.write_bytes(pointer.read_bytes())
    source=dict(manifest_sha=model.sha(manifest),source_prefix='old')
    child=model.ForkStore(tmp_path/'child','new',source=source,parent_root=parent.root,manifest_file=manifest)
    monkeypatch.setattr(model,'adam_counts',lambda value:[int(v) for v in value])
    saved,metadata=child.restore('FULL-KD',{'child':166},payload)
    assert metadata['step']==157685 and int(saved['data_windows'])==16993
    assert not child.directory('FULL-KD').exists()  # no22GB copy just to preflight
    child.save('FULL-KD',{'child':166},saved,157686)
    pointer.write_text('parent unavailable');own,_=child.restore('FULL-KD',{'child':166},payload)
    assert int(own['data_windows'])==16993


def test_compact_freezer_externalizes_recursive_metadata(tmp_path):
    root=tmp_path/'work';(root/'results').mkdir(parents=True);pointer=root/'states/runtime-full-next/FULL-KD/checkpoint.json';pointer.parent.mkdir(parents=True)
    base_branch=dict(step=154852,adam_count=36548,manifest_sha='base-sha',metadata={'contract':{'large':'x'*20000}})
    parent=dict(format='methods-parent-v1',branches={'FULL-KD':base_branch},runtime={},corpus_identity='c4')
    contract=dict(protocol='exp165-full-steady-v1',code={},parent=parent)
    meta=dict(step=157685,contract=dict(contract,arm='FULL-KD'));pointer.write_text(json.dumps(meta))
    report=dict(contract=contract,status='failed',runtime={},pending_slots=[],corpus_identity='c4',branches={'FULL-KD':dict(
        step=157685,durable_step=157685,accepted_new_updates=0,training_windows_seen=16993,source_controls={'D':{'nll':5.,'windows':[5.]}})})
    (root/'results/runtime-full-next.json').write_text(json.dumps(report))
    path,value=model.freeze_parent(root)
    assert path.stat().st_size<2000 and 'metadata' not in value['branches']['FULL-KD'] and 'metadata' not in value['full_source']
    manifest=root/value['full_source']['manifest_file'];assert manifest.read_bytes()==pointer.read_bytes()


def test_actual_planner_wires_streamed_audit_without_accepting_proposals(tmp_path,monkeypatch):
    p,t,current,_,teacher,_,_,_,_=initialized_parent(tmp_path)
    spec=dict(model.recipes(166)[0],align_layers=list(range(p.cfg.num_layers)))
    q=coordinates(p.base,current,p.cfg,p.columns,spec);tx=optimizer(q,spec);state=tx.init(q)
    tokens=jnp.concatenate([(t+i)%p.cfg.vocab_size for i in range(8)])
    before=jax.tree.map(np.array,(q,state));measured=[]
    def selected(factory,invoke,release,check,record):
        engine=factory(3);invoke(engine,3);measured.append(True)
        return dict(selected_batch=3,trials=[dict(batch=3,status='measured',completed_probe_proposals=1)])
    monkeypatch.setattr(planner,'benchmark',selected)
    engine,result=planner.plan_kernel(p.cfg,p.tc,p.columns,spec,tx,q,state,p.base,teacher,tokens,jnp.asarray(157685.),lambda:None,lambda value:None)
    assert measured and result['gradient_audit']['passed'] and result['accepted_updates']==0
    assert isinstance(engine,MethodReference) and not engine.return_gradients
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves((q,state))):np.testing.assert_array_equal(a,b)
    out=engine(q,state,p.base,teacher,tokens[:3],jnp.asarray(157685.))
    assert len(out)==3 and adam_counts(out[1])==[1]*6


def test_compact_base_reference_expands_only_for_restore_not_report(tmp_path,monkeypatch):
    pointer=tmp_path/'KD-C4-FRESH/checkpoint.json';pointer.parent.mkdir()
    meta=dict(contract={'parent':161},step=154852);pointer.write_text(json.dumps(meta))
    parent=dict(branches={'FULL-KD':dict(step=154852,adam_count=36548,manifest_sha=model.sha(pointer))})
    calls=[]
    def restore(*args,**kwargs):
        calls.append(kwargs['parent']);return 'base','coordinates',154852
    monkeypatch.setattr(model,'prepare163',restore)
    base,q,step=model.prepare(None,{'name':'FULL-KD'},None,None,parent=parent,parent_root=tmp_path)
    assert (base,q,step)==('base','coordinates',157685)
    assert calls[0]['branches']['FULL-KD']['metadata']==meta and 'metadata' not in parent['branches']['FULL-KD']


def test_cli_budget_accounts_other_processes_and_keeps_reserve(monkeypatch):
    import engine_full_stream as launcher
    monkeypatch.setattr(launcher.subprocess,'check_output',lambda *a,**kw:'97887, 15638\n')
    for name in ('XLA_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_ALLOCATOR','TF_GPU_ALLOCATOR','EXTENT_GPU_BUDGET_POLICY'):
        monkeypatch.setenv(name,'old')
    policy=launcher.configure_gpu_budget('0',5.)
    assert policy['pool_gib']==pytest.approx(73.3212890625)
    assert policy['pool_gib']+policy['other_used_mib']/1024+2+5==policy['total_mib']/1024
    import os
    assert float(os.environ['XLA_CLIENT_MEM_FRACTION'])==policy['fraction'] and 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ
    with pytest.raises(ValueError):launcher.configure_gpu_budget('0',1000.)
