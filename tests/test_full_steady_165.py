import json
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding,Mesh,PartitionSpec
from tests.test_batch_recovery_161 import initialized_parent
from extent.method_sweep_163 import coordinates,optimizer,adam_counts
from extent.steady_execution import proposal_step,gradient_step
from extent.steady_batch_planner import independent_best
from extent.method_execution_163 import compiled_step
from extent.reset_streamed_gpu import MemoryCheckedJit
from extent.recovery_durability import RecoveryCheckpointStore
from extent import full_steady_165 as model


def test_steady_best_and_two_real_updates_keep_one_executable_and_exact_math(tmp_path):
    p,t,current,_,teacher,_,_,_,_=initialized_parent(tmp_path)
    spec=dict(model.recipes(165)[0],align_layers=list(range(p.cfg.num_layers)))
    q=coordinates(p.base,current,p.cfg,p.columns,spec);tx=optimizer(q,spec)
    sharding=NamedSharding(Mesh(np.array(jax.devices('cpu')),('data',)),PartitionSpec())
    q=jax.device_put(q,sharding);state=jax.device_put(tx.init(q),sharding)
    base=jax.device_put(p.base,sharding);teacher=jax.device_put(teacher,sharding);tokens=jax.device_put(t,sharding)
    best=independent_best(q)
    for a,b in zip(jax.tree.leaves(q),jax.tree.leaves(best)):
        np.testing.assert_array_equal(a,b)
        assert isinstance(b,np.ndarray) and not np.shares_memory(np.asarray(a),b)
    arguments=(q,state,base,teacher,tokens,jnp.asarray(157685.,jnp.float32))
    old=jax.jit(compiled_step(p.cfg,p.tc,p.columns,spec,tx))(*arguments)
    engine=MemoryCheckedJit(proposal_step(p.cfg,p.tc,p.columns,spec,tx))
    actual=engine(*arguments)
    assert len(actual)==3 and len(old)==4
    for a,b in zip(jax.tree.leaves(old[:2]),jax.tree.leaves(actual[:2])):np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6)
    grad,health=jax.jit(gradient_step(p.cfg,p.tc,p.columns,spec,tx))(*arguments)
    from extent.reset_followup import compare_gradients
    assert compare_gradients(old[3],grad)['passed'] and bool(health['finite'])
    q1=jax.device_put(actual[0],sharding);state1=jax.device_put(actual[1],sharding)
    again=engine(q1,state1,base,teacher,tokens,jnp.asarray(157686.,jnp.float32))
    assert len(engine.cache)==1  # second accepted step MUST reuse the planned layout
    assert adam_counts(again[1])==[2]*6 and bool(again[2]['finite'])
    assert any(np.any(np.asarray(a)!=np.asarray(b)) for a,b in zip(jax.tree.leaves(q1),jax.tree.leaves(best)))
    old_bytes=sum(v.nbytes for v in jax.tree.leaves(old))
    new_bytes=sum(v.nbytes for v in jax.tree.leaves(actual))
    assert old_bytes-new_bytes==sum(v.nbytes for v in jax.tree.leaves(grad))


def test_full165_fork_preserves_164_accepted_step_adam_and_data(tmp_path):
    q={'weight':jnp.arange(4,dtype=jnp.float32)}
    # Scalar optimizer fixture exercises exact source clocks without initializing a model.
    state=tuple(jnp.asarray(2833,jnp.int32) for _ in range(6))
    payload=dict(parameters=q,optimizer=state,best=q,best_step=jnp.asarray(157684,jnp.int32),data_windows=jnp.asarray(16993,jnp.int32))
    store=RecoveryCheckpointStore(tmp_path/'parent','old')
    meta=store.save('FULL-KD',{'source':164},payload,157685)
    source=dict(metadata=meta,manifest_sha=model.sha(store.directory('FULL-KD')/'checkpoint.json'))
    # adam_counts expects nested named optax state; use an independent clock assertion here.
    original=model.adam_counts
    try:
        model.adam_counts=lambda value:[int(v) for v in value]
        child=model.ForkStore(tmp_path/'child','new',source=source,parent_root=store.root)
        got,metadata=child.restore('FULL-KD',{'child':165},payload)
        assert metadata['step']==157685 and int(got['data_windows'])==16993
        assert [int(v) for v in got['optimizer']]==[2833]*6
        for a,b in zip(jax.tree.leaves(got),jax.tree.leaves(payload)):np.testing.assert_array_equal(a,b)
        (store.directory('FULL-KD')/'checkpoint.json').write_text('parent unavailable after first fork')
        resumed,_=child.restore('FULL-KD',{'child':165},payload)
        assert int(resumed['data_windows'])==16993
    finally:model.adam_counts=original


def test_freeze_165_accepts_durable_164_memory_failure_not_an_unsaved_step(tmp_path):
    root=tmp_path/'work';(root/'results').mkdir(parents=True)
    pointer=root/'states/runtime-full/FULL-KD/checkpoint.json';pointer.parent.mkdir(parents=True)
    parent=dict(format='methods-parent-v1',branches={'FULL-KD':dict(step=154852)},runtime={},corpus_identity='c4')
    contract=dict(protocol='exp164-full-adaptive-v1',code={},parent=parent)
    meta=dict(step=157685,contract=dict(contract,arm='FULL-KD'))
    pointer.write_text(json.dumps(meta))
    report=dict(contract=contract,status='failed',error=dict(type='MemoryError'),runtime={},pending_slots=[],branches={'FULL-KD':dict(
        step=157685,durable_step=157685,training_windows_seen=16993,best_step=157684,best_nll=5.1765)})
    (root/'results/runtime-full.json').write_text(json.dumps(report))
    path,plan=model.freeze_parent(root)
    assert plan['full_source']['adam_count']==2833 and plan['full_source']['metadata']==meta
    assert path==root/'full-steady/source.json'


def test_contract165_and_session_use_saved164_clocks(tmp_path,monkeypatch):
    from importlib.metadata import version
    from types import SimpleNamespace
    from scripts import m3q_full_steady_165 as runner
    import extent.broad_corpus,extent.hf_artifact_sync
    from scripts.prepare_edu_corpus import REVISION
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
    c4=SimpleNamespace(identity='fixed-c4',manifest=expected)
    edu=SimpleNamespace(identity='fixed-edu',arrays={'train':np.zeros(20000*256,np.int32)},manifest=dict(contract=dict(
        dataset='HuggingFaceFW/fineweb-edu',revision=REVISION,excluded_c4_identity='fixed-c4',sequence_length=256,
        tokenizer_revision=expected['contract']['tokenizer_revision'])))
    monkeypatch.setattr(extent.broad_corpus,'PackedCorpus',lambda path:c4 if path=='c4' else edu)
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    source=dict(step=157685,windows=16993,adam_count=2833,runtime=runtime)
    parent=dict(format='methods-parent-v1',corpus_identity='fixed-c4',runtime=runtime,branches={'FULL-KD':dict(step=154852,parent_windows=0)},full_source=source)
    plan=tmp_path/'source.json';plan.write_text(json.dumps(parent))
    c=model.contract(165,None,1000000000,'c4','server-balanced',plan,'edu')
    assert c['experiment']==165 and c['batch'] is None and len(c['recipes'])==1
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',lambda *a:False)
    args=runner.parser().parse_args(['--source-plan',str(plan),'--corpus-dir','c4','--edu-dir','edu','--output-dir',str(tmp_path/'results'),
        '--checkpoint-dir',str(tmp_path/'states'),'--max-wall-hours','13','--no-telegram'])
    s=model.make_session(args,c,None)
    assert s.path.name=='runtime-full-next.json' and s.result['branches']['FULL-KD']['source_step']==157685
    assert s.result['branches']['FULL-KD']['training_windows_seen']==16993
