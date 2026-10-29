import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import optax
from extent import full_lr_167 as model
from extent import lr_batch_planner_167 as planner
from extent.method_sweep_163 import optimizer,adam_counts
from extent.recovery_durability import RecoveryCheckpointStore
from tests.test_full_stream_166 import small_contract


def test_contract_scales_only_rates_and_carries_same_clocks(tmp_path,monkeypatch):
    plan,_=small_contract(tmp_path,monkeypatch)
    source=json.loads(plan.read_text());source['lr_multiplier']=1
    source['full_source'].update(step=181145,windows=204673,adam_count=26293)
    results=[]
    for multiplier in (1,3,10):
        source['lr_multiplier']=multiplier;plan.write_text(json.dumps(source))
        results.append(model.contract(167,None,4096,'c4','server-balanced',plan,'edu'))
    for multiplier,c in zip((1,3,10),results):
        assert c['batch']==8 and c['horizon']==4096 and c['full_source']['adam_count']==26293
        assert c['full_source']['windows']==204673 and c['full_source']['step']==181145
        assert c['recipes'][0]['mamba_lr']==pytest.approx(multiplier*1e-5)
        assert c['recipes'][0]['mlp_lr']==pytest.approx(multiplier*1e-5)
        assert c['recipes'][0]['norm_lr']==pytest.approx(multiplier*1e-6)
        assert c['execution_namespace'].endswith('/x'+str(multiplier))
    specs=[{k:v for k,v in c['recipes'][0].items() if k not in ('mamba_lr','mlp_lr','norm_lr')} for c in results]
    assert specs[0]==specs[1]==specs[2]


def test_real_optimizer_preserves_carried_adam_and_scales_update():
    q={'0':{'dense_mlp':{'w':jnp.ones(2)},'input_layernorm':jnp.ones(2),'mixer':jnp.ones(2)}}
    base=model.recipes(167)[0];initial=optimizer(q,base).init(q)
    carried=jax.tree.map(lambda x:jnp.full_like(x,26293) if jnp.issubdtype(x.dtype,jnp.integer) else x,initial)
    frozen=jax.tree.map(np.array,carried);proposals=[]
    for multiplier in (1,3,10):
        spec=dict(base)
        for key in ('mamba_lr','mlp_lr','norm_lr'):spec[key]*=multiplier
        tx=optimizer(q,spec);delta,after=tx.update(jax.tree.map(jnp.ones_like,q),carried,q)
        assert adam_counts(after)==[26294]*6
        proposals.append(delta)
        _,second=tx.update(jax.tree.map(jnp.ones_like,q),after,optax.apply_updates(q,delta))
        assert adam_counts(second)==[26295]*6
    for multiplier,delta in zip((1,3,10),proposals):
        for baseline,actual in zip(jax.tree.leaves(proposals[0]),jax.tree.leaves(delta)):
            np.testing.assert_allclose(actual,baseline*multiplier,rtol=1e-5)
    for before,after in zip(jax.tree.leaves(frozen),jax.tree.leaves(carried)):np.testing.assert_array_equal(before,after)


def test_planner_requires_common_batch_even_when_smaller_is_faster():
    ticks=[0.]
    def invoke(engine,batch):ticks[0]+=1 if batch==4 else batch
    result=planner.benchmark(lambda b:b,invoke,lambda:None,clock=lambda:ticks[0],snapshot=lambda:[])
    assert len(result['trials'])==8 and result['selected_batch']==8
    def oom(engine,batch):
        if batch==8:raise MemoryError('batch8 cannot fit')
        ticks[0]+=1
    with pytest.raises(RuntimeError,match='requires physical batch8'):
        planner.benchmark(lambda b:b,oom,lambda:None,clock=lambda:ticks[0],snapshot=lambda:[])


def test_source_fork_carries_current_and_resets_only_best(tmp_path):
    q={'w':jnp.arange(4,dtype=jnp.float32)}
    initial=optimizer(q,model.recipes(167)[0]).init(q)
    state=jax.tree.map(lambda x:jnp.full_like(x,26293) if jnp.issubdtype(x.dtype,jnp.integer) else x,initial)
    payload=dict(parameters=q,optimizer=state,best={'w':q['w']+100},best_step=jnp.asarray(178165),data_windows=jnp.asarray(204673))
    parent=tmp_path/'states/runtime-full-stream';store=RecoveryCheckpointStore(parent,'read-only',None)
    store.save('FULL-KD',{'old':'contract'},payload,181145)
    pointer=parent/'FULL-KD/checkpoint.json';manifest=tmp_path/'parent.json';manifest.write_bytes(pointer.read_bytes())
    fork=model.ForkStore(tmp_path/'own','new',None,source=dict(manifest_sha=model.sha(manifest),source_prefix='old'),parent_root=parent,manifest_file=manifest)
    result,meta=fork.restore('FULL-KD',{'new':'contract'},payload)
    assert meta['step']==181145 and model.adam_counts(result['optimizer'])==[26293]*6
    assert int(result['data_windows'])==204673 and int(result['best_step'])==181145
    np.testing.assert_array_equal(result['best']['w'],q['w'])
    assert not (tmp_path/'own/FULL-KD/checkpoint.json').exists()


def test_summary_refuses_unequal_tails(tmp_path):
    from engine_lr import summarize
    directory=tmp_path/'results';directory.mkdir()
    for m in (1,3,10):
        row=dict(accepted_new_updates=4096,accepted_input_tokens=8388608,
            final=dict(last=dict(panels={'E':dict(nll=5-m*.001)})))
        report=dict(status='completed',finalization_status='completed',pending_slots=[],branches={'FULL-KD':row})
        (directory/f'runtime-lr-x{m}.json').write_text(json.dumps(report))
    assert summarize(tmp_path,4096)['ranking_by_edu_E']==['LR-x10','LR-x3','LR-x1']
    report['branches']['FULL-KD']['accepted_input_tokens']-=2048
    (directory/'runtime-lr-x10.json').write_text(json.dumps(report))
    assert not summarize(tmp_path,4096)['ranking_by_edu_E']


def test_wrapper_lr_routing_keeps_full166(monkeypatch):
    import sys,engine_launch
    calls=[]
    monkeypatch.setitem(sys.modules,'setproctitle',SimpleNamespace(setproctitle=lambda t:None))
    monkeypatch.setattr(engine_launch,'nvml_process_name',lambda:None)
    monkeypatch.setitem(sys.modules,'engine_lr',SimpleNamespace(main=lambda args:calls.append(args)))
    engine_launch.main(['--runner','lr','--max-wall-hours','9'])
    assert calls==[['--max-wall-hours','9']]


def test_freezer_pins_all_arms_to_same_final_current(tmp_path,monkeypatch):
    directory=tmp_path/'results';directory.mkdir()
    contract=dict(protocol='exp166-full-stream-v1',code={},parent=dict(branches={'FULL-KD':dict(
        step=154852,adam_count=36548,manifest_sha='base',parent_windows=0)}))
    pointer=tmp_path/'states/runtime-full-stream/FULL-KD/checkpoint.json';pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps(dict(step=181145,contract=dict(contract,arm='FULL-KD'))))
    report=dict(contract=contract,runtime={},corpus_identity='c4',pending_slots=[],finalization_status='completed',
        branches={'FULL-KD':dict(step=181145,durable_step=181145,training_windows_seen=204673,
            final=dict(last=dict(panels={'D':{'nll':5.13,'windows':[5.13]}},fresh_nll={})))})
    (directory/'runtime-full-stream.json').write_text(json.dumps(report))
    plans=[model.freeze_parent(tmp_path,multiplier=m)[1] for m in (1,3)]
    # A missing local pointer must be restored from the durable donor on HF.
    import extent.hf_artifact_sync
    contents=pointer.read_bytes();pointer.unlink();downloads=[]
    def restore(target,remote,hub):
        downloads.append(remote);target.write_bytes(contents);return True
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',restore)
    monkeypatch.setattr(extent.hf_artifact_sync,'upload_artifacts_together',lambda *a,**kw:True)
    plans.append(model.freeze_parent(tmp_path,hub=object(),multiplier=10)[1])
    assert downloads==['experiments/exp166-full-stream-v1/checkpoints/FULL-KD/checkpoint.json']
    assert plans[0]['full_source']==plans[1]['full_source']==plans[2]['full_source']
    assert plans[0]['full_source']['manifest_sha']==model.sha(pointer)
    assert all(len(json.dumps(p))<2000 for p in plans)
    pointer.write_text('changed donor')
    fork=model.ForkStore(tmp_path/'own','new',None,source=plans[0]['full_source'],
        parent_root=pointer.parent.parent,manifest_file=tmp_path/'lr-sweep/parent-manifest.json')
    with pytest.raises(ValueError,match='source pointer changed'):fork.restore('FULL-KD',{}, {})


def test_orchestrator_runs_serial_arms_with_one_common_budget(tmp_path,monkeypatch):
    import engine_lr,extent.server_control
    calls=[]
    monkeypatch.setattr(extent.server_control,'storage_status',lambda *a:dict(free_bytes=200e9,work_bytes=0))
    class Child:
        def __init__(self,command,**kwargs):
            self.command=command;calls.append(command)
        def wait(self):
            m=int(self.command[self.command.index('--lr-multiplier')+1]);directory=tmp_path/'results';directory.mkdir(exist_ok=True)
            row=dict(accepted_new_updates=4096,accepted_input_tokens=8388608,
                final=dict(last=dict(panels={'E':dict(nll=5-m*.001)})))
            (directory/f'runtime-lr-x{m}.json').write_text(json.dumps(dict(status='completed',
                finalization_status='completed',pending_slots=[],branches={'FULL-KD':row})))
            return 0
    monkeypatch.setattr(engine_lr.subprocess,'Popen',Child)
    result=engine_lr.main(['--work-dir',str(tmp_path),'--max-wall-hours','9','--no-telegram'])
    assert result['matched_complete'] and len(calls)==3
    assert [int(c[c.index('--lr-multiplier')+1]) for c in calls]==[1,3,10]
    assert all(c[c.index('--pilot-updates')+1]=='4096' and '--clear-stop' not in c for c in calls)


def test_deleted_base_manifest_and_chunks_restore_and_verify(tmp_path,monkeypatch):
    import extent.hf_artifact_sync
    remote=RecoveryCheckpointStore(tmp_path/'remote','unused',None)
    remote.save('KD-C4-FRESH',{'source':161},{'parameters':{'w':jnp.arange(4.)}},154852)
    directory=remote.directory('KD-C4-FRESH');files={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
    # Chunk filenames contain generation subdirectories.
    files={str(p.relative_to(directory)).replace('\\','/'):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    calls=[]
    def restore(target,path,hub):
        key=path.split('/KD-C4-FRESH/',1)[1];calls.append(key);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(files[key]);return True
    monkeypatch.setattr(extent.hf_artifact_sync,'restore_artifact',restore)
    pointer=tmp_path/'local/KD-C4-FRESH/checkpoint.json';source={'manifest_sha':model.sha(directory/'checkpoint.json')}
    model.restore_base_dependency(pointer,source,object())
    assert calls[0]=='checkpoint.json' and len(calls)>1
    before=len(calls);model.restore_base_dependency(pointer,source,object());assert len(calls)==before
    pointer.write_text('{}')
    with pytest.raises(ValueError,match='manifest SHA mismatch'):model.restore_base_dependency(pointer,source,object())
