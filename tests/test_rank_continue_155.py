"""Resume clocks, trained extra coordinates and actual runner gates on a tiny BF16 hybrid."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from importlib.metadata import version
import hashlib
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.test_rank_growth_153 import mature_source
from extent import rank_growth_153 as growth
from extent import rank_continue_155 as continuation
from extent.competitor_recovery import optimizer
from extent.competitor_tpu import token_nll
from extent.recovery_durability import RecoveryCheckpointStore


def trained_growth():
    p,t,q=mature_source();spec=continuation.recipes(155)[0]
    q=growth.grow_coordinates(p.base,q,p.cfg,256)
    tx=optimizer(q,spec)
    kernel=growth.make_kernel(p.cfg,p.tc,p.columns,spec,tx)
    q,state,_=kernel(q,tx.init(q),p.base,{},t,jnp.asarray(37908.,jnp.float32))
    best=q
    q,state,_=kernel(q,state,p.base,{},t,jnp.asarray(37909.,jnp.float32))
    return p,t,spec,dict(parameters=q,optimizer=state,best=best,best_step=jnp.asarray(37909,jnp.int32))


def test_restore_real_optimizer_tree_and_selected_best(tmp_path,monkeypatch):
    p,t,spec,payload=trained_growth()
    store=RecoveryCheckpointStore(tmp_path/'immutable'/'153v2-r256-current','source')
    contract={'model':asdict(p.cfg),'recipes':[spec],'code':{}}
    store.save('GROW-R256',contract,payload,37910)
    pointer=store.directory('GROW-R256')/'checkpoint.json';meta=json.loads(pointer.read_text())
    pin=dict(repo_id='offline',revision='frozen',prefix='source',slot='GROW-R256',path='source/GROW-R256/checkpoint.json',
             manifest_sha256=hashlib.sha256(pointer.read_bytes()).hexdigest(),metadata=meta,source_step=37910,adam_step=2,best_step=37909)
    monkeypatch.setattr(continuation,'source_pin',lambda:pin)
    from extent.hf_artifact_sync import HubArtifactConfig
    actual,_=continuation.restore_source('153v2-r256-current',HubArtifactConfig('offline','dataset',''),tmp_path/'immutable')
    assert continuation.adam_counts(actual['optimizer'])==[2]*6
    for a,b in zip(jax.tree.leaves(payload),jax.tree.leaves(actual)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='only pinned'):
        continuation.restore_source('147-original',None,tmp_path)


def test_growth_tpu_engine_audits_actual_extra_gradients_and_preserves_adam():
    p,t,spec,payload=trained_growth();tx=optimizer(payload['parameters'],spec)
    engine=continuation.make_kernel(p.cfg,p.tc,p.columns,spec,tx,tpu=True)
    actual=engine(payload['parameters'],payload['optimizer'],p.base,{},t,jnp.asarray(37910.,jnp.float32))
    assert bool(actual[2]['finite'])
    assert continuation.adam_counts(actual[1])==[3]*6
    assert engine.audit['selected'] in ('compiled','streamed')
    assert engine.audit['gradient_evaluations']==2
    assert any(np.any(np.asarray(actual[3][str(i)]['extra_lora']['out_proj']['b'])) for i in p.cfg.mamba_layer_indices)
    before=jax.tree.leaves(payload['parameters']);after=jax.tree.leaves(actual[0])
    assert any(not np.array_equal(a,b) for a,b in zip(before,after))
    # Force a discrepant compiled proposal; returned update MUST be reference.
    engine=continuation.make_kernel(p.cfg,p.tc,p.columns,spec,tx,tpu=True)
    reference=engine.streamed(payload['parameters'],payload['optimizer'],p.base,{},t,jnp.asarray(37910.,jnp.float32))
    bad=reference[:3]+(jax.tree.map(lambda x:x*2,reference[3]),)
    engine.compiled=lambda *a:bad
    result=engine(payload['parameters'],payload['optimizer'],p.base,{},t,jnp.asarray(37910.,jnp.float32))
    assert result[2]['engine_gate']['selected']=='streamed'
    for a,b in zip(jax.tree.leaves(reference[:2]),jax.tree.leaves(result[:2])):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('bad_replay',[False,True])
def test_runner_continues_current_with_inherited_best_and_persists_rejection(tmp_path,monkeypatch,bad_replay):
    from scripts import m3q_rank_continue_155_campaign as runner
    import extent.config,extent.competitor_tpu,extent.reset_long_patient,extent.t4_scale_data,extent.hf_artifact_sync
    p,t,spec,payload=trained_growth();cursor=37910
    c=runner.contract_for(155)
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    ref=float(token_nll(p.cfg,p.base,payload['best'],t,p.columns,spec,jnp.asarray(37909.),block_fn=growth.block_forward).mean())
    pin=dict(source_step=cursor,adam_step=2,best_step=37909,runtime=runtime,data={},
             reference={k:dict(windows=[ref+(1 if bad_replay else 0)]) for k in 'ABC'})
    monkeypatch.setattr(continuation,'source_pin',lambda:pin)
    monkeypatch.setattr(continuation,'restore_source',lambda *a:(payload,pin))
    monkeypatch.setattr(continuation,'base_coordinates',lambda *a:{})
    monkeypatch.setattr(continuation,'prepare',lambda *a:(p.base,payload['parameters'],cursor))
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}))
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    monkeypatch.setattr(extent.competitor_tpu,'training_batch',lambda *a:np.asarray(t))
    from extent.hf_artifact_sync import HubArtifactConfig
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:HubArtifactConfig('offline','dataset',''))
    devices=jax.devices
    monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu') if platform=='tpu' else devices(platform))
    # Actual one-device mesh for CPU tests; don't simulate eight physical TPUs.
    monkeypatch.setattr(runner,'contract_for',lambda *a:c)
    original_run=runner.run
    # The production guard stays8; test-only proxy reports length8 but iterates one CPU.
    class Devices(list):
        def __len__(self):return 8
    monkeypatch.setattr(jax,'devices',lambda platform=None:Devices(devices('cpu')) if platform=='tpu' else devices(platform))
    class Session:
        def __init__(self):
            self.result=dict(branches={},runtime=runtime,cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0),pending_slots=[])
            self.store=RecoveryCheckpointStore(tmp_path/'state','155')
            self.wall=time.monotonic()+3600;self.deadline=self.wall-1800
        def check(self):
            if self.result['cost'].get('accepted_updates',0)>=1:raise TimeoutError('test work deadline')
        def emit(self,*a):pass
        def persist(self,**kw):
            (tmp_path/'report.json').write_text(json.dumps(self.result));return True
        def checkpoint(self,n,v,s):self.store.save(n,dict(c,arm=n),v,s)
    session=Session();monkeypatch.setattr(runner,'session',lambda *a:session)
    args=SimpleNamespace(experiment=155,worker=None,max_steps=262144,cache_dir=str(tmp_path/'cache'))
    if bad_replay:
        with pytest.raises(RuntimeError):original_run(args)
        row=next(iter(session.result['branches'].values()))
        assert not row['source_replay']['passed']
        assert row['source_replay']['reference_step']==37909
        assert 'source_best_controls' in row
        assert session.result['cost'].get('accepted_updates',0)==0
    else:
        r=original_run(args);row=next(iter(r['branches'].values()))
        assert r['status']=='deadline_partial' and r['cost']['accepted_updates']==1
        assert row['step']==37911 and row['adam_count_at_restore']==[2]*6
        meta=session.store.metadata('CONTINUE-R256-ADAM',dict(c,arm='CONTINUE-R256-ADAM'))
        assert meta['step']==37911


def test_contract_does_not_change_old_sources_or_dispatch_gpu(capsys):
    from scripts import m3q_rank_continue_155_campaign as runner
    from scripts.m3q_rank_growth_153_campaign import contract_for as old
    old_plan=json.loads((continuation.ROOT/'results/EXP-153-rank-growth-v2-engineering-plan.json').read_text())
    # Production source hashes are checked directly, including old runner bytes.
    for file,sha in continuation.source_pin()['metadata']['contract']['code'].items():
        assert hashlib.sha256((continuation.ROOT/file).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==sha
    c=runner.main(['--plan-only']);assert len(c['recipes'])==1
    assert c['recipes'][0]['total_rank']==256 and c['recipes'][0]['mode']=='exact'
    assert c['experiment']==155 and '6784' in c['clock']
    assert runner.session_budget(155,8.7,45)==(8.7*3600-2700,45)
    with pytest.raises(SystemExit):runner.main(['--worker','1','--plan-only'])
    capsys.readouterr()
