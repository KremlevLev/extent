"""Actual fold/source/evaluation/update/save tests; no fake model losses."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.test_fold_continue import setup_sources
from extent import continuation_matrix as m
from extent.competitor_recovery import optimizer
from extent.reset_followup import block_forward
from extent.competitor_tpu import token_nll
from extent.recovery_durability import RecoveryCheckpointStore
from extent.hf_artifact_sync import HubArtifactConfig
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding
from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer


@pytest.mark.parametrize('experiment,worker',[(149,0),(150,0),(151,0),(152,0),(153,0),(153,1)])
def test_real_intervention_forward_update_and_save(experiment,worker,monkeypatch,tmp_path):
    p,t,ancestor,base,payload,pin136=setup_sources()
    import extent.fold_continue_recovery as old
    monkeypatch.setattr(old,'source_pin',lambda key:pin136)
    spec=m.recipes(experiment)[worker]
    source_spec=m.recipes(149)[0]
    pin=dict(source_step=31126,adam_step=3,metadata=dict(contract=dict(recipes=[source_spec])))
    actual,q,cursor=m.prepare(p,spec,payload,pin,ancestor)
    assert cursor==31126
    expected=token_nll(p.cfg,base,payload['parameters'],t,p.columns,source_spec,jnp.asarray(cursor),block_fn=block_forward)
    found=token_nll(p.cfg,actual,q,t,p.columns,spec,jnp.asarray(cursor),block_fn=block_forward)
    np.testing.assert_allclose(found,expected,rtol=3e-5,atol=3e-5)
    for i in p.cfg.mamba_layer_indices:
        np.testing.assert_array_equal(q[str(i)]['pregate_weight'],payload['parameters'][str(i)]['pregate_weight'])
        if spec['mode']=='fold-new':
            assert q[str(i)]['lora']['in_proj']['a'].shape[1]==spec['correction_rank']
            assert not np.any(q[str(i)]['lora']['in_proj']['b'])
    # TPU path uses eight virtual devices, GPU path one; real same model gradient.
    mesh=create_v5e_mesh(jax.devices('cpu')[:8 if experiment in m.TPU_EXPERIMENTS else 1])
    actual=jax.device_put(actual,named_sharding_tree(actual,mesh));layout=coordinate_layout(q,mesh);q=jax.device_put(q,layout)
    tx=optimizer(q,spec);state,slayout=sharded_optimizer(tx,q,layout,mesh)
    if spec['mode']=='exact':state=jax.device_put(payload['optimizer'],slayout)
    before=3 if spec['mode']=='exact' else 0
    assert m.adam_counts(state)==[before]*6
    args=(q,state,actual,{},jax.device_put(t,batch_sharding(mesh)),jnp.asarray(cursor,jnp.float32))
    result=m.make_kernel(p.cfg,p.tc,p.columns,spec,tx,tpu=experiment in m.TPU_EXPERIMENTS)(*args)
    assert bool(result[2]['finite']);assert m.adam_counts(result[1])==[before+1]*6
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result[0]))
    own=dict(parameters=result[0],optimizer=result[1],best=q,best_step=jnp.asarray(cursor))
    store=RecoveryCheckpointStore(tmp_path,'own');store.save('ARM',{'suite':'new'},own,cursor+1)
    restored=store.restore('ARM',{'suite':'new'},own)[0]
    for a,b in zip(jax.tree.leaves(own),jax.tree.leaves(restored)):np.testing.assert_array_equal(a,b)


def test_real147_loader_checks_adam_not_global(tmp_path,monkeypatch):
    p,t,ancestor,base,payload,pin136=setup_sources();spec=m.recipes(149)[0]
    code={}
    for file in ('extent/reset_streamed_gpu.py','extent/reset_followup.py','extent/competitor_recovery.py'):
        raw=(m.ROOT/file).read_bytes().replace(b'\r\n',b'\n');target=tmp_path/file
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);code[file]=hashlib.sha256(raw).hexdigest()
    store=RecoveryCheckpointStore(tmp_path/'cache'/'147-original','prefix')
    meta=store.save('ARM',dict(model=asdict(p.cfg),recipes=[spec],code=code),payload,31126)
    raw=(store.directory('ARM')/'checkpoint.json').read_bytes()
    pin=dict(repo_id='offline',revision='pin',prefix='prefix',slot='ARM',path='prefix/ARM/checkpoint.json',manifest_sha256=hashlib.sha256(raw).hexdigest(),metadata=json.loads(raw),adam_step=3,best_step=int(payload['best_step']))
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'source_pin',lambda key:pin)
    restored,_=m.restore_source('147-original',HubArtifactConfig('offline','dataset',''),tmp_path/'cache')
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(payload)):np.testing.assert_array_equal(a,b)
    pin['adam_step']=31126
    with pytest.raises(ValueError,match='Adam'):m.restore_source('147-original',HubArtifactConfig('offline','dataset',''),tmp_path/'cache')


def test_actual_cli_and_hardware_dispatch(tmp_path,monkeypatch,capsys):
    from scripts import m3q_continuation_matrix_campaign as runner
    commands=[];tpus=[]
    def run(args):
        tpus.append(args.experiment);return dict(status='deadline_partial',branches={})
    monkeypatch.setattr(runner,'run',run)
    class Child:
        returncode=0
        def __init__(self,cmd,**kw):
            commands.append(cmd);worker=int(cmd[cmd.index('--worker')+1]);exp=int(cmd[cmd.index('--experiment')+1])
            assert 'scripts.m3q_continuation_matrix_campaign' in cmd and cmd[-1]=='--no-telegram'
            assert kw['env']['CUDA_VISIBLE_DEVICES']==str(worker)
            (tmp_path/f'extent-m3q-continuation-matrix-{exp}-worker-{worker}.json').write_text(json.dumps(dict(status='deadline_partial')))
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    for exp in range(149,155):
        c=runner.main(['--experiment',str(exp),'--plan-only']);assert c['recipes']==m.recipes(exp)
        r=runner.main(['--experiment',str(exp),'--output-dir',str(tmp_path),'--no-telegram']);assert r['status']=='deadline_partial'
    assert tpus==[149,150,151,152] and len(commands)==4
    for exp,limit in m.LIMITS.items():
        with pytest.raises(ValueError):runner.session_budget(exp,limit+.1,30)
    with pytest.raises(ValueError):runner.contract_for(149,0)
    capsys.readouterr()


@pytest.mark.parametrize('experiment,bad_replay',[(149,False),(151,False),(149,True)])
def test_actual_runner_initialization_first_update_deadline_and_checkpoint(tmp_path,monkeypatch,experiment,bad_replay):
    from types import SimpleNamespace
    from importlib.metadata import version
    import time
    import extent.config,extent.competitor_tpu,extent.reset_long_patient,extent.t4_scale_data
    import extent.hf_artifact_sync,extent.fold_continue_recovery
    from scripts import m3q_continuation_matrix_campaign as runner
    p,t,ancestor,base,payload,pin136=setup_sources();spec=m.recipes(experiment)[0]
    cursor=31126;source_spec=m.recipes(149)[0]
    ref=float(token_nll(p.cfg,base,payload['parameters'],t,p.columns,source_spec,jnp.asarray(cursor),block_fn=block_forward).mean())
    reference={k:dict(windows=[ref+(1. if bad_replay else 0.)]) for k in 'ABC'}
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    pin=dict(source_step=cursor,adam_step=3,runtime=runtime,reference=reference,
        metadata=dict(step=cursor,contract=dict(recipes=[source_spec])))
    monkeypatch.setattr(extent.fold_continue_recovery,'source_pin',lambda key:pin136)
    monkeypatch.setattr(m,'restore_source',lambda *a:(payload,pin))
    monkeypatch.setattr(m,'base_coordinates',lambda *a:ancestor)
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}))
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ['A','B','C','FIT']},{}))
    monkeypatch.setattr(extent.competitor_tpu,'training_batch',lambda *a:np.asarray(t))
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:HubArtifactConfig('offline','dataset',''))
    devices=jax.devices
    monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu')[:8] if platform=='tpu' else devices(platform))
    class LocalSession:
        def __init__(self):
            self.result=dict(status='preflight',branches={},pending_slots=[],runtime=runtime,cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0))
            self.store=RecoveryCheckpointStore(tmp_path/'state','own')
            self.wall=time.monotonic()+3600;self.deadline=self.wall-1800
        def check(self):
            if any(b.get('accepted_new_updates',0)>=1 for b in self.result['branches'].values()):raise TimeoutError('test work deadline')
        def persist(self,**kw):return True
        def emit(self,*a):pass
        def checkpoint(self,name,value,step):self.store.save(name,{'own':True},value,step)
    session=LocalSession();monkeypatch.setattr(runner,'session',lambda *a:session)
    args=SimpleNamespace(experiment=experiment,worker=None,max_steps=262144,cache_dir=str(tmp_path/'cache'))
    if bad_replay:
        with pytest.raises(RuntimeError):runner.run(args)
        assert session.result['cost'].get('accepted_updates',0)==0
        assert 'source NLL replay differs' in session.result['error']['message']
    else:
        report=runner.run(args);assert report['status']=='deadline_partial'
        assert report['cost']['accepted_updates']==1
        row=next(iter(report['branches'].values()));assert row['step']==cursor+1
        saved=json.loads((session.store.directory(spec['name'])/'checkpoint.json').read_text())
        assert saved['step']==cursor+1 and row['first_update_health']['health']['finite']


@pytest.mark.parametrize('failure',['parity','HBM'])
def test_tpu_engine_falls_back_without_accepting_bad_compiled_proposal(monkeypatch,failure):
    p,t,ancestor,base,payload,pin136=setup_sources();spec=m.recipes(149)[0]
    tx=optimizer(payload['parameters'],spec)
    engine=m.make_kernel(p.cfg,p.tc,p.columns,spec,tx,tpu=True)
    if failure=='parity':
        import extent.reset_followup as f
        original=f.compare_gradients
        def reject(reference,actual):
            result=original(reference,actual);result['passed']=False;return result
        monkeypatch.setattr(f,'compare_gradients',reject)
    else:
        def reject(*args):raise MemoryError('compiled test HBM guard')
        engine.compiled=reject
    args=(payload['parameters'],payload['optimizer'],base,{},t,jnp.asarray(31126.))
    found=engine(*args);expected=engine.streamed(*args)
    assert engine.selected=='streamed' and engine.compiled is None
    assert bool(found[2]['finite']) and not found[2]['engine_gate']['compiled_accepted']
    for a,b in zip(jax.tree.leaves(found[:2]),jax.tree.leaves(expected[:2])):np.testing.assert_array_equal(a,b)
