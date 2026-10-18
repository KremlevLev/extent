"""Real chunk roundtrips followed by actual hybrid evaluation/gradient kernels."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from tests.test_reset_streamed_gpu import patient
from extent.competitor_recovery import initialize, optimizer
from extent.reset_followup import recipes, prepare, block_forward, Inspector
from extent.competitor_tpu import token_nll, make_step as compiled_step
from extent.reset_streamed_gpu import make_step as streamed_step
from extent.recovery_durability import RecoveryCheckpointStore
from extent.hf_artifact_sync import HubArtifactConfig
from extent.sharding import create_v5e_mesh, named_sharding_tree, batch_sharding
from scripts.m3q_peer_tpu_campaign import coordinate_layout, sharded_optimizer


@pytest.mark.parametrize('experiment,worker',[(136,0),(136,1),(138,0)])
def test_actual_source_restore_then_eval_profile_and_update(tmp_path,monkeypatch,experiment,worker):
    from extent import reset_long_fixed_sources as fixed
    from extent import reset_long_recovery as old
    p,t=patient();spec=recipes(experiment)[worker]
    from extent.competitor_recovery import arm_for
    source={'parameters':initialize(p,arm_for(130,1))}
    base,q,_=prepare(p,spec,source)
    tx=optimizer(q,spec);state=tx.init(q)
    # Nonzero moments and clocks must survive; no fresh-Adam shortcut.
    for _ in range(3):
        changes,state=tx.update(jax.tree.map(jnp.ones_like,q),state,q)
        import optax
        q=optax.apply_updates(q,changes)
    payload=dict(parameters=q,optimizer=state,best=q,best_step=jnp.asarray(1,jnp.int32))
    model=asdict(p.cfg);code={}
    for name in ('extent/reset_streamed_gpu.py','extent/reset_followup.py','extent/competitor_recovery.py'):
        raw=(Path(__file__).resolve().parents[1]/name).read_bytes().replace(b'\r\n',b'\n')
        destination=tmp_path/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
        code[name]=hashlib.sha256(raw).hexdigest()
    store=RecoveryCheckpointStore(tmp_path/'cache'/'source','prefix')
    meta=store.save('ARM',dict(model=model,recipes=[spec],code=code),payload,3)
    raw=(store.directory('ARM')/'checkpoint.json').read_bytes()
    pin=dict(repo_id='offline',revision='immutable',prefix='prefix',slot='ARM',path='prefix/ARM/checkpoint.json',
             manifest_sha256=hashlib.sha256(raw).hexdigest(),metadata=meta,best_step=1)
    (tmp_path/'results').mkdir();(tmp_path/'results/EXP-143-146-source-pin.json').write_text(json.dumps({'source':pin}))
    monkeypatch.setattr(old,'ROOT',tmp_path);monkeypatch.setattr(fixed,'ROOT',tmp_path)
    hub=HubArtifactConfig('offline','dataset','')
    broken,_=old.restore_source('source',hub,tmp_path/'cache')
    # Reproduce production defect through the REAL evaluator, without mocking NLL.
    with pytest.raises(KeyError):
        token_nll(p.cfg,base,broken['parameters'],t,p.columns,spec,jnp.asarray(2048.),block_fn=block_forward)
    restored,_=fixed.restore_source('source',hub,tmp_path/'cache')
    assert jax.tree.structure(restored)==jax.tree.structure(payload)
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(payload)):
        np.testing.assert_array_equal(a,b)
    expected=token_nll(p.cfg,base,q,t,p.columns,spec,jnp.asarray(2048.),block_fn=block_forward)
    for field in ('parameters','best'):
        actual=jax.jit(lambda r:token_nll(p.cfg,base,r,t,p.columns,spec,jnp.asarray(2048.),block_fn=block_forward))(restored[field])
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
    profile=Inspector(p.cfg,base,p.columns,spec).profile(restored['parameters'],t,2048)
    assert len(profile['layers'])==p.cfg.num_layers
    mesh=create_v5e_mesh(jax.devices('cpu')[:1]);base=jax.device_put(base,named_sharding_tree(base,mesh))
    q=jax.device_put(restored['parameters'],coordinate_layout(restored['parameters'],mesh));tx=optimizer(q,spec)
    _,layout=sharded_optimizer(tx,q,coordinate_layout(q,mesh),mesh)
    state=jax.device_put(restored['optimizer'],layout)
    arguments=(q,state,base,{},jax.device_put(t,batch_sharding(mesh)),jnp.asarray(2048.))
    reference=jax.jit(compiled_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=block_forward))(*arguments)
    actual=streamed_step(p.cfg,p.tc,p.columns,spec,tx)(*arguments)
    assert bool(actual[2]['finite'])
    for a,b in zip(jax.tree.leaves(actual[:2]),jax.tree.leaves(reference[:2])):
        np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-6)
    own=dict(parameters=actual[0],optimizer=actual[1],best=restored['best'],best_step=restored['best_step'])
    local=RecoveryCheckpointStore(tmp_path/'own','fixed-v2')
    local.save('ARM',{'fixed':True},own,4)
    replay=local.restore('ARM',{'fixed':True},own)[0]
    assert jax.tree.structure(replay)==jax.tree.structure(own)
    for a,b in zip(jax.tree.leaves(replay),jax.tree.leaves(own)):np.testing.assert_array_equal(a,b)


def test_only_empty_frozen_attention_can_be_recreated():
    from extent.reset_long_fixed_sources import complete_coordinates
    with pytest.raises(ValueError,match='Mamba'):
        complete_coordinates({},dict(num_layers=2,attention_layer_indices=[1]))
    q={'0':{'learned':object()},'1':{'learned_attention':object()}}
    out=complete_coordinates(q,dict(num_layers=2,attention_layer_indices=[1]))
    assert out['0'] is q['0'] and out['1'] is q['1']
    with pytest.raises(ValueError,match='unexpected'):
        complete_coordinates({'9':{}},dict(num_layers=2,attention_layer_indices=[1]))


def test_fixed_launcher_is_isolated_and_old_contracts_unchanged(capsys):
    from scripts.m3q_reset_long_fixed_campaign import main
    from scripts.m3q_reset_long_campaign import contract_for
    plan=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-143-146-engineering-plan.json').read_text())
    for row in plan['experiments']:assert contract_for(row['experiment'])==row['contract']
    for e in (143,144):
        c=main(['--experiment',str(e),'--plan-only'])
        assert c['protocol']=='exp143-144-long-fixed-v2'
        assert 'long-fixed-v2' in c['execution_namespace']
        assert 'scripts/m3q_reset_long_fixed_campaign.py' in c['code']
        assert 'extent/reset_long_fixed_sources.py' in c['code']
    with pytest.raises(SystemExit):main(['--experiment','146','--plan-only'])
    capsys.readouterr()


def test_parent_dispatches_fixed_workers_without_touching_old_reports(tmp_path,monkeypatch):
    from scripts import m3q_reset_long_fixed_campaign as runner
    commands=[]
    class Child:
        returncode=0
        def __init__(self,cmd,**kwargs):
            commands.append(cmd)
            worker=int(cmd[cmd.index('--worker')+1])
            assert 'scripts.m3q_reset_long_fixed_campaign' in cmd
            assert kwargs['env']['CUDA_VISIBLE_DEVICES']==str(worker)
            assert '--no-telegram' in cmd
            (tmp_path/f'extent-m3q-long-fixed-143-worker-{worker}.json').write_text(json.dumps(dict(status='completed',branches={})))
        def poll(self):return 0
        def wait(self):return 0
    old=tmp_path/'extent-m3q-long-143-worker-0.json';old.write_text('old failed report')
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    result=runner.main(['--experiment','143','--output-dir',str(tmp_path),'--no-telegram'])
    assert len(commands)==2 and result['status']=='completed'
    assert old.read_text()=='old failed report'
