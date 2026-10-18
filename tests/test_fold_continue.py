"""Continuation clocks, ancestry, real source restore and real next updates."""
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from tests.test_reset_streamed_gpu import patient
from extent.competitor_recovery import initialize,optimizer,arm_for
from extent.reset_long_recovery import recipes as old_recipes,prepare as old_prepare
from extent.reset_followup import recipes as reset_recipes,block_forward
from extent.fold_continue_recovery import recipes,prepare
from extent.hf_artifact_sync import HubArtifactConfig
from extent.recovery_durability import RecoveryCheckpointStore
from extent.competitor_tpu import token_nll,make_step as compiled_step
from extent.reset_streamed_gpu import make_step as streamed_step
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding
from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer


def setup_sources():
    p,t=patient();s136=reset_recipes(136)[0];q136=initialize(p,s136)
    q136=jax.tree.map(lambda a:a+1e-5,q136)
    original={'parameters':q136};pin136=dict(metadata=dict(step=12586,contract=dict(recipes=[s136])))
    base,q,_=old_prepare(p,old_recipes(145)[1],original,pin136)
    tx=optimizer(q,old_recipes(145)[1]);state=tx.init(q)
    import optax
    for _ in range(3):
        u,state=tx.update(jax.tree.map(jnp.ones_like,q),state,q);q=optax.apply_updates(q,u)
    payload=dict(parameters=q,optimizer=state,best=q,best_step=jnp.asarray(24874,jnp.int32))
    return p,t,q136,base,payload,pin136


@pytest.mark.parametrize('experiment,worker',[(147,0),(147,1),(148,0),(148,1)])
def test_base_source_clocks_and_real_update(experiment,worker,monkeypatch,tmp_path):
    import extent.fold_continue_recovery as module
    p,t,q136,base145,payload145,pin136=setup_sources()
    spec=recipes(experiment)[worker]
    monkeypatch.setattr(module,'source_pin',lambda key:pin136)
    if spec['source']=='145-fold':
        payload=payload145;pin=dict(metadata=dict(step=24965),adam_step=3)
        expected_base=base145;expected_cursor=24965
    else:
        q=q136;payload=dict(parameters=q,optimizer=optimizer(q,spec).init(q),best=q,best_step=jnp.asarray(12586))
        pin=pin136;expected_base,_,_=old_prepare(p,old_recipes(145)[0],payload,pin136);expected_cursor=12586
    base,q,cursor=prepare(p,spec,payload,pin,q136)
    assert cursor==expected_cursor
    assert q is payload['parameters']
    for a,b in zip(jax.tree.leaves(base),jax.tree.leaves(expected_base)):np.testing.assert_array_equal(a,b)
    actual=token_nll(p.cfg,base,q,t,p.columns,spec,jnp.asarray(cursor),block_fn=block_forward)
    reference=token_nll(p.cfg,expected_base,q,t,p.columns,spec,jnp.asarray(cursor),block_fn=block_forward)
    np.testing.assert_array_equal(actual,reference)
    mesh=create_v5e_mesh(jax.devices('cpu')[:1]);base=jax.device_put(base,named_sharding_tree(base,mesh))
    q=jax.device_put(q,coordinate_layout(q,mesh));tx=optimizer(q,spec)
    state,layout=sharded_optimizer(tx,q,coordinate_layout(q,mesh),mesh)
    if spec['mode']=='exact':state=jax.device_put(payload['optimizer'],layout)
    count=lambda state:[int(v) for path,v in jax.tree.leaves_with_path(state) if getattr(path[-1],'name',None)=='count']
    before=3 if spec['mode']=='exact' else 0
    assert count(state)==[before]*6
    args=(q,state,base,{},jax.device_put(t,batch_sharding(mesh)),jnp.asarray(cursor,jnp.float32))
    expected=jax.jit(compiled_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=block_forward))(*args)
    actual=streamed_step(p.cfg,p.tc,p.columns,spec,tx)(*args)
    assert bool(actual[2]['finite']) and count(actual[1])==[before+1]*6
    for a,b in zip(jax.tree.leaves(actual[:2]),jax.tree.leaves(expected[:2])):
        np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-6)
    own=dict(parameters=actual[0],optimizer=actual[1],best=q,best_step=jnp.asarray(cursor))
    store=RecoveryCheckpointStore(tmp_path,'own');store.save('ARM',{'new':True},own,cursor+1)
    replay=store.restore('ARM',{'new':True},own)[0]
    assert jax.tree.structure(replay)==jax.tree.structure(own)
    for a,b in zip(jax.tree.leaves(replay),jax.tree.leaves(own)):np.testing.assert_array_equal(a,b)


def test_restore_uses_source_adam_count_not_global_data_cursor(tmp_path,monkeypatch):
    import extent.fold_continue_recovery as module
    p,t,q136,base,payload,pin136=setup_sources();spec=old_recipes(145)[1]
    code={}
    for name in ('extent/reset_streamed_gpu.py','extent/reset_followup.py','extent/competitor_recovery.py'):
        raw=(Path(__file__).resolve().parents[1]/name).read_bytes().replace(b'\r\n',b'\n')
        destination=tmp_path/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
        code[name]=hashlib.sha256(raw).hexdigest()
    store=RecoveryCheckpointStore(tmp_path/'cache'/'145-fold','prefix')
    meta=store.save('ARM',dict(model=asdict(p.cfg),recipes=[spec],code=code),payload,24965)
    raw=(store.directory('ARM')/'checkpoint.json').read_bytes()
    pin=dict(repo_id='offline',revision='pin',prefix='prefix',slot='ARM',path='prefix/ARM/checkpoint.json',manifest_sha256=hashlib.sha256(raw).hexdigest(),metadata=json.loads(raw),adam_step=3,best_step=24874)
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'source_pin',lambda key:pin)
    restored,_=module.restore_source('145-fold',HubArtifactConfig('offline','dataset',''),tmp_path/'cache')
    assert jax.tree.structure(restored)==jax.tree.structure(payload)
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(payload)):np.testing.assert_array_equal(a,b)
    pin['adam_step']=24965
    with pytest.raises(ValueError,match='Adam'):module.restore_source('145-fold',HubArtifactConfig('offline','dataset',''),tmp_path/'cache')


def test_lr_divisors_keep_moments_and_change_only_update_scale():
    p,t,q136,base,payload,pin136=setup_sources();q=payload['parameters'];state=payload['optimizer']
    specs=[*recipes(147),recipes(148)[1]]
    changes=[]
    for spec in specs:
        u,after=optimizer(q,spec).update(jax.tree.map(jnp.ones_like,q),state,q);changes.append(u)
        assert [int(v) for path,v in jax.tree.leaves_with_path(after) if getattr(path[-1],'name',None)=='count']==[4]*6
    for i,divisor in ((1,3),(2,10)):
        for a,b in zip(jax.tree.leaves(changes[0]),jax.tree.leaves(changes[i])):
            np.testing.assert_allclose(b,a/divisor,rtol=2e-5,atol=1e-10)


def test_actual_cli_dispatch_and_budget(tmp_path,monkeypatch,capsys):
    from scripts import m3q_fold_continue_campaign as runner
    commands=[]
    class Child:
        returncode=0
        def __init__(self,cmd,**kw):
            commands.append(cmd);worker=int(cmd[cmd.index('--worker')+1]);experiment=int(cmd[cmd.index('--experiment')+1])
            assert 'scripts.m3q_fold_continue_campaign' in cmd and cmd[-1]=='--no-telegram'
            assert cmd[cmd.index('--max-wall-hours')+1]=='6.8'
            (tmp_path/f'extent-m3q-fold-continue-{experiment}-worker-{worker}.json').write_text(json.dumps(dict(status='deadline_partial',branches={})))
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    for e in (147,148):
        c=runner.main(['--experiment',str(e),'--plan-only'])
        assert c['protocol']=='exp147-148-fold-continue-v1' and c['first_goal_new_steps']==8192
        assert c['recipes']==recipes(e)
        result=runner.main(['--experiment',str(e),'--output-dir',str(tmp_path),'--no-telegram'])
        assert result['status']=='deadline_partial'
    assert len(commands)==4
    assert runner.session_budget(147,6.8,30)[0]==pytest.approx(6.3*3600)
    with pytest.raises(ValueError,match='7'):runner.session_budget(147,7.01,30)
    with pytest.raises(SystemExit):runner.main(['--experiment','145'])
    capsys.readouterr()
