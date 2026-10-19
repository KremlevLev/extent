"""Mature nonzero source factors, BF16/host checkpoint path, real gradients."""
from dataclasses import replace,asdict
from types import SimpleNamespace
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from extent import HybridForCausalLM,tiny_config
from extent.model import _attention_source_config
from extent.competitor_recovery import initialize,optimizer
from extent.recovery_subspace import initialize_corrections
from extent.competitor_tpu import token_nll,make_step as compiled_step
from extent.reset_followup import block_forward as source_block
from extent import rank_growth_153 as growth
from extent.recovery_durability import RecoveryCheckpointStore


def mature_source():
    c=tiny_config()
    cfg=replace(c,mla=replace(c.mla,implementation='qwen3_gqa'),param_dtype='bfloat16',
        compute_dtype='bfloat16',logits_dtype='float32',tie_word_embeddings=True,remat_policy='none')
    tokens=jnp.arange(32,dtype=jnp.int32)[None]%cfg.vocab_size
    base=HybridForCausalLM(cfg).init(jax.random.key(53),tokens)['params']
    cols=2*int(cfg.hidden_size*cfg.mamba.expand)+2*cfg.mamba.mimo_rank*cfg.mamba.groups*cfg.mamba.d_state
    coords=initialize_corrections(base,cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=32)
    p=SimpleNamespace(cfg=cfg,base=base,current=base,coordinates=coords,columns=cols,tc=_attention_source_config(cfg))
    q=initialize(p,growth.recipes(153)[0]);rng=np.random.default_rng(153)
    for i in cfg.mamba_layer_indices:
        for f in q[str(i)]['lora'].values():
            f['b']=jnp.asarray(rng.normal(0,.04,f['b'].shape),jnp.float32)
        q[str(i)]['pregate_weight']=jnp.asarray(rng.uniform(.7,1.3,q[str(i)]['pregate_weight'].shape),jnp.float32)
    return p,tokens,q


@pytest.mark.parametrize('worker',[0,1])
def test_host_bf16_source_rank_growth_preserves_nll_and_trains_extra(worker,tmp_path):
    p,t,q=mature_source();spec=growth.recipes(153)[worker]
    # Production loader restores NumPy arrays, NOT the device-only state used
    # by older folding tests. Material source B is nonzero, not a tiny update.
    store=RecoveryCheckpointStore(tmp_path/'source','source');tx=optimizer(q,spec)
    source=dict(parameters=q,optimizer=tx.init(q),best=q,best_step=jnp.asarray(31109))
    store.save('SOURCE',{'source':True},source,31126)
    host=store.restore('SOURCE',{'source':True},source)[0]['parameters']
    base=jax.device_get(p.base);q=growth.grow_coordinates(base,host,p.cfg,spec['total_rank'])
    for old,new in zip(jax.tree.leaves(host),jax.tree.leaves({k:{n:v for n,v in row.items() if n!='extra_lora'} for k,row in q.items()})):
        np.testing.assert_array_equal(old,new)
    original=jax.jit(lambda base,q,t:token_nll(p.cfg,base,q,t,p.columns,spec,jnp.asarray(31126.),block_fn=source_block))
    grown=jax.jit(lambda base,q,t:token_nll(p.cfg,base,q,t,p.columns,spec,jnp.asarray(31126.),block_fn=growth.block_forward))
    for shift in (0,7,31):
        tokens=(t+shift)%p.cfg.vocab_size
        np.testing.assert_allclose(grown(base,q,tokens),original(base,host,tokens),rtol=0,atol=.003)
    for i in p.cfg.mamba_layer_indices:
        assert q[str(i)]['lora']['in_proj']['a'].shape[1]+q[str(i)]['extra_lora']['in_proj']['a'].shape[1]==spec['total_rank']
        assert not np.any(q[str(i)]['extra_lora']['out_proj']['b'])
    q=jax.tree.map(jnp.asarray,q);base=jax.tree.map(jnp.asarray,base);tx=optimizer(q,spec);state=tx.init(q)
    kernel=growth.make_kernel(p.cfg,p.tc,p.columns,spec,tx,tpu=False,return_gradients=True)
    actual=kernel(q,state,base,{},t,jnp.asarray(31126.,jnp.float32))
    expected=jax.jit(compiled_step(p.cfg,p.tc,p.columns,spec,tx,block_fn=growth.block_forward,return_gradients=True))(q,state,base,{},t,jnp.asarray(31126.,jnp.float32))
    assert bool(actual[2]['finite']) and growth.adam_counts(actual[1])==[1]*6
    assert any(np.any(np.asarray(actual[0][str(i)]['extra_lora']['out_proj']['b'])) for i in p.cfg.mamba_layer_indices)
    assert any(np.any(np.asarray(expected[3][str(i)]['extra_lora']['out_proj']['b'])) for i in p.cfg.mamba_layer_indices)
    for i in p.cfg.mamba_layer_indices:
        np.testing.assert_array_equal(np.asarray(expected[3][str(i)]['extra_lora']['in_proj']['b'])[:,p.columns:],0.)
    # Mature BF16 forward boundaries change fusion/rounding. Use the EXISTING
    # global gradient gate (.15/.98), not an invented elementwise ratio near0.
    from extent.reset_followup import compare_gradients
    parity=compare_gradients(expected[3],actual[3]);assert parity['passed'],parity
    assert abs(float(actual[2]['ce'])-float(expected[2]['ce']))<=.03
    import optax
    update,after=jax.jit(tx.update)(actual[3],state,q)
    proposal=optax.apply_updates(q,update)
    for a,b in zip(jax.tree.leaves(actual[:2]),jax.tree.leaves((proposal,after))):np.testing.assert_allclose(a,b,rtol=2e-5,atol=1e-7)
    own=dict(parameters=actual[0],optimizer=actual[1],best=q,best_step=jnp.asarray(31126))
    store=RecoveryCheckpointStore(tmp_path/'own','v2');store.save('ARM',{'new':True},own,31127)
    restored=store.restore('ARM',{'new':True},own)[0]
    for a,b in zip(jax.tree.leaves(own),jax.tree.leaves(restored)):np.testing.assert_array_equal(a,b)


def test_diagnostics_include_failure_values_and_worst_window():
    pin=dict(reference={p:dict(windows=[5.,6.]) for p in 'ABC'})
    actual={p:dict(windows=[5.,6.]) for p in 'ABC'}
    actual['B']['windows'][1]=6.1
    r=growth.replay_report(actual,pin)
    assert not r['passed'] and r['panels']['B']['worst_window']==1
    assert r['panels']['B']['worst_actual']==6.1
    assert r['panels']['B']['worst_reference']==6.
    assert r['tolerance']==.03


def test_dispatch_new_namespace_and_old_plans_unchanged(tmp_path,monkeypatch,capsys):
    from scripts import m3q_rank_growth_153_campaign as runner
    from scripts.m3q_continuation_matrix_campaign import contract_for as old
    plan=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-149-154-engineering-plan.json').read_text())
    for row in plan['experiments']:assert old(row['experiment'])==row['contract']
    commands=[]
    class Child:
        returncode=0
        def __init__(self,cmd,**kw):
            commands.append(cmd);worker=int(cmd[cmd.index('--worker')+1])
            assert 'scripts.m3q_rank_growth_153_campaign' in cmd and cmd[-1]=='--no-telegram'
            (tmp_path/f'extent-m3q-rank-growth-153-worker-{worker}.json').write_text(json.dumps(dict(status='deadline_partial')))
        def poll(self):return 0
        def wait(self):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Child)
    plan=runner.main(['--experiment','153','--plan-only'])
    assert plan['protocol']=='exp153-rank-growth-v2'
    assert [r['total_rank'] for r in plan['recipes']]==[64,256]
    result=runner.main(['--experiment','153','--output-dir',str(tmp_path),'--no-telegram'])
    assert result['status']=='deadline_partial' and len(commands)==2
    with pytest.raises(SystemExit):runner.main(['--experiment','154','--plan-only'])
    capsys.readouterr()


@pytest.mark.parametrize('bad_replay',[False,True])
def test_real_runner_first_update_and_failure_diagnostics_persisted(tmp_path,monkeypatch,bad_replay):
    from importlib.metadata import version
    import time
    import extent.config,extent.competitor_tpu,extent.reset_long_patient,extent.t4_scale_data
    import extent.hf_artifact_sync
    from scripts import m3q_rank_growth_153_campaign as runner
    p,t,q=mature_source();source_spec=growth.recipes(153)[0];cursor=31126
    base=jax.tree.map(jnp.asarray,p.base)
    ref=float(token_nll(p.cfg,base,q,t,p.columns,source_spec,jnp.asarray(cursor),block_fn=source_block).mean())
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    pin=dict(source_step=cursor,adam_step=18540,runtime=runtime,
        reference={k:dict(windows=[ref+(1 if bad_replay else 0)]) for k in 'ABC'})
    payload=dict(parameters=q,optimizer=optimizer(q,source_spec).init(q),best=q,best_step=jnp.asarray(cursor))
    monkeypatch.setattr(growth,'old_prepare',lambda patient,spec,payload,pin,ancestor:(p.base,payload['parameters'],cursor))
    monkeypatch.setattr(growth,'restore_source',lambda *a:(payload,pin))
    monkeypatch.setattr(growth,'base_coordinates',lambda *a:{})
    monkeypatch.setattr(extent.reset_long_patient,'Patient',lambda *a,**kw:p)
    monkeypatch.setattr(extent.config,'load_config',lambda *a:(p.cfg,{}))
    monkeypatch.setattr(extent.t4_scale_data,'load_data',lambda *a:({k:np.asarray(t) for k in ('A','B','C','FIT')},{}))
    monkeypatch.setattr(extent.competitor_tpu,'training_batch',lambda *a:np.asarray(t))
    from extent.hf_artifact_sync import HubArtifactConfig
    monkeypatch.setattr(extent.hf_artifact_sync,'artifact_config_from_env',lambda:HubArtifactConfig('offline','dataset',''))
    devices=jax.devices
    monkeypatch.setattr(jax,'devices',lambda platform=None:devices('cpu')[:1] if platform=='gpu' else devices(platform))
    path=tmp_path/'persisted.json'
    class LocalSession:
        def __init__(self):
            self.result=dict(status='preflight',branches={},pending_slots=[],runtime=runtime,cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0))
            self.store=RecoveryCheckpointStore(tmp_path/'state','v2')
            self.wall=time.monotonic()+3600;self.deadline=self.wall-1800
        def check(self):
            if any(b.get('accepted_new_updates',0)>=1 for b in self.result['branches'].values()):raise TimeoutError('test work deadline')
        def persist(self,**kw):path.write_text(json.dumps(self.result));return True
        def emit(self,*a):pass
        def checkpoint(self,name,value,step):self.store.save(name,{'own':True},value,step)
    session=LocalSession();monkeypatch.setattr(runner,'session',lambda *a:session)
    args=SimpleNamespace(experiment=153,worker=0,max_steps=262144,cache_dir=str(tmp_path/'cache'))
    if bad_replay:
        with pytest.raises(RuntimeError):runner.run(args)
        saved=json.loads(path.read_text());row=next(iter(saved['branches'].values()))
        assert saved['cost'].get('accepted_updates',0)==0
        assert not row['source_replay']['passed']
        assert row['source_replay']['panels']['A']['worst_reference']>row['source_replay']['panels']['A']['worst_actual']
        assert 'source_controls'in row
    else:
        report=runner.run(args)
        assert report['cost']['accepted_updates']==1 and report['status']=='deadline_partial'
        row=next(iter(report['branches'].values()))
        assert row['step']==31127 and row['source_replay']['passed']
