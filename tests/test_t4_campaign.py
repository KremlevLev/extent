"""Engineering CPU checks. Not a physical T4 run or an experimental result."""
from pathlib import Path
import json
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from extent.t4_patient import read_parameter_chunks, check_path
from extent.chunked_checkpoint import ChunkedCheckpointStore
from extent.t4_search import initialize, materialize, fit_step, RECIPES
from scripts.m3q_t4_preflight import check_streaming, tiny_patient
from scripts.m3q_t4_campaign import contract_for, main


def test_streamed_layer_forward_and_tiled_head_match_real_model():
    assert check_streaming(jax.devices('cpu')[0])['passed']


def test_parameter_only_reader_ignores_optimizer_and_verifies_sha(tmp_path):
    store=ChunkedCheckpointStore(tmp_path,'unused')
    meta=store.save('source',{},dict(parameters={'x':np.arange(8,dtype=np.float32)},
        optimizer={'large':np.ones(16,dtype=np.float32)}),32768)
    for row in meta['leaves']:
        if row['path'][0]=='optimizer':
            for c in row['chunks']:(store.directory('source')/c['file']).unlink()
    found=read_parameter_chunks(meta,store.directory('source'),'unused',None)
    np.testing.assert_array_equal(found['x'],np.arange(8,dtype=np.float32))
    chunk=next(r for r in meta['leaves'] if r['path'][0]=='parameters')['chunks'][0]
    (store.directory('source')/chunk['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA'):read_parameter_chunks(meta,store.directory('source'),'unused',None)


def test_all_local_search_recipes_start_exactly_at_patient():
    patient,_,params,t=tiny_patient(jax.devices('cpu')[0]);p=params['layers_0']
    cols=2*int(patient.cfg.hidden_size*patient.cfg.mamba.expand)+2*patient.cfg.mamba.mimo_rank*patient.cfg.mamba.d_state
    for recipe in RECIPES:
        q=initialize(p,recipe);actual=materialize(p,q,recipe,cols)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(p)):
            np.testing.assert_array_equal(a,b)
    small=initialize(p,'ADD32');large=initialize(p,'ADD256')
    for name in small:np.testing.assert_array_equal(small[name]['a'],large[name]['a'][:,:32])


def test_local_fit_updates_are_finite_and_protection_is_enforced():
    patient,_,params,t=tiny_patient(jax.devices('cpu')[0]);p=params['layers_0'];cfg=patient.cfg
    cols=2*int(cfg.hidden_size*cfg.mamba.expand)+2*cfg.mamba.mimo_rank*cfg.mamba.d_state
    q=initialize(p,'ADD32');tx,step,loss=fit_step(cfg,'ADD32',cols);state=tx.init(q)
    x=jnp.ones((1,8,cfg.hidden_size),jnp.bfloat16);target=jnp.ones_like(x)*.01
    after,_,health=step(q,state,p,x,target)
    assert bool(health['finite'])
    assert any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(q),jax.tree.leaves(after)))
    effective=materialize(p,after,'ADD32',cols)
    np.testing.assert_array_equal(effective['mamba']['in_proj']['kernel'][:,cols:],p['mamba']['in_proj']['kernel'][:,cols:])


def test_fit_checkpoint_restores_next_optimizer_update_exactly(tmp_path):
    patient,_,params,t=tiny_patient(jax.devices('cpu')[0]);p=params['layers_0'];cfg=patient.cfg
    q=initialize(p,'ADD32');tx,step,_=fit_step(cfg,'ADD32',32);state=tx.init(q)
    x=jnp.ones((1,4,cfg.hidden_size),jnp.bfloat16);target=x*.02
    q,state,_=step(q,state,p,x,target)
    store=ChunkedCheckpointStore(tmp_path,'unused');template=dict(parameters=q,optimizer=state)
    store.save('a',{},template,1);restored,_=store.restore('a',{},template)
    expected=step(q,state,p,x,target);actual=step(restored['parameters'],restored['optimizer'],p,x,target)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):np.testing.assert_array_equal(a,b)


def test_plans_use_distinct_ids_and_invalid_limits_stop_before_gpu():
    assert len({contract_for(i)['experiment'] for i in (116,117,118,119)})==4
    with pytest.raises(ValueError,match='reserve'):main(['--experiment','116','--max-wall-hours','.5','--save-reserve-minutes','45','--plan-only'])
    with pytest.raises(ValueError,match='unsafe'):check_path('../state.npy')
