"""EXP143-146 recipes and exact immutable checkpoint restoration."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import hashlib,json
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import unfreeze
from extent.reset_followup import recipes as old_recipes
from extent.competitor_recovery import initialize,optimizer,effective_layer,with_reset_targets,arm_for
from extent.recovery_subspace import initialize_corrections
from extent.t4_patient import read_parameter_chunks
from extent.recovery_durability import RecoveryCheckpointStore
from extent.hf_artifact_sync import restore_artifact
ROOT=Path(__file__).resolve().parents[1]


def recipes(experiment):
    full,projections=old_recipes(136)
    ce=old_recipes(138)[0]
    if experiment==143:
        return [dict(full,name='ALL-CONTINUE',source='136-all',mode='exact',source_seed=123),dict(projections,name='PROJECTIONS-CONTINUE',source='136-projections',mode='exact',source_seed=123)]
    if experiment==144:
        return [dict(ce,name='CE-CONTINUE',source='138-ce',mode='exact',source_seed=123),dict(ce,name='CE-LR-DIV3',source='138-ce',mode='lr-div3',source_seed=123,mamba_lr=ce['mamba_lr']/3,norm_lr=ce['norm_lr']/3,mlp_lr=ce['mlp_lr']/3)]
    if experiment==145:
        return [dict(full,name='SAME-FACTORS-NEW-ADAM',source='136-all',mode='fresh-adam',source_seed=123),dict(full,name='FOLD-NEW-R32-NEW-ADAM',source='136-all',mode='fold-new',source_seed=123,reset=False,ramp_steps=0)]
    if experiment==146:
        return [dict(full,name='SEED456-ALL',source='111-456',mode='new-seed',source_seed=456),dict(projections,name='SEED456-PROJECTIONS',source='111-456',mode='new-seed',source_seed=456)]
    raise ValueError('EXP143-146 only')


def restore_source(key,hub,cache,check=lambda:None):
    pin=json.loads((ROOT/'results/EXP-143-146-source-pin.json').read_text())[key]
    sourcehub=replace(hub,repo_id=pin['repo_id'],repo_type='dataset',revision=pin['revision'])
    store=RecoveryCheckpointStore(Path(cache)/key,pin['prefix'],sourcehub);slot=pin['slot'];directory=store.directory(slot);pointer=directory/'checkpoint.json'
    check()
    if not pointer.exists():restore_artifact(pointer,pin['path'],sourcehub)
    if hashlib.sha256(pointer.read_bytes()).hexdigest()!=pin['manifest_sha256']:raise ValueError('source manifest SHA differs')
    meta=json.loads(pointer.read_text())
    if meta!=pin['metadata']:raise ValueError('source metadata differs')
    for name in ('extent/reset_streamed_gpu.py','extent/reset_followup.py','extent/competitor_recovery.py'):
        expected=meta['contract'].get('code',{}).get(name)
        if expected is None or hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:raise ValueError('source numerical code differs: '+name)
    q=read_parameter_chunks(meta,directory,pin['prefix']+'/'+slot,sourcehub,check)
    tx=optimizer(q,meta['contract']['recipes'][0])
    with jax.default_device(jax.devices('cpu')[0]):template=dict(parameters=q,optimizer=tx.init(q),best=q,best_step=jnp.asarray(-1,jnp.int32))
    saved=store.restore(slot,meta['contract'],template)
    if saved is None or saved[1]!=meta:raise ValueError('source restore failed')
    payload=saved[0]
    counts=[int(np.asarray(jax.device_get(v))) for path,v in jax.tree.leaves_with_path(payload['optimizer']) if getattr(path[-1],'name',getattr(path[-1],'key',None))=='count']
    if len(counts)!=6 or any(v!=meta['step'] for v in counts):raise ValueError('source Adam/schedule count differs')
    if int(payload['best_step'])!=pin['best_step']:raise ValueError('source best step differs')
    return payload,pin


def structural_copy(tree):
    # Copy containers only. Host BF16 MLP/embedding buffers can be multi-GB.
    return jax.tree.map(lambda value:value,tree)


def attach_resets(base,cfg):
    base=structural_copy(base)
    for i in cfg.mamba_layer_indices:
        p=base[f'layers_{i}'];reset=with_reset_targets({'mamba':p['mamba']},cfg,i)
        base[f'layers_{i}']=dict(p,**reset)
    return base


def fold_weights(base,q,cfg,columns,spec):
    base=attach_resets(base,cfg) if spec['reset'] else structural_copy(base)
    for i in cfg.mamba_layer_indices:
        p=base[f'layers_{i}'];part={'mamba':p['mamba']}
        if '_reset' in p:part['_reset']=p['_reset']
        if 'mlp_lora' in q[str(i)]:raise ValueError('this source folding supports Mamba corrections only')
        changed=effective_layer(part,q[str(i)],columns,spec,jnp.asarray(1.))
        out=dict(p,mamba=jax.device_get(changed['mamba']));out.pop('_reset',None);base[f'layers_{i}']=out
    return base


def prepare(patient,spec,payload=None,pin=None,payload130=None):
    if spec['mode']=='new-seed':
        p=SimpleNamespace(cfg=patient.cfg,base=patient.base,current=patient.base,coordinates=patient.coordinates,columns=patient.columns)
        q=initialize(p,spec);base=attach_resets(patient.base,patient.cfg)
        return base,q,0
    source_spec=pin['metadata']['contract']['recipes'][0]
    base=fold_weights(patient.base,payload130['parameters'],patient.cfg,patient.columns,arm_for(130,1)) if spec['source']=='138-ce' else structural_copy(patient.base)
    if source_spec['reset']:base=attach_resets(base,patient.cfg)
    q=payload['parameters']
    if spec['mode']=='fold-new':
        base=fold_weights(base,q,patient.cfg,patient.columns,source_spec)
        coordinates=initialize_corrections(base,patient.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=32,head_dim=patient.cfg.mamba.head_dim)
        p=SimpleNamespace(cfg=patient.cfg,base=base,current=base,coordinates=coordinates,columns=patient.columns)
        fresh=initialize(p,spec)
        for i in patient.cfg.mamba_layer_indices:fresh[str(i)]['pregate_weight']=q[str(i)]['pregate_weight']
        q=fresh
    return base,q,pin['metadata']['step']
