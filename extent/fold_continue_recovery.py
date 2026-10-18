"""Immutable EXP145 continuation and repaired SAME-FACTORS control.

Global data cursor24965 is NOT Adam step12379. Restore all moments and six
clocks exactly, including empty GQA nodes. Base145 is reconstructed from the
original136 current12586 folded ONCE, not by folding145 corrections again.
"""
from pathlib import Path
from dataclasses import replace
import hashlib,json
import numpy as np
import jax
import jax.numpy as jnp
from extent.reset_long_recovery import recipes as long_recipes,attach_resets,fold_weights
from extent.reset_long_fixed_sources import complete_coordinates
from extent.competitor_recovery import optimizer
from extent.t4_patient import read_parameter_chunks
from extent.recovery_durability import RecoveryCheckpointStore
from extent.hf_artifact_sync import restore_artifact
ROOT=Path(__file__).resolve().parents[1]


def source_pin(key):
    file='EXP-147-148-source-pin.json' if key=='145-fold' else 'EXP-143-146-source-pin.json'
    return json.loads((ROOT/'results'/file).read_text())[key]


def recipes(experiment):
    treatment=long_recipes(145)[1]
    def continuation(divisor):
        return dict(treatment,name=f'CONTINUE145-LR-DIV{divisor}',source='145-fold',mode='exact',
            mamba_lr=treatment['mamba_lr']/divisor,norm_lr=treatment['norm_lr']/divisor,mlp_lr=treatment['mlp_lr']/divisor)
    if experiment==147:return [continuation(1),continuation(3)]
    if experiment==148:
        control=dict(long_recipes(145)[0],name='REPAIRED145-SAME-FACTORS-NEW-ADAM')
        return [control,continuation(10)]
    raise ValueError('EXP147/148 only')


def base_coordinates(hub,cache,check=lambda:None):
    """Only source136 q is needed for the frozen folded base, not its Adam/best."""
    pin=source_pin('136-all');sourcehub=replace(hub,repo_id=pin['repo_id'],repo_type='dataset',revision=pin['revision'])
    directory=Path(cache)/'136-all'/pin['slot'];pointer=directory/'checkpoint.json'
    check()
    if not pointer.exists():restore_artifact(pointer,pin['path'],sourcehub)
    if hashlib.sha256(pointer.read_bytes()).hexdigest()!=pin['manifest_sha256']:raise ValueError('base source manifest SHA differs')
    meta=json.loads(pointer.read_text())
    if meta!=pin['metadata']:raise ValueError('base source metadata differs')
    q=read_parameter_chunks(meta,directory,pin['prefix']+'/'+pin['slot'],sourcehub,check)
    return complete_coordinates(q,meta['contract']['model'])


def prepare(patient,spec,payload,pin,ancestor=None):
    if spec['source']=='145-fold':
        if ancestor is None:raise ValueError('immutable136 base corrections required')
        old=source_pin('136-all')['metadata']['contract']['recipes'][0]
        base=fold_weights(patient.base,ancestor,patient.cfg,patient.columns,old)
    else:
        # The missing145 control keeps SAME learned factors and resets Adam ONLY.
        base=attach_resets(patient.base,patient.cfg)
    return base,payload['parameters'],pin['metadata']['step']


def restore_source(key,hub,cache,check=lambda:None):
    pin=source_pin(key)
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
    q=complete_coordinates(q,meta['contract']['model'])
    tx=optimizer(q,meta['contract']['recipes'][0])
    with jax.default_device(jax.devices('cpu')[0]):template=dict(parameters=q,optimizer=tx.init(q),best=q,best_step=jnp.asarray(-1,jnp.int32))
    saved=store.restore(slot,meta['contract'],template)
    if saved is None or saved[1]!=meta:raise ValueError('source restore failed')
    payload=saved[0]
    counts=[int(np.asarray(jax.device_get(v))) for path,v in jax.tree.leaves_with_path(payload['optimizer']) if getattr(path[-1],'name',getattr(path[-1],'key',None))=='count']
    if len(counts)!=6 or any(v!=pin.get('adam_step',meta['step']) for v in counts):raise ValueError('source Adam/schedule count differs')
    if int(payload['best_step'])!=pin['best_step']:raise ValueError('source best step differs')
    return payload,pin

