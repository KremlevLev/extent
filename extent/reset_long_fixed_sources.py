"""EXP143/144 source restore with empty frozen-attention containers retained.

Chunk manifests contain tensor leaves only; frozen GQA coordinate entries have
no leaves. Recreate ONLY those empty entries before initializing the optimizer
TEMPLATE. Actual q, Adam moments/counts, best and checksums come from the original
immutable source. Existing135-146 files remain unchanged.
"""
from pathlib import Path
from dataclasses import replace
import hashlib,json
import numpy as np
import jax
import jax.numpy as jnp
from extent.competitor_recovery import optimizer
from extent.t4_patient import read_parameter_chunks
from extent.recovery_durability import RecoveryCheckpointStore
from extent.hf_artifact_sync import restore_artifact
ROOT=Path(__file__).resolve().parents[1]


def complete_coordinates(q,model):
    expected={str(i) for i in range(model['num_layers'])}
    attention={str(i) for i in model['attention_layer_indices']}
    if set(q)-expected:
        raise ValueError('unexpected source coordinate layer')
    missing=expected-set(q)
    if missing-attention:
        raise ValueError('missing trainable Mamba coordinates')
    return {str(i):q.get(str(i),{}) for i in range(model['num_layers'])}


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
    q=complete_coordinates(q,meta['contract']['model'])
    tx=optimizer(q,meta['contract']['recipes'][0])
    with jax.default_device(jax.devices('cpu')[0]):template=dict(parameters=q,optimizer=tx.init(q),best=q,best_step=jnp.asarray(-1,jnp.int32))
    saved=store.restore(slot,meta['contract'],template)
    if saved is None or saved[1]!=meta:raise ValueError('source restore failed')
    payload=saved[0]
    counts=[int(np.asarray(jax.device_get(v))) for path,v in jax.tree.leaves_with_path(payload['optimizer']) if getattr(path[-1],'name',getattr(path[-1],'key',None))=='count']
    if len(counts)!=6 or any(v!=meta['step'] for v in counts):raise ValueError('source Adam/schedule count differs')
    if int(payload['best_step'])!=pin['best_step']:raise ValueError('source best step differs')
    return payload,pin

