"""EXP155: unchanged trained EXP153-v2 R256, optimizer-preserving TPU continuation."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import jax
import jax.numpy as jnp
from extent.rank_growth_153 import block_forward, Inspector, replay_report
from extent.continuation_matrix import adam_counts, base_coordinates
from extent.continuation_matrix import prepare as reconstruct_base
from extent.continuation_matrix import VerifiedTPUStep
from extent.rank_growth_153 import GrowthStreamedStep
from extent.competitor_recovery import optimizer
from extent.t4_patient import read_parameter_chunks
from extent.reset_long_fixed_sources import complete_coordinates
from extent.recovery_durability import RecoveryCheckpointStore
from extent.hf_artifact_sync import restore_artifact

ROOT = Path(__file__).resolve().parents[1]
TPU_EXPERIMENTS = (155,)
GPU_EXPERIMENTS = ()
LIMITS = {155: 9.}


def source_pin():
    return json.loads((ROOT/'results/EXP-155-source-pin.json').read_text())


def recipes(experiment):
    if experiment != 155:
        raise ValueError('EXP155 only')
    source = source_pin()['metadata']['contract']['recipes'][0]
    return [dict(source, name='CONTINUE-R256-ADAM', mode='exact', source='153v2-r256-current')]


def restore_source(key, hub, cache, check=lambda: None):
    if key != '153v2-r256-current':
        raise ValueError('only pinned153v2 current R256')
    pin = source_pin()
    sourcehub = replace(hub, repo_id=pin['repo_id'], repo_type='dataset', revision=pin['revision'])
    store = RecoveryCheckpointStore(Path(cache)/key, pin['prefix'], sourcehub)
    slot = pin['slot']; directory = store.directory(slot); pointer = directory/'checkpoint.json'
    check()
    if not pointer.exists():
        restore_artifact(pointer, pin['path'], sourcehub)
    raw = pointer.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['manifest_sha256']:
        raise ValueError('immutable153v2 manifest SHA differs')
    meta = json.loads(raw)
    if meta != pin['metadata']:
        raise ValueError('immutable153v2 metadata differs')
    for name, expected in meta['contract']['code'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise ValueError('immutable153v2 numerical/source code differs: '+name)
    q = complete_coordinates(read_parameter_chunks(meta, directory, pin['prefix']+'/'+slot,
        sourcehub, check), meta['contract']['model'])
    with jax.default_device(jax.devices('cpu')[0]):
        tx = optimizer(q, meta['contract']['recipes'][0])
        template = dict(parameters=q, optimizer=tx.init(q), best=q, best_step=jnp.asarray(-1,jnp.int32))
    saved = store.restore(slot, meta['contract'], template)
    check()
    if saved is None or saved[1] != meta:
        raise ValueError('immutable153v2 restore failed')
    payload = saved[0]
    if adam_counts(payload['optimizer']) != [pin['adam_step']]*6:
        raise ValueError('source Adam count must be6784, not global37910')
    if int(payload['best_step']) != pin['best_step']:
        raise ValueError('source selected-best cursor differs')
    return payload, pin


def prepare(patient, spec, payload, pin, ancestor=None):
    # The validated136 fold reconstructs the existing base; no153 factors folded.
    return reconstruct_base(patient, dict(spec,mode='exact'), payload, pin, ancestor)


class GrowthVerifiedTPUStep(VerifiedTPUStep):
    """Existing parity/fallback policy, BOTH paths using growth-aware arithmetic."""
    def __init__(self,cfg,tc,columns,spec,tx):
        from extent.competitor_tpu import make_step as compiled
        from extent.plateau_forensics import GuardedJit
        self.compiled = GuardedJit(compiled(cfg,tc,columns,spec,tx,
            block_fn=block_forward,return_gradients=True))
        # Legacy make_step ignores block_fn: explicitly use GrowthStreamedStep.
        self.streamed = GrowthStreamedStep(cfg,tc,columns,spec,tx,return_gradients=True)
        self.check = lambda: None
        self.selected = None
        self.audit = None


def make_kernel(cfg,tc,columns,spec,tx,*,tpu):
    if not tpu:
        raise ValueError('EXP155 requires TPU v5e8')
    return GrowthVerifiedTPUStep(cfg,tc,columns,spec,tx)
