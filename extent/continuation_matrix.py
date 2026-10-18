"""EXP149-154: one immutable147 source, Adam and fold/rank interventions.

Old147/148 and earlier scientific source files remain unchanged.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import jax
import jax.numpy as jnp
from extent.fold_continue_recovery import base_coordinates
from extent.reset_long_recovery import fold_weights
from extent.reset_long_fixed_sources import complete_coordinates
from extent.competitor_recovery import initialize, optimizer
from extent.recovery_subspace import initialize_corrections
from extent.t4_patient import read_parameter_chunks
from extent.recovery_durability import RecoveryCheckpointStore
from extent.hf_artifact_sync import restore_artifact

ROOT = Path(__file__).resolve().parents[1]
TPU_EXPERIMENTS = (149, 150, 151, 152)
GPU_EXPERIMENTS = (153, 154)
LIMITS = {149: 9., 150: 9., 151: 9., 152: 9., 153: 12., 154: 6.}


def source_pin(key='147-original'):
    if key != '147-original':
        raise ValueError('only immutable147 original-rate source is registered')
    return json.loads((ROOT/'results/EXP-149-154-source-pin.json').read_text())[key]


def recipes(experiment):
    from extent.reset_long_recovery import recipes as old_recipes
    original = old_recipes(145)[1]
    def arm(mode, rank=32):
        return dict(original, name={'exact': 'KEEP-ADAM-R32',
            'fresh-adam': 'SAME-R32-FRESH-ADAM',
            'fold-new': f'FOLD-FRESH-R{rank}'}[mode], source='147-original',
            mode=mode, correction_rank=rank, reset=False, ramp_steps=0)
    if experiment == 149: return [arm('exact')]
    if experiment == 150: return [arm('fresh-adam')]
    if experiment == 151: return [arm('fold-new', 32)]
    if experiment == 152: return [arm('fold-new', 128)]
    if experiment == 153: return [arm('fold-new', 64), arm('fold-new', 256)]
    if experiment == 154: return [arm('exact'), arm('fold-new', 32)]
    raise ValueError('EXP149-154 only')


def adam_counts(state):
    return [int(np.asarray(jax.device_get(v))) for path,v in jax.tree.leaves_with_path(state)
        if getattr(path[-1], 'name', getattr(path[-1], 'key', None)) == 'count']


def restore_source(key, hub, cache, check=lambda: None):
    pin = source_pin(key)
    sourcehub = replace(hub, repo_id=pin['repo_id'], repo_type='dataset', revision=pin['revision'])
    store = RecoveryCheckpointStore(Path(cache)/key, pin['prefix'], sourcehub)
    slot = pin['slot']; directory = store.directory(slot); pointer = directory/'checkpoint.json'
    check()
    if not pointer.exists(): restore_artifact(pointer, pin['path'], sourcehub)
    raw = pointer.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['manifest_sha256']:
        raise ValueError('immutable147 manifest SHA differs')
    meta = json.loads(raw)
    if meta != pin['metadata']: raise ValueError('immutable147 metadata differs')
    for name in ('extent/reset_streamed_gpu.py', 'extent/reset_followup.py', 'extent/competitor_recovery.py'):
        expected = meta['contract'].get('code', {}).get(name)
        actual = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != expected: raise ValueError('source numerical code differs: '+name)
    q = complete_coordinates(read_parameter_chunks(meta, directory, pin['prefix']+'/'+slot,
        sourcehub, check), meta['contract']['model'])
    tx = optimizer(q, meta['contract']['recipes'][0])
    with jax.default_device(jax.devices('cpu')[0]):
        template = dict(parameters=q, optimizer=tx.init(q), best=q, best_step=jnp.asarray(-1, jnp.int32))
    saved = store.restore(slot, meta['contract'], template)
    if saved is None or saved[1] != meta: raise ValueError('immutable147 restore failed')
    payload = saved[0]
    if adam_counts(payload['optimizer']) != [pin['adam_step']]*6:
        raise ValueError('source Adam count must be18540, not global31126')
    if int(payload['best_step']) != pin['best_step']: raise ValueError('source selected-best cursor differs')
    return payload, pin


def prepare(patient, spec, payload, pin, ancestor=None):
    """Rebuild145 frozen base ONCE; source147 q is folded only for interventions."""
    if ancestor is None: raise ValueError('original136 coordinates required')
    from extent.fold_continue_recovery import source_pin as ancestor_pin
    old = ancestor_pin('136-all')['metadata']['contract']['recipes'][0]
    base = fold_weights(patient.base, ancestor, patient.cfg, patient.columns, old)
    q = payload['parameters']
    if spec['mode'] == 'fold-new':
        source_spec = pin['metadata']['contract']['recipes'][0]
        base = fold_weights(base, q, patient.cfg, patient.columns, source_spec)
        coordinates = initialize_corrections(base, patient.cfg.mamba_layer_indices,
            'INOUT-LORA', seed=123, rank=spec['correction_rank'], head_dim=patient.cfg.mamba.head_dim)
        p = SimpleNamespace(cfg=patient.cfg, base=base, current=base,
            coordinates=coordinates, columns=patient.columns)
        fresh = initialize(p, spec)
        for i in patient.cfg.mamba_layer_indices:
            fresh[str(i)]['pregate_weight'] = q[str(i)]['pregate_weight']
        q = fresh
    return base, q, pin['source_step']


def make_kernel(cfg, tc, columns, spec, tx, *, tpu):
    from extent.reset_followup import block_forward
    if tpu:
        return VerifiedTPUStep(cfg, tc, columns, spec, tx)
    from extent.reset_streamed_gpu import make_step
    return make_step(cfg, tc, columns, spec, tx, block_fn=block_forward)


class VerifiedTPUStep:
    """Audit first real gradient before selecting compiled or streamed execution.

    Failed compiled parity never weakens the gate: use the existing streamed
    engine instead, with unchanged objective, source, state, rates and cursor.
    Both proposals are unaccepted until the caller checks the returned health.
    """
    def __init__(self, cfg, tc, columns, spec, tx):
        from extent.reset_followup import block_forward
        from extent.competitor_tpu import make_step as compiled
        from extent.reset_streamed_gpu import make_step as streamed
        from extent.plateau_forensics import GuardedJit
        self.compiled=GuardedJit(compiled(cfg,tc,columns,spec,tx,
            block_fn=block_forward,return_gradients=True))
        self.streamed=streamed(cfg,tc,columns,spec,tx,
            block_fn=block_forward,return_gradients=True)
        self.check=lambda:None
        self.selected=None
        self.audit=None

    def __call__(self,*args):
        if self.selected is not None:
            self.streamed.check=self.check
            return (self.compiled if self.selected=='compiled' else self.streamed)(*args)
        from extent.reset_followup import compare_gradients
        self.check();self.streamed.check=self.check
        reference=self.streamed(*args)
        if not bool(reference[2]['finite']):
            self.selected='streamed'
            self.audit=dict(selected='streamed',passed=False,gradient_evaluations=1,reason='nonfinite reference; caller must reject proposal')
            return reference[:2]+(dict(reference[2],engine_gate=self.audit),)+reference[3:]
        try:
            actual=self.compiled(*args)
            comparison=compare_gradients(reference[3],actual[3])
            ce_error=abs(float(reference[2]['ce'])-float(actual[2]['ce']))
            passed=bool(comparison['passed'] and ce_error<=.03 and bool(actual[2]['finite']))
            self.audit=dict(**comparison,ce_error=ce_error,ce_tolerance=.03,
                gradient_evaluations=2,
                compiled_accepted=passed,selected='compiled' if passed else 'streamed',
                scope='first real unaccepted proposal per invocation; audit tokens/time extra, not an accepted update')
            self.selected=self.audit['selected']
            result=actual if passed else reference
        except MemoryError:
            self.selected='streamed';result=reference
            self.audit=dict(compiled_accepted=False,selected='streamed',gradient_evaluations=1,reason='compiled HBM guard',
                scope='reference finite; compiled proposal not executed/accepted')
        if self.selected=='streamed':self.compiled=None
        return result[:2]+(dict(result[2],engine_gate=self.audit),)+result[3:]
