"""Explicit, narrow migration of EXP114's first traced-scan parity failure.

No source model, FIT optimizer, data, selection or tolerance change is allowed.
The repaired trace has no accepted old measurements in the eligible result.
"""
from pathlib import Path
import copy
import json

ROOT = Path(__file__).resolve().parents[1]
CHANGED = {'extent/plateau_forensics.py', 'scripts/m3q_plateau_forensics_114_campaign.py',
           'extent/plateau_trace_repair.py'}

def migrate_trace_failure(result, current, enabled):
    if not enabled:
        raise ValueError('EXP114 resume contract mismatch; trace repair requires --repair-trace')
    original = json.loads((ROOT/'results/EXP-114-trace-repair-source-contract.json').read_text())
    if result['contract'] != original:
        raise ValueError('trace repair source contract is not the registered original')
    outside = {k:v for k,v in current.items() if k != 'code'}
    if outside != {k:v for k,v in original.items() if k != 'code'}:
        raise ValueError('trace repair cannot change source/data/protocol/tolerances')
    changed = {k for k in set(original['code']) | set(current['code'])
               if original['code'].get(k) != current['code'].get(k)}
    if changed != CHANGED:
        raise ValueError('trace repair includes unapproved dependency changes')
    repair_code = json.loads((ROOT/'results/EXP-114-trace-repair-code.json').read_text())
    if current['code'] != repair_code:
        raise ValueError('trace repair implementation differs from audited repair')
    if result.get('status') != 'failed' or not result.get('error','').startswith('traced scan output parity:'):
        raise ValueError('trace repair only accepts the recorded parity failure')
    if result.get('pending_slots'):
        raise ValueError('trace repair requires durable readout checkpoints')
    for module in ('profile','sweep','fit','temperature','readout','optimizer'):
        if result['modules'].get(module) != 'completed':
            raise ValueError('trace repair requires the completed pre-dynamics modules')
    forbidden = ('dynamics/','context/','reset/','precision/','locked/')
    if any(k.startswith(forbidden) or '/C_wiki/' in k or '/C_pg19/' in k for k in result['tasks']):
        raise ValueError('trace repair cannot reuse affected traces or viewed locked panels')
    if any(result['readout'].get(str(s),{}).get('durable_step') != 4096 for s in (123,456)):
        raise ValueError('trace repair needs both durable 4096-step readouts')
    result.setdefault('repair_history',[]).append(dict(
        original_contract=copy.deepcopy(original), repaired_contract=copy.deepcopy(current),
        error=result['error'], traceback=result.get('traceback'),
        reason='Restore official read+skip before gate/output multiplication; threshold remains1e-4',
        retained_modules=['profile','sweep','fit','temperature','readout','optimizer'],
        discarded_measurements=0))
    result['readout_checkpoint_contract'] = copy.deepcopy(original)
    result['contract'] = copy.deepcopy(current)
    result['modules'].pop('dynamics',None)
    result.pop('error_type',None)
    result.pop('traceback',None)
