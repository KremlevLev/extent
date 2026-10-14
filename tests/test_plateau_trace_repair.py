"""The repair retains only unaffected probes under explicitly pinned contracts."""
import copy
import json
from pathlib import Path
import pytest
from extent.config import load_config
from extent.plateau_trace_repair import migrate_trace_failure
from scripts.m3q_plateau_forensics_114_campaign import contract_for

ROOT = Path(__file__).resolve().parents[1]

def fixture():
    old=json.loads((ROOT/'results/EXP-114-trace-repair-source-contract.json').read_text())
    cfg,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    current=contract_for(cfg)
    result=dict(contract=old,status='failed',error='traced scan output parity: 2.0',
        traceback='original error',pending_slots=[],
        modules={m:'completed' for m in ('profile','sweep','fit','temperature','readout','optimizer')},
        tasks={'baseline/123/A/0':dict(value=7.3)},
        readout={s:dict(step=4096,durable_step=4096) for s in ('123','456')})
    return result,current

def test_explicit_repair_keeps_probe_values_and_original_checkpoint_contract():
    result,current=fixture();old=copy.deepcopy(result['contract']);tasks=copy.deepcopy(result['tasks'])
    migrate_trace_failure(result,current,True)
    assert result['contract']==current and result['tasks']==tasks
    assert result['readout_checkpoint_contract']==old
    assert result['repair_history'][0]['error']=='traced scan output parity: 2.0'
    assert result['contract']['tolerance']==old['tolerance']==1e-4

@pytest.mark.parametrize('change',['disabled','source','code','trace','locked','pending','incomplete','durable'])
def test_unsafe_migration_is_rejected(change):
    result,current=fixture()
    if change=='source':current['source_revision']='other'
    if change=='code':current['code']['extent/plateau_forensics.py']='other'
    if change=='trace':result['tasks']['dynamics/123/layer0/0']=dict(value={})
    if change=='locked':result['tasks']['baseline/123/C_wiki/0']=dict(value=7)
    if change=='pending':result['pending_slots']=['123/READOUT']
    if change=='incomplete':result['modules']['fit']='running'
    if change=='durable':result['readout']['123']['durable_step']=1024
    before=copy.deepcopy(result)
    with pytest.raises(ValueError):migrate_trace_failure(result,current,change!='disabled')
    assert result==before
