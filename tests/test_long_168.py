import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from extent.broad_corpus import build_corpus
from scripts.prepare_extended_edu import verify_prefix
from extent.fast_pause import install_fast_pause
from extent.state_cleanup_168 import cleanup,PREFIXES


def test_extended_pack_keeps_truncated_document_tail_and_validation(tmp_path):
    rows=[dict(split='validation',index=0,text='v'*250,url=''),dict(split='train',index=0,text='a'*250,url=''),
          dict(split='train',index=1,text='b'*250,url='')]
    base=dict(sequence_length=4,train_tokens=8,validation_tokens=4,vocab_size=20,eos_token_id=19)
    build_corpus(tmp_path/'old',rows,lambda t:list(range(12)),base,min_free_bytes=0)
    build_corpus(tmp_path/'new',rows,lambda t:list(range(12)),dict(base,train_tokens=20),min_free_bytes=0)
    proof=verify_prefix(tmp_path/'old',tmp_path/'new')
    assert proof['verified'] and proof['old_train_tokens']==8 and proof['new_train_tokens']==20
    # The newly available positions include the end of the old truncated doc.
    tokens=np.fromfile(tmp_path/'new/train.bin',dtype='<i4')
    np.testing.assert_array_equal(tokens[8:12],np.arange(8,12))


def test_fast_pause_requests_one_local_save_and_never_syncs(tmp_path):
    calls=[]
    s=SimpleNamespace(result={'cost':{}},invocation={},check=lambda:None,
        persist=lambda **kw:calls.append(('persist',kw)),checkpoint=lambda *a:calls.append(('save',a)))
    install_fast_pause(s,SimpleNamespace(work_root=tmp_path))
    (tmp_path/'FAST_STOP').touch()
    with pytest.raises(TimeoutError,match='fast local'):s.check()
    s.checkpoint('FULL-KD',{},1);s.persist(force=True)
    assert calls[0][0]=='save' and calls[1][1]==dict(force=False,local_only=True,checkpoints=False)
    assert s.result['fast_pause_requested']


def test_cleanup_requires_matching_remote_manifest_and_known_namespace(tmp_path):
    states=tmp_path/'states';meta=dict(step=10,generation='step-10',leaves=[])
    for name in ('runtime-long','runtime-lr-x1','unrelated'):
        directory=states/name/'FULL-KD';directory.mkdir(parents=True)
        (directory/'checkpoint.json').write_text(json.dumps(meta));(directory/'weights').write_bytes(b'test')
    def restore(path,remote,hub):
        value=meta if 'exp168' in remote else dict(meta,step=9)
        path.write_text(json.dumps(value));return True
    rows=cleanup(tmp_path,None,restore=restore)
    assert not (states/'runtime-long/FULL-KD').exists()
    assert (states/'runtime-lr-x1/FULL-KD/weights').exists() and (states/'unrelated/FULL-KD/weights').exists()
    assert sum(row['deleted'] for row in rows)==1


def test_cleanup_unavailable_hf_retains_local(tmp_path):
    directory=tmp_path/'states/runtime-long/FULL-KD';directory.mkdir(parents=True)
    (directory/'checkpoint.json').write_text(json.dumps(dict(step=1)))
    rows=cleanup(tmp_path,None,restore=lambda *a:False)
    assert directory.exists() and not rows[0]['deleted']


def test_168_contract_and_source_are_separate_from167(tmp_path,monkeypatch):
    from tests.test_full_stream_166 import small_contract
    from extent import full_long_168 as model
    import scripts.prepare_extended_edu
    plan,_=small_contract(tmp_path,monkeypatch)
    parent=json.loads(plan.read_text());parent['lr_multiplier']=1
    parent['full_source'].update(step=185241,windows=237441,adam_count=30389)
    plan.write_text(json.dumps(parent))
    monkeypatch.setattr(scripts.prepare_extended_edu,'verify_prefix',lambda *a:dict(verified=True))
    c=model.contract(168,None,156250,'c4','server-balanced',plan,'edu')
    assert c['long_budget_input_tokens']==320000000 and c['batch']==8
    assert c['recipes'][0]['mamba_lr']==1e-5 and c['full_source']['adam_count']==30389
    assert c['execution_namespace']=='experiments/exp168-full-long-v1'


def test_fast_control_waits_for_process_exit_before_ack(tmp_path,monkeypatch):
    import engine_control
    import scripts.qwen_extended_horizon_campaign as notices
    from filelock import FileLock
    (tmp_path/'results').mkdir();(tmp_path/'results/runtime-long.json').write_text('{}')
    sent=[];monkeypatch.setattr(notices,'_safe_notify',lambda *a:sent.append(a))
    def sleep(seconds):
        (tmp_path/'fast-pause-ready.json').write_text(json.dumps(dict(pid=123,step=100)))
    monkeypatch.setattr(engine_control.time,'sleep',sleep)
    checks=[]
    def alive(pid):checks.append(pid);return len(checks)<2
    monkeypatch.setattr(engine_control,'alive',alive)
    with FileLock(str(tmp_path/'server.lock')):
        result=engine_control.main(['--fast','--work-dir',str(tmp_path),'--wait-seconds','10'])
    assert result['step']==100 and len(checks)==2 and len(sent)==1


def test_168_freezes_completed_lr1_current(tmp_path):
    from extent import full_long_168 as model
    directory=tmp_path/'results';directory.mkdir()
    c=dict(protocol='exp167-lr-sweep-v1',code={},parent=dict(branches={'FULL-KD':dict(step=154852,manifest_sha='base',adam_count=36548)}))
    pointer=tmp_path/'states/runtime-lr-x1/FULL-KD/checkpoint.json';pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps(dict(step=185241,contract=dict(c,arm='FULL-KD'))))
    (directory/'runtime-lr-x1.json').write_text(json.dumps(dict(contract=c,status='completed',runtime={},corpus_identity='c4',pending_slots=[],finalization_status='completed',branches={'FULL-KD':dict(step=185241,durable_step=185241,training_windows_seen=237441,final=dict(last=dict(panels={'D':dict(nll=5.118,windows=[5.118])},fresh_nll={} )))})))
    _,plan=model.freeze_parent(tmp_path)
    assert plan['full_source']['step']==185241 and plan['full_source']['adam_count']==30389
    assert plan['full_source']['source_namespace']=='runtime-lr-x1'
