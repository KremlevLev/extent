import json,os,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from extent import status_autostart as auto
import engine_status


def test_registry_merges_sessions_and_switches_same_work_report(tmp_path):
    path=tmp_path/'sessions.json'
    auto.register(path,tmp_path/'a','results/old.json')
    auto.register(path,tmp_path/'b','results/other.json')
    rows=auto.register(path,tmp_path/'a','results/runtime-full.json')
    assert len(rows)==2 and rows[0]['report']=='results/runtime-full.json' and rows[1]['id']=='blackwell-2'
    cmd=engine_status.Commands([],123,config=str(path),root=tmp_path)
    assert 'blackwell-2' in cmd.reply(dict(chat=dict(id=123,type='private'),text='/sessions'))[0]
    rows.append(dict(id='tpu',work_dir=str(tmp_path/'tpu'),report='r.json'));path.write_text(json.dumps(dict(sessions=rows)))
    assert 'tpu' in cmd.reply(dict(chat=dict(id=123,type='private'),text='/sessions'))[0]


def test_auto_start_once_reuses_running_listener_and_never_prints_token(tmp_path,monkeypatch,capsys):
    directory=tmp_path/'service';directory.mkdir();lock=tmp_path/'listener.lock'
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN','test-secret');monkeypatch.setenv('TELEGRAM_CHAT_ID','123')
    monkeypatch.setattr(auto,'paths',lambda token:(directory,lock))
    running=[False];calls=[]
    monkeypatch.setattr(auto,'lock_free',lambda path:not running[0])
    monkeypatch.setattr(auto,'legacy_pids',lambda token:[901] if running[0] else [])
    def popen(args,**kw):
        calls.append(args);running[0]=True
        ready=Path(args[args.index('--ready-file')+1]);config=Path(args[args.index('--config')+1])
        ready.write_text(json.dumps(dict(pid=901,config=str(config))))
        return SimpleNamespace(pid=901,poll=lambda:None)
    monkeypatch.setattr(auto.subprocess,'Popen',popen)
    first=auto.ensure_listener(tmp_path/'work','results/runtime-full.json')
    second=auto.ensure_listener(tmp_path/'work','results/new.json')
    assert first['status']=='started' and second['status']=='registered' and len(calls)==1
    assert 'test-secret' not in capsys.readouterr().out
    assert json.loads((directory/'sessions.json').read_text())['sessions'][0]['report']=='results/new.json'


def test_unknown_bot_lock_never_spawns_or_kills_any_process(tmp_path,monkeypatch):
    directory=tmp_path/'service';directory.mkdir()
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN','test-secret');monkeypatch.setenv('TELEGRAM_CHAT_ID','123')
    monkeypatch.setattr(auto,'paths',lambda token:(directory,tmp_path/'listener.lock'))
    monkeypatch.setattr(auto,'lock_free',lambda path:False);monkeypatch.setattr(auto,'legacy_pids',lambda token:[])
    monkeypatch.setattr(auto.os,'kill',lambda *a:pytest.fail('must not kill unknown process'))
    monkeypatch.setattr(auto.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not create duplicate'))
    with pytest.raises(RuntimeError,match='unknown listener'):auto.ensure_listener(tmp_path/'work','report.json')


def test_adaptive_status_uses_explicit_token_clock_and_per_update_batch():
    report=dict(contract=dict(experiment=164,batch=None),status='running',pending_slots=[],branches={'FULL-KD':dict(
        step=157686,source_step=157684,accepted_input_tokens=2816,selected_batch=5,history=[dict(step=157686,seconds=1.,input_tokens=1280,accepted_input_tokens=2816)],evaluations=[])})
    text=engine_status.format_report('blackwell',report,0,True,now=0)
    assert '0.003 млн' in text and '1280 ток/с' in text and 'batch: 5' in text
