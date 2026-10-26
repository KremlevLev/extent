import json
import threading
import time
from urllib import error

import pytest

import engine_status as status
from tests.test_engine_dashboard import fixture


def message(text='/check',chat='123',kind='private'):
    return dict(chat=dict(id=chat,type=kind),text=text,date=time.time())


def test_summary_preserves_current_best_token_clock_and_panel_steps():
    report=fixture()
    result=status.format_report('blackwell',report,1000,True,now=2200)
    assert '0.003 млн' in result  # two six-window updates, not 120183*256
    assert 'NLL wiki_A: 4.9000 (-0.1000) @ 120183' in result
    assert 'NLL c4_D: 6.9000 (-0.1000) @ 120183' in result
    assert '1.0000' not in result  # BEST endpoint is not CURRENT
    assert 'старше 15 мин' in result and 'держит lock' in result
    assert '1707 ток/с' in result  # 1536/.9; initial audit timing excluded
    assert '\n' in result and '\\n' not in result


def test_private_chat_authorization_and_two_session_selection():
    class Reports:
        def __init__(self):self.called=[]
        def get(self,row):
            self.called.append(row['id']);return fixture(),time.time(),None,False
    reports=Reports();cmd=status.Commands([dict(id='blackwell'),dict(id='tpu')],'123',reports)
    assert cmd.reply(message(chat='456'))==[]
    assert cmd.reply(message(kind='group'))==[]
    assert reports.called==[]
    assert len(cmd.reply(message()))==2 and reports.called==['blackwell','tpu']
    assert len(cmd.reply(message('/check TPU')))==1 and reports.called[-1]=='tpu'
    assert 'Нет такой' in cmd.reply(message('/check nope'))[0]
    assert '/check blackwell' in cmd.reply(message('/sessions'))[0]
    assert cmd.reply(message('/stop'))==[]  # cannot control training


def test_local_read_missing_report_and_remote_stale_fallback(tmp_path,monkeypatch):
    root=tmp_path/'work';root.mkdir();path=root/'report.json';path.write_text(json.dumps(fixture()))
    monkeypatch.setattr(status,'trainer_active',lambda root:True)
    reports=status.Reports()
    result=reports.get(dict(work_dir=str(root),report='report.json'))
    assert result[2:] == (True,False)
    with pytest.raises(ValueError):reports.get(dict(work_dir=str(root),report='absent.json'))
    row=dict(id='tpu',repo_id='owner/repo',hf_file='experiments/report.json')
    reports.cache[json.dumps(row,sort_keys=True)]=(time.time()-120,fixture(),1000)
    def unavailable(*args,**kwargs):raise error.URLError('SECRET must never escape')
    monkeypatch.setattr(status.request,'urlopen',unavailable)
    cached=reports.get(row)
    assert cached[2:] == (None,True)
    assert 'предыдущий снимок' in status.format_report('tpu',*cached)


def test_network_error_redacts_token_and_honors_429(monkeypatch):
    import io
    def limited(*args,**kw):
        raise error.HTTPError('https://tokenSECRET',429,'SECRET',{},io.BytesIO(b'{"parameters":{"retry_after":31}}'))
    monkeypatch.setattr(status.request,'urlopen',limited)
    with pytest.raises(status.TransportError) as found:status.api('tokenSECRET','getUpdates')
    assert found.value.code==429 and found.value.retry_after==31
    assert 'SECRET' not in str(found.value)


def test_poller_authorization_cursor_and_restart_without_replay(tmp_path):
    stop=threading.Event();calls=[]
    cmd=status.Commands([dict(id='blackwell')],'123')
    def call(token,method,payload=None):
        calls.append((method,payload))
        if method=='getWebhookInfo':return dict(url='')
        if method=='getUpdates':
            stop.set()
            return [dict(update_id=10,message=message('/help',chat='999')),
                    dict(update_id=11,message=message('/help'))]
        return dict(message_id=1)
    state=tmp_path/'cursor.json';status.poll('not-real',cmd,state,stop,call)
    assert len([c for c in calls if c[0]=='sendMessage'])==1
    assert json.loads(state.read_text())['offset']==12
    calls.clear();stop.clear()
    def restart(token,method,payload=None):
        if method=='getWebhookInfo':return dict(url='')
        assert payload['offset']==12
        stop.set();return []
    status.poll('not-real',cmd,state,stop,restart)


def test_webhook_not_deleted_conflicting_listener_fatal(tmp_path):
    stop=threading.Event();cmd=status.Commands([],'123');called=[]
    def webhook(token,method,payload=None):called.append(method);return dict(url='https://existing')
    with pytest.raises(RuntimeError,match='already has a webhook'):
        status.poll('not-real',cmd,tmp_path/'cursor.json',stop,webhook)
    assert called==['getWebhookInfo']
    def conflict(token,method,payload=None):
        if method=='getWebhookInfo':return dict(url='')
        raise status.TransportError(409)
    with pytest.raises(RuntimeError,match='409'):
        status.poll('not-real',cmd,tmp_path/'cursor.json',stop,conflict)


def test_registry_validation_and_message_limit(tmp_path):
    path=tmp_path/'sessions.json'
    path.write_text(json.dumps(dict(sessions=[dict(id='one',work_dir='/tmp/a'),dict(id='ONE',work_dir='/tmp/b')])))
    with pytest.raises(ValueError):status.sessions(path,tmp_path)
    assert max(map(len,status.chunks('a'*9000)))<=3500
    assert status.sessions(None,tmp_path)[0]['id']=='blackwell'


def test_comparison_reports_edu_russian_and_correct_selection_panel():
    report=fixture();row=report['branches']['KD-C4-FRESH'];row['best_panel']='D'
    row['final']['last']['panels']['E']={'nll':5.2}
    row['final']['last']['fresh_nll']['russian_wikipedia']={'nll':7.7,'windows':[7.7]*128}
    rendered=status.format_report('compare',report,1000,True,now=1100)
    assert 'BEST C4 D:' in rendered and 'BEST Wiki A:' not in rendered
    assert 'NLL edu_E: 5.2000' in rendered
    assert 'russian_wikipedia_128windows_nll: 7.7000' in rendered


def test_bootstrap_rate_limit_retries_and_old_commands_are_not_sent(tmp_path):
    class Stop:
        done=False
        delays=[]
        def is_set(self):return self.done
        def wait(self,delay):self.delays.append(delay)
    stop=Stop();calls=[]
    def call(token,method,payload=None):
        calls.append(method)
        if len(calls)==1:raise status.TransportError(429,31)
        if method=='getWebhookInfo':return dict(url='')
        if method=='getUpdates':
            stop.done=True
            old=message('/help');old['date']=time.time()-600
            return [dict(update_id=12,message=old)]
        raise AssertionError('old command must not be delivered')
    status.poll('not-real',status.Commands([],'123'),tmp_path/'cursor.json',stop,call)
    assert 31<=stop.delays[0]<32
    assert calls==['getWebhookInfo','getWebhookInfo','getUpdates']
    assert json.loads((tmp_path/'cursor.json').read_text())['offset']==13
