from datetime import datetime, timezone
from email.utils import format_datetime
import hashlib
import json
from types import SimpleNamespace

import pytest

from extent import hf_download_retry as retry


class Clock:
    now = 1000.
    def __init__(self):
        self.waits = []
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class HttpError(Exception):
    def __init__(self, status, headers=None):
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


def policy(tmp_path, clock, **kwargs):
    return retry.DownloadPolicy(tmp_path, clock=clock, sleep=clock.sleep,
                                uniform=lambda a, b: a, log=lambda *a, **k: None,
                                **kwargs)


def test_many_xet_429_failures_then_success_preserves_arguments(tmp_path):
    clock = Clock(); p = policy(tmp_path, clock, deadline=10000)
    attempts = []
    def download(*args, **kwargs):
        attempts.append((args, kwargs))
        if len(attempts) <= 8:
            raise HttpError(429)
        return 'downloaded.npy'
    assert p.call(download, 'repo', revision='immutable', token='secret') == 'downloaded.npy'
    assert len(attempts) == 9 and all(a == attempts[0] for a in attempts)
    assert clock.now == 1000 + 8 * 301
    assert max(clock.waits) <= 30


@pytest.mark.parametrize('header', ['125', format_datetime(datetime.fromtimestamp(1125, timezone.utc))])
def test_honors_numeric_and_http_date_retry_after(tmp_path, header):
    clock = Clock(); p = policy(tmp_path, clock)
    attempts = []
    def download():
        attempts.append(clock())
        if len(attempts) == 1:
            raise HttpError(429, {'Retry-After': header})
        return 42
    assert p.call(download) == 42
    assert attempts == [1000, 1126]


@pytest.mark.parametrize('error', [HttpError(401), HttpError(403), HttpError(404),
                                  ValueError('SHA mismatch'), RuntimeError('bad model')])
def test_permanent_failures_do_not_retry(tmp_path, error):
    clock = Clock(); p = policy(tmp_path, clock)
    def download():
        raise error
    with pytest.raises(type(error)) as caught:
        p.call(download)
    assert caught.value is error and clock.waits == []


def test_retry_wait_stops_at_deadline_instead_of_eating_save_reserve(tmp_path):
    clock = Clock(); p = policy(tmp_path, clock, deadline=1060)
    with pytest.raises(TimeoutError, match='deadline'):
        p.call(lambda: (_ for _ in ()).throw(HttpError(429)))
    assert clock.now == 1060 and max(clock.waits) <= 30


def test_two_independent_workers_share_cooldown_and_request_pacing(tmp_path):
    clock = Clock(); a = policy(tmp_path, clock); b = policy(tmp_path, clock)
    a.update(cooldown=1100)
    observed = []
    assert b.call(lambda: observed.append(clock())) is None
    assert a.call(lambda: observed.append(clock())) is None
    assert observed == [1100, 1102]


def test_preempted_worker_must_reserve_slot_after_waiting(tmp_path):
    clock = Clock(); a = policy(tmp_path, clock); b = policy(tmp_path, clock)
    a.call(lambda: None)  # Slot at1000; next slot at1002.
    original = b.update
    def preempted(**kwargs):
        result = original(**kwargs)
        if kwargs.get('reserve'):
            clock.now = max(clock.now, result[0])
        return result
    b.update = preempted
    observed = []
    b.call(lambda: observed.append(clock()))
    a.call(lambda: observed.append(clock()))
    assert observed == [1002, 1004]


def test_transient_503_and_network_timeout_are_retried(tmp_path):
    clock = Clock(); p = policy(tmp_path, clock)
    class ReadTimeout(Exception): pass
    failures = [HttpError(503), ReadTimeout(), None]
    def download():
        item = failures.pop(0)
        if item: raise item
        return 1
    assert p.call(download) == 1 and failures == []


def test_sdk_install_covers_atomic_chunks_and_legacy_checkpoint_paths(tmp_path, monkeypatch):
    import huggingface_hub
    from extent.hf_artifact_sync import restore_artifact, HubArtifactConfig
    from extent.hf_checkpoint_sync import download_checkpoint_from_hub
    clock = Clock(); p = policy(tmp_path / 'coord', clock)
    monkeypatch.setattr(retry, '_policy', p)
    monkeypatch.delenv('EXTENT_HF_DOWNLOAD_DEADLINE_UNIX', raising=False)
    data = b'valid payload'; calls = {}; arguments = []
    meta = dict(checkpoint_file='state.bin', checkpoint_bytes=len(data),
                checkpoint_sha256=hashlib.sha256(data).hexdigest(), step=7)
    def download(**kwargs):
        name = kwargs['filename']; arguments.append(kwargs)
        calls[name] = calls.get(name, 0) + 1
        if calls[name] == 1: raise HttpError(429)
        path = tmp_path / ('remote-' + name.split('/')[-1])
        path.write_bytes(json.dumps(meta).encode() if name.endswith('checkpoint.json') else data)
        return str(path)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    retry.install_download_retries()
    wrapped = huggingface_hub.hf_hub_download
    retry.install_download_retries()
    assert wrapped is huggingface_hub.hf_hub_download
    target = tmp_path / 'chunk.npy'
    assert restore_artifact(target, 'chunks/a.npy', HubArtifactConfig('r', 'dataset', 'token', 'pin'))
    assert target.read_bytes() == data
    assert download_checkpoint_from_hub(tmp_path / 'legacy', repo_id='r',
        path_in_repo='old', token='token', revision='pin', repo_type='dataset') == meta
    assert (tmp_path / 'legacy/state.bin').read_bytes() == data
    assert set(calls.values()) == {2}
    assert all(a['revision'] == 'pin' and a['token'] == 'token' for a in arguments)


def test_retry_logs_do_not_expose_exception_token_or_url(tmp_path, capsys):
    clock = Clock()
    p = retry.DownloadPolicy(tmp_path, clock=clock, sleep=clock.sleep, uniform=lambda a,b:a)
    calls = []
    def download():
        calls.append(1)
        if len(calls) == 1:
            error = HttpError(429); error.args = ('https://secret?token=private',)
            raise error
    p.call(download)
    text = capsys.readouterr().out
    assert 'status=429' in text and 'private' not in text and 'https' not in text


def test_real_subprocesses_share_rate_limit_state(tmp_path):
    import subprocess
    import sys
    script = '''
import json, sys, time
from pathlib import Path
from types import SimpleNamespace
from extent.hf_download_retry import DownloadPolicy
root=Path(sys.argv[1]); worker=sys.argv[2]; calls=[]
p=DownloadPolicy(root/'shared', deadline=time.time()+5, interval=.04,
    uniform=lambda a,b:.01, log=lambda *a,**k:None)
class Limited(Exception):
    response=SimpleNamespace(status_code=429,headers={'Retry-After':'.1'})
def download():
    calls.append(time.time())
    if len(calls)==1:raise Limited()
    return 'ok'
assert p.call(download)=='ok'
(root/(worker+'.json')).write_text(json.dumps(calls))
'''
    children = [subprocess.Popen([sys.executable, '-c', script, str(tmp_path), str(i)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(2)]
    for child in children:
        stdout, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, (stdout, stderr)
    rows = [json.loads((tmp_path / f'{i}.json').read_text()) for i in range(2)]
    assert all(len(row) == 2 and row[1] - row[0] >= .1 for row in rows)
    calls = sorted(t for row in rows for t in row)
    assert all(b-a >= .025 for a,b in zip(calls,calls[1:]))
