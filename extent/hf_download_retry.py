"""Operational HF download retries, independent of scientific resume contracts.

Install before source reconstruction. All runtime imports of hf_hub_download
then use the same policy, including Xet errors raised inside the SDK. No tokens,
URLs or exception bodies are logged. Two workers share a local cooldown/pace;
separate Kaggle VMs use jitter (they cannot share a filesystem lock).
"""
from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
from functools import wraps
import json
import os
from pathlib import Path
import random
import tempfile
import time


def retryable(exc):
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    if status is not None:
        return status in (408, 429, 500, 502, 503, 504)
    # Do not retry arbitrary RuntimeError, bad credentials or corrupt payloads.
    names = {c.__name__ for c in type(exc).__mro__}
    return bool(names & {'Timeout', 'TimeoutException', 'ReadTimeout',
                         'ConnectTimeout', 'ConnectionError', 'ConnectError',
                         'ReadError', 'RemoteProtocolError'})


def retry_after(exc, now):
    headers = getattr(getattr(exc, 'response', None), 'headers', {}) or {}
    value = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
    if value is None:
        return None
    try:
        return max(0., float(value))
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0., date.timestamp() - now)
        except (ValueError, TypeError, OverflowError):
            return None


class DownloadPolicy:
    def __init__(self, directory, *, deadline=None, interval=2., clock=time.time,
                 sleep=time.sleep, uniform=random.uniform, log=print):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'download-pacing.json'
        from filelock import FileLock
        self.lock = FileLock(str(self.path) + '.lock', timeout=10)
        self.deadline = deadline if deadline is not None else clock() + 11.2 * 3600
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.uniform, self.log = uniform, log

    def update(self, cooldown=None, reserve=False):
        # No sleep/download under lock; another worker can publish a cooldown.
        with self.lock:
            state = json.loads(self.path.read_text()) if self.path.exists() else {}
            now = self.clock()
            if cooldown is not None:
                state['cooldown'] = max(state.get('cooldown', 0), cooldown)
            ready = max(now, state.get('cooldown', 0), state.get('next', 0))
            granted = reserve and ready <= now
            if granted:
                state['next'] = now + self.interval
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state))
            temporary.replace(self.path)
            return (ready, granted) if reserve else ready

    def pause(self, until, deadline=None):
        deadline = self.deadline if deadline is None else deadline
        while True:
            now = self.clock()
            if now >= until:
                break
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError('HF download wait reached session work deadline')
            self.sleep(min(30., until - now, remaining))
        if self.clock() >= deadline:
            raise TimeoutError('HF download wait reached session work deadline')

    def call(self, download, *args, _retry_deadline=None, **kwargs):
        deadline = self.deadline if _retry_deadline is None else _retry_deadline
        failures = 0
        while True:
            if self.clock() >= deadline:
                raise TimeoutError('HF download reached session work deadline')
            ready, granted = self.update(reserve=True)
            if not granted:
                self.pause(ready, deadline)
                continue  # Recheck: another worker may have extended cooldown.
            try:
                return download(*args, **kwargs)
            except Exception as exc:
                if not retryable(exc):
                    raise
                failures += 1
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                advised = retry_after(exc, self.clock())
                base = advised if advised is not None else (300. if status == 429 else min(120., 5. * 2 ** min(failures - 1, 5)))
                delay = base + self.uniform(1., 60. if status == 429 else 10.)
                until = self.clock() + delay
                self.update(cooldown=until)
                self.log(f'hf_download_retry status={status or "network"} attempt={failures} wait_seconds={delay:.1f}', flush=True)
                self.pause(until, deadline)


_policy = None


def install_download_retries():
    """Idempotent SDK entry-point patch; does not alter downloaded bytes/math.

    Source helpers import this SDK symbol inside their functions. This covers
    artifact chunks, legacy JSON/binary checkpoints, Qwen and data downloads.
    The SDK's own per-file/Xet retries finish before this outer retry executes.
    """
    import huggingface_hub
    global _policy
    if _policy is None:
        directory = os.environ.get('EXTENT_HF_RETRY_STATE_DIR',
                                   str(Path(tempfile.gettempdir()) / 'extent-hf-download'))
        deadline = os.environ.get('EXTENT_HF_DOWNLOAD_DEADLINE_UNIX')
        _policy = DownloadPolicy(directory, deadline=float(deadline) if deadline else None)
    elif os.environ.get('EXTENT_HF_DOWNLOAD_DEADLINE_UNIX'):
        # A reused notebook can start a fresh invocation with a new work budget.
        _policy.deadline = float(os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX'])
    original = huggingface_hub.hf_hub_download
    if getattr(original, '_extent_download_retry', False):
        return

    @wraps(original)
    def download(*args, **kwargs):
        # RecoveryCheckpointStore reads its just-uploaded manifest/sample into
        # remote-verification/ during the SAVE reserve. The source/work deadline
        # must not block that read. Only this existing verification staging path
        # gets the separate save budget; all source/training downloads keep theirs.
        verification = 'remote-verification' in Path(kwargs.get('local_dir') or '').parts
        deadline = None
        if verification:
            deadline = float(os.environ.get('EXTENT_HF_SAVE_DEADLINE_UNIX', _policy.deadline + 29 * 60))
        return _policy.call(original, *args, _retry_deadline=deadline, **kwargs)
    download._extent_download_retry = True
    huggingface_hub.hf_hub_download = download
