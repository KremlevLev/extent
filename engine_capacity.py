"""Bounded, read-only batch sizing on the current engine state."""
import argparse
import json
import math
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import time


def select_best(report):
    valid=[r for r in report.get('trials',[]) if r.get('status')=='measured' and r.get('finite')]
    if not valid:return None
    best=max(v['input_tokens_per_second'] for v in valid)
    return min(v['batch'] for v in valid if v['input_tokens_per_second']>=.97*best)


def stop_probe(process):
    if os.name=='posix':
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
    else:process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name=='posix':
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        else:process.kill()
        process.wait(timeout=5)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default=str(Path(__file__).resolve().parent.parent/'engine-core'))
    p.add_argument('--max-minutes',type=float,default=5.)
    p.add_argument('--batches',default='1,2,4,8,16');p.add_argument('--gpu',default='0')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args(argv)
    if not math.isfinite(a.max_minutes) or a.max_minutes<=0:raise ValueError('positive finite minute budget required')
    batches=[int(b) for b in a.batches.split(',')]
    if not batches or sorted(set(batches))!=batches or any(b<1 or b>16 for b in batches):
        raise ValueError('use ascending unique integer batch sizes between1 and16')
    root=Path(a.work_dir).resolve()
    if not (root/'results/runtime-report.json').is_file():raise FileNotFoundError('current runtime-report.json required')
    if shutil.disk_usage(root).free<20*1e9:raise OSError('keep20GB free disk reserve before profiling')
    from filelock import FileLock,Timeout
    lock=FileLock(str(root/'server.lock'),timeout=0)
    try:lock.acquire()
    except Timeout:raise RuntimeError('engine worker is still running; request STOP and wait for its exit before probing') from None
    path=root/'results/runtime-capacity.json';log=root/'runtime-capacity.log'
    # Preserve previous report, rather than interpreting stale results as this run.
    if path.exists():path.replace(path.with_suffix('.previous.json'))
    begun=time.time();deadline=begun+a.max_minutes*60;process=None;timed_out=False
    try:
        with log.open('w',encoding='utf-8') as out:
            process=subprocess.Popen([sys.executable,'-u','-m','scripts.engine_probe_worker',
                '--work-dir',str(root),'--deadline',str(deadline),'--batches',a.batches,'--gpu',a.gpu],
                stdout=out,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
            try:process.wait(timeout=max(0,deadline-time.time()))
            except subprocess.TimeoutExpired:
                timed_out=True
                stop_probe(process)
        value=json.loads(path.read_text()) if path.exists() else dict(trials=[])
        value.update(recommended_batch=select_best(value),elapsed_seconds=time.time()-begun,
            budget_seconds=a.max_minutes*60,worker_returncode=process.returncode,
            stop_reason='time_budget' if timed_out else 'worker_error' if process.returncode else 'completed',
            accepted_updates=0,applied_to_training=False,
            limitation='recommendation only; a larger effective batch requires a new explicit training contract',
            log=str(log))
        value['status']='completed' if process.returncode==0 and value.get('status')=='completed' else 'partial' if value['recommended_batch'] is not None else 'no_completed_measurement'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
        if a.telegram:
            from extent.notifications import send_telegram_message,TelegramNotifierError
            try:
                send_telegram_message(f"Engine capacity: {value['status']}; batch={value['recommended_batch']}; {value['elapsed_seconds']:.0f}s; report={path}")
            except TelegramNotifierError as exc:print('Capacity notification unavailable:',exc,flush=True)
        print(json.dumps(value,indent=2))
        return value
    finally:
        if process is not None and process.poll() is None:
            stop_probe(process)
        lock.release()


if __name__=='__main__':main()
