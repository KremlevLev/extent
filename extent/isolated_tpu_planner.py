"""CPU-only supervisor for disposable accelerator probes; no JAX import here.

Worker protocol: write JSON {finite, input_tokens_per_second, ...}, exit zero.
Every worker must restore the same immutable CURRENT and accept zero updates.
This module never starts training or treats an unverified probe as a fit.
"""
import json,os,signal,subprocess,time
from pathlib import Path


def terminate_group(process):
    if process.poll() is not None:return
    if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
    else:process.terminate()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
        else:process.kill()
        process.wait(timeout=10)


def run_probe(command,result_path,log_path,timeout_seconds,*,stop=lambda:False):
    """Capture Python failures, signals, missing JSON and hangs outside TPU worker."""
    result_path=Path(result_path);log_path=Path(log_path)
    result_path.unlink(missing_ok=True);log_path.parent.mkdir(parents=True,exist_ok=True)
    begun=time.monotonic()
    with log_path.open('w',encoding='utf-8') as output:
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,
            start_new_session=(os.name=='posix'))
        reason=None
        try:
            while process.poll() is None:
                if stop():reason='requested_stop';break
                if time.monotonic()-begun>=timeout_seconds:reason='probe_timeout';break
                time.sleep(.25)
        finally:
            if process.poll() is None:terminate_group(process)
    row=dict(returncode=process.returncode,seconds=time.monotonic()-begun,log=str(log_path))
    if reason:return dict(row,status=reason)
    if process.returncode!=0:return dict(row,status='worker_failed')
    try:
        report=json.loads(result_path.read_text(encoding='utf-8'))
        import math
        speed=report['input_tokens_per_second']
        if report.get('finite') is not True or not isinstance(speed,(float,int)) or not math.isfinite(speed) or speed<=0:
            return dict(row,status='invalid_measurement',report=report)
        if report.get('accepted_updates')!=0:return dict(row,status='probe_mutated_state',report=report)
        return dict(row,status='measured',report=report)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return dict(row,status='invalid_result',error=str(exc))


def grow(probe,*,first=1,max_batch=1024,max_trials=1024,deadline=None,record=lambda result:None):
    """Keep successful fits after a failed disposable probe; choose fastest fit.

    Stop at first failed batch or explicit time/count cap. A failed batch is not
    automatically an OOM: preserve its diagnostics and mark the boundary unknown.
    """
    if first<1 or max_batch<first or max_trials<1:raise ValueError('invalid probe bounds')
    trials=[];batch=first
    while batch<=max_batch and len(trials)<max_trials:
        if deadline is not None and time.monotonic()>=deadline:break
        try:row=dict(probe(batch),batch=batch)
        except Exception as exc:row=dict(batch=batch,status='supervisor_error',error_type=type(exc).__name__,error=str(exc))
        trials.append(row);record(dict(trials=trials))
        if row['status']!='measured':break
        batch+=1
    fits=[r for r in trials if r['status']=='measured']
    best=max(fits,key=lambda r:r['report']['input_tokens_per_second']) if fits else None
    result=dict(trials=trials,selected_batch=best['batch'] if best else None,
        last_successful_batch=fits[-1]['batch'] if fits else None,
        status='candidate_requires_training_preflight' if best else 'no_valid_fit',
        reason='failed_probe' if trials and trials[-1]['status']!='measured' else 'explicit_probe_cap')
    record(result);return result
