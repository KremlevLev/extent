"""Request a fast local pause or a normal HF-backed soft stop."""
import argparse,json,os,time
from pathlib import Path


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-dir',required=True)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--fast',action='store_true');group.add_argument('--soft',action='store_true')
    p.add_argument('--wait-seconds',type=float,default=300.)
    a=p.parse_args(argv);root=Path(a.work_dir).resolve()
    if not (root/'results/runtime-long.json').exists():raise ValueError('fast control requires EXP168 runtime-long report')
    from filelock import FileLock,Timeout
    try:
        with FileLock(str(root/'server.lock'),timeout=0):raise ValueError('no training process holds this work directory')
    except Timeout:pass
    marker=root/('FAST_STOP' if a.fast else 'STOP');receipt=root/'fast-pause-ready.json'
    if a.fast:receipt.unlink(missing_ok=True)
    marker.write_text('requested '+str(time.time()));print('Pause requested; waiting for local save and process exit' if a.fast else 'Soft stop requested',flush=True)
    if not a.fast:return
    end=time.monotonic()+a.wait_seconds
    while time.monotonic()<end:
        if receipt.exists():
            value=json.loads(receipt.read_text())
            if not alive(value['pid']):
                print('GPU training process exited; local CURRENT saved at step',value['step'],flush=True)
                from scripts.qwen_extended_horizon_campaign import _safe_notify
                _safe_notify(True,f"EXP168: fast pause complete; process exited; local step={value['step']}; HF sync deferred")
                return value
        time.sleep(1)
    raise TimeoutError('Pause not confirmed yet; do not assume GPU is released. Inspect runtime-long.log')


if __name__=='__main__':main()
