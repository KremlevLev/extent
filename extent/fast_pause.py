"""EXP168 local-only pause, without final diagnostics or remote checkpoint sync."""
import time
from pathlib import Path


def install_fast_pause(session,args):
    original=session.check;persist=session.persist;checkpoint=session.checkpoint
    session.result['phase_seconds']={}
    session.invocation['initial_training_seconds']=session.result.get('cost',{}).get('training_seconds',0)
    def requested():return (Path(args.work_root)/'FAST_STOP').exists()
    def check():
        if requested() and not session.result.get('fast_pause_requested'):
            session.result['fast_pause_requested']=True
            raise TimeoutError('requested fast local pause')
        original()
    def write(*a,**kw):
        if session.result.get('fast_pause_requested'):
            kw.update(local_only=True,force=False,checkpoints=False)
        started=time.monotonic()
        result=persist(*a,**kw)
        if not kw.get('local_only',False):
            phases=session.result['phase_seconds'];phases['report_and_remote_sync']=phases.get('report_and_remote_sync',0)+time.monotonic()-started
        return result
    def save(name,payload,step):
        started=time.monotonic()
        result=checkpoint(name,payload,step)
        session.result.setdefault('phase_seconds',{}).setdefault('local_checkpoint',0.)
        session.result['phase_seconds']['local_checkpoint']+=time.monotonic()-started
        return result
    session.check=check;session.persist=write;session.checkpoint=save
