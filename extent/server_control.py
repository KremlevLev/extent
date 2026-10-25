"""Local stop requests and storage budget; never kill unrelated GPU processes."""
import os
from pathlib import Path
import shutil
import time


def storage_status(root,max_local_gb=80.,min_free_gb=20.):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    free=shutil.disk_usage(root).free
    if used>max_local_gb*1e9 or free<min_free_gb*1e9:
        raise OSError(f'Work storage budget reached: used={used/1e9:.2f}GB, free={free/1e9:.2f}GB')
    return dict(work_bytes=used,free_bytes=free,max_local_gb=max_local_gb,min_free_gb=min_free_gb)


def install_session_control(session,args):
    original=session.check;session.finalizing=False;last=[0.]
    def check():
        if not session.finalizing:
            reason=None
            if args.stop_file and Path(args.stop_file).exists():reason='requested soft stop'
            if args.work_root and time.monotonic()-last[0]>=60:
                try:session.result['storage_monitor']=storage_status(args.work_root,args.max_local_gb,args.min_free_gb)
                except OSError as exc:reason=str(exc)
                last[0]=time.monotonic()
            if reason:
                session.wall=min(session.wall,time.monotonic()+args.save_reserve_minutes*60)
                os.environ['EXTENT_HF_SAVE_DEADLINE_UNIX']=str(time.time()+max(0,session.wall-time.monotonic()-60))
                raise TimeoutError(reason)
        original()
    session.check=check
    # Keep one atomic current generation, without writing a GB every fast block.
    if hasattr(session,'checkpoint'):
        checkpoint=session.checkpoint;saved={}
        def bounded_checkpoint(name,payload,step):
            now=time.monotonic()
            interval=getattr(args,'local_checkpoint_minutes',15.)*60
            if not session.finalizing and name in saved and now-saved[name]<interval:return
            checkpoint(name,payload,step);saved[name]=now
        session.checkpoint=bounded_checkpoint
