"""Engine core runtime worker. The wall clock includes preparation and saving."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-wall-hours',type=float,default=21.)
    p.add_argument('--save-reserve-minutes',type=float,default=18.)
    p.add_argument('--work-dir',default='/data/workdir/engine-core')
    p.add_argument('--gpu',default='0')
    p.add_argument('--max-new-hf-gb',type=float,default=32.)
    p.add_argument('--max-local-gb',type=float,default=80.)
    p.add_argument('--min-free-gb',type=float,default=20.)
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--clear-stop',action='store_true')
    return p


def remaining_hours(start,hours,clock=time.time):
    return max(0.,(start+hours*3600-clock())/3600)


def main(argv=None):
    args=parser().parse_args(argv);started=time.time()
    for name in ('max_wall_hours','save_reserve_minutes','max_new_hf_gb','max_local_gb','min_free_gb'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:raise ValueError('positive finite '+name+' required')
    if args.save_reserve_minutes<18 or args.max_wall_hours*60<=args.save_reserve_minutes+10:
        raise ValueError('need >=18min reserve and >10min work time')
    # Set BEFORE importing JAX anywhere in the launcher.
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    os.environ.setdefault('EXTENT_HF_CHECKPOINT_REPO','levkremlev888/extent')
    os.environ.setdefault('EXTENT_HF_REPO_TYPE','dataset')
    if not os.environ.get('TELEGRAM_CHAT_ID') and os.environ.get('TELEGRAM_USER_ID'):
        os.environ['TELEGRAM_CHAT_ID']=os.environ['TELEGRAM_USER_ID']
    required=['HF_TOKEN']+(['TELEGRAM_BOT_TOKEN','TELEGRAM_CHAT_ID'] if args.telegram else [])
    missing=[k for k in required if not os.environ.get(k)]
    if missing:raise ValueError('Set environment variables: '+', '.join(missing))
    root=Path(args.work_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    with FileLock(str(root/'server.lock'),timeout=0):
        stop=root/'STOP'
        if args.clear_stop and stop.exists():stop.unlink()
        if stop.exists():raise ValueError(f'STOP exists; resume with --clear-stop or remove {stop}')
        def request_stop(signum,frame):
            stop.write_text('requested by signal '+str(signum))
            print('Soft stop requested; waiting for bounded save, do not kill -9',flush=True)
        old={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
        from extent.reset_quiet import ParentNotifier
        from extent.server_control import storage_status
        from extent.campaign_checkpoint import write_json_atomic
        notice=ParentNotifier(args.telegram,160)
        launch_path=root/'runtime-status.json';result=dict(status='preparing',max_wall_hours=args.max_wall_hours)
        try:
            result['storage']=storage_status(root,args.max_local_gb,args.min_free_gb)
            notice.emit('started',f'RTX server; {args.max_wall_hours:g}h total incl preparation/save; Wiki control versus C4')
            corpus=root/'data-pack'
            if not (corpus/'manifest.json').exists():
                from scripts.prepare_broad_corpus import main as prepare
                budget=min(2.,remaining_hours(started,args.max_wall_hours)-(args.save_reserve_minutes+10)/60)
                if budget<=0:raise TimeoutError('no corpus preparation budget remains')
                prepare(['--output',str(corpus),'--train-tokens','100000000','--validation-windows','512',
                         '--max-hours',str(budget),'--max-local-gb','8','--stop-file',str(stop)])
            from extent.broad_corpus import PackedCorpus
            packed=PackedCorpus(corpus)
            expected=json.loads((Path(__file__).resolve().parents[1]/'results/EXP-160-corpus-manifest.json').read_text())
            if packed.manifest['contract']!=expected['contract'] or packed.manifest['files']!=expected['files']:
                raise ValueError('server corpus differs from registered C4 bytes')
            if stop.exists():raise TimeoutError('requested stop before model loading')
            remaining=remaining_hours(started,args.max_wall_hours)
            if remaining*60<=args.save_reserve_minutes+10:raise TimeoutError('insufficient training time after preparation')
            from scripts.m3q_corpus_recovery_160 import parser as runner_parser,run
            options=runner_parser().parse_args(['--execution','server-balanced','--corpus-dir',str(corpus),
                '--max-wall-hours',str(remaining),'--save-reserve-minutes',str(args.save_reserve_minutes),
                '--max-steps','1000000000','--max-new-hf-gb',str(args.max_new_hf_gb),
                '--output-dir',str(root/'results'),'--cache-dir',str(root/'cache'),
                '--checkpoint-dir',str(root/'states'),'--stop-file',str(stop),
                '--work-root',str(root),'--max-local-gb',str(args.max_local_gb),
                '--min-free-gb',str(args.min_free_gb),'--no-telegram'])
            wall=started+args.max_wall_hours*3600
            os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(wall-args.save_reserve_minutes*60)
            os.environ['EXTENT_HF_SAVE_DEADLINE_UNIX']=str(wall-60)
            from extent.hf_download_retry import install_download_retries
            install_download_retries()
            import jax
            devices=jax.devices('gpu')
            if len(devices)!=1:raise ValueError('one visible GPU required')
            result['device']=dict(kind=devices[0].device_kind,memory=devices[0].memory_stats() or {})
            print('engine-core:',result['device']['kind'],'remaining_hours=',round(remaining,3),flush=True)
            result=run(options)
            return result
        except TimeoutError as exc:
            result.update(status='preparation_partial',stop_reason=str(exc));return result
        except Exception as exc:
            result.update(status='failed',error_type=type(exc).__name__);raise
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
            result['launcher_elapsed_hours']=(time.time()-started)/3600
            write_json_atomic(launch_path,result)
            pending=result.get('pending_slots',[])
            notice.emit('finished',f"server {result.get('status')}; HF pending={len(pending)}; report={root/'results/runtime-report.json'}")


if __name__=='__main__':main()
