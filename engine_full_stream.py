"""Full decoder runtime with mandatory batch planner and automatic status bot."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-wall-hours',type=float,default=14.)
    p.add_argument('--save-reserve-minutes',type=float,default=18.)
    p.add_argument('--work-dir',default=str(Path(__file__).resolve().parent.parent/'engine-core'))
    p.add_argument('--edu-train-tokens',type=int,default=100000000)
    p.add_argument('--gpu',default='0')
    p.add_argument('--leave-free-gib',type=float,help='derive JAX pool from current GPU use, reserve this much plus2GiB CUDA margin')
    p.add_argument('--max-new-hf-gb',type=float,default=32.)
    p.add_argument('--max-local-gb',type=float,default=130.)
    p.add_argument('--min-free-gb',type=float,default=20.)
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--clear-stop',action='store_true')
    return p


def remaining_hours(start,hours,clock=time.time):
    return max(0.,(start+hours*3600-clock())/3600)


def configure_gpu_budget(gpu,leave_free_gib):
    if leave_free_gib is None:return None
    if not math.isfinite(leave_free_gib) or leave_free_gib<0:raise ValueError('leave-free-gib must be finite and nonnegative')
    raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    if len(raw.strip().splitlines())!=1:raise ValueError('one physical GPU required for memory budget')
    total,used=map(int,raw.strip().split(','));margin=2.;pool=total-used-(leave_free_gib+margin)*1024
    if pool<=0 or total<=0 or used<0:raise ValueError('insufficient free GPU memory for requested reserve')
    policy=dict(total_mib=total,other_used_mib=used,reserve_gib=leave_free_gib,cuda_margin_gib=margin,pool_gib=pool/1024,fraction=pool/total)
    for name in ('XLA_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_ALLOCATOR','TF_GPU_ALLOCATOR'):
        os.environ.pop(name,None)
    os.environ['XLA_CLIENT_MEM_FRACTION']=str(policy['fraction'])
    os.environ['EXTENT_GPU_BUDGET_POLICY']=json.dumps(policy)
    print('gpu_budget: JAX_pool_GiB=',round(policy['pool_gib'],2),'reserve_GiB=',leave_free_gib,'CUDA_margin_GiB=',margin,flush=True)
    return policy


def main(argv=None):
    args=parser().parse_args(argv);started=time.time()
    for name in ('max_wall_hours','save_reserve_minutes','max_new_hf_gb','max_local_gb','min_free_gb'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name)<=0:raise ValueError('positive finite '+name+' required')
    if args.save_reserve_minutes<18 or args.max_wall_hours*60<=args.save_reserve_minutes+10:
        raise ValueError('need >=18min reserve and >10min work time')
    # Set BEFORE importing JAX anywhere in the launcher.
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    configure_gpu_budget(args.gpu,args.leave_free_gib)
    os.environ.setdefault('EXTENT_HF_CHECKPOINT_REPO','levkremlev888/extent')
    os.environ.setdefault('EXTENT_HF_REPO_TYPE','dataset')
    os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(started+args.max_wall_hours*3600-args.save_reserve_minutes*60)
    os.environ['EXTENT_HF_SAVE_DEADLINE_UNIX']=str(started+args.max_wall_hours*3600-60)
    if not os.environ.get('TELEGRAM_CHAT_ID') and os.environ.get('TELEGRAM_USER_ID'):
        os.environ['TELEGRAM_CHAT_ID']=os.environ['TELEGRAM_USER_ID']
    required=['HF_TOKEN']+(['TELEGRAM_BOT_TOKEN','TELEGRAM_CHAT_ID'] if args.telegram else [])
    missing=[k for k in required if not os.environ.get(k)]
    if missing:raise ValueError('Set environment variables: '+', '.join(missing))
    root=Path(args.work_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    os.environ['EXTENT_HF_RETRY_STATE_DIR']=str(root/'transport')
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
        from extent.hf_artifact_sync import artifact_config_from_env
        notice=ParentNotifier(args.telegram,166)
        launch_path=root/'runtime-full-stream-status.json';result=dict(status='preparing',max_wall_hours=args.max_wall_hours)
        try:
            if args.telegram:
                from extent.status_autostart import ensure_listener
                result['status_bot']=ensure_listener(root,'results/runtime-full-stream.json')
            result['storage']=storage_status(root,args.max_local_gb,args.min_free_gb)
            notice.emit('started',f'RTX server; {args.max_wall_hours:g}h total incl preparation/save; FULL-KD CURRENT157685; mandatory batch1..8 planner; sequential Edu')
            corpus=root/'data-pack'
            if not (corpus/'manifest.json').exists():
                from scripts.prepare_broad_corpus import main as prepare
                budget=min(2.,remaining_hours(started,args.max_wall_hours)-(args.save_reserve_minutes+10)/60)
                if budget<=0:raise TimeoutError('no corpus preparation budget remains')
                prepare(['--output',str(corpus),'--train-tokens','100000000','--validation-windows','512',
                         '--max-hours',str(budget),'--max-local-gb','8','--stop-file',str(stop)])
            from extent.broad_corpus import PackedCorpus
            packed=PackedCorpus(corpus)
            expected=json.loads((Path(__file__).resolve().parent/'results/EXP-160-corpus-manifest.json').read_text())
            if packed.manifest['contract']!=expected['contract'] or packed.manifest['files']!=expected['files']:
                raise ValueError('server corpus differs from registered C4 bytes')
            if stop.exists():raise TimeoutError('requested stop before model loading')
            remaining=remaining_hours(started,args.max_wall_hours)
            if remaining*60<=args.save_reserve_minutes+10:raise TimeoutError('insufficient training time after preparation')
            from extent.full_stream_166 import freeze_parent
            os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(started+args.max_wall_hours*3600-args.save_reserve_minutes*60)
            edu=root/'data-pack-edu'
            if not (edu/'manifest.json').exists():
                from scripts.prepare_edu_corpus import main as prepare_edu
                prep_hours=min(3.,remaining_hours(started,args.max_wall_hours)-(args.save_reserve_minutes+10)/60)
                if prep_hours<=0:raise TimeoutError('no Edu preparation budget remains')
                prepare_edu(['--output',str(edu),'--c4-dir',str(corpus),'--train-tokens',str(args.edu_train_tokens),
                    '--max-hours',str(prep_hours),'--stop-file',str(stop)])
            source_plan,_=freeze_parent(root,artifact_config_from_env(include_kaggle_secrets=False),
                deadline=started+args.max_wall_hours*3600-args.save_reserve_minutes*60)
            remaining=remaining_hours(started,args.max_wall_hours)
            if remaining*60<=args.save_reserve_minutes+10:raise TimeoutError('insufficient time after source plan')
            from scripts.m3q_full_stream_166 import parser as runner_parser,run
            options=runner_parser().parse_args(['--source-plan',str(source_plan),'--execution','server-balanced','--corpus-dir',str(corpus),'--edu-dir',str(edu),
                '--deadline-unix',str(started+args.max_wall_hours*3600),
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
            jax.config.update('jax_compilation_cache_dir',str(root/'compile-cache'))
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
            notice.emit('finished',f"server {result.get('status')}; HF pending={len(pending)}; report={root/'results/runtime-full-stream.json'}")


if __name__=='__main__':main()
