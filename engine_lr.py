"""EXP167: sequential matched-token LR pilots, one GPU and one shared STOP."""
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

MULTIPLIERS=(1,3,10)


def summarize(root,updates):
    rows={}
    for multiplier in MULTIPLIERS:
        path=Path(root)/'results'/f'runtime-lr-x{multiplier}.json'
        if not path.exists():continue
        report=json.loads(path.read_text());row=report['branches'].get('FULL-KD',{})
        point=row.get('final',{}).get('last',{})
        rows[f'LR-x{multiplier}']=dict(row,lr_multiplier=multiplier,
            report_status=report.get('status'),pending_slots=report.get('pending_slots',[]),
            complete=(report.get('status')=='completed' and row.get('accepted_new_updates')==updates
                and row.get('accepted_input_tokens')==updates*2048
                and report.get('finalization_status')=='completed'
                and not report.get('pending_slots') and not report.get('report_sync_error')),
            final_panels={k:v['nll'] for k,v in point.get('panels',{}).items()})
    matched=len(rows)==3 and all(row['complete'] for row in rows.values())
    # No winner or causal LR conclusion from unequal/incomplete tails.
    ranking=sorted(rows,key=lambda name:rows[name]['final_panels']['E']) if matched else []
    return dict(experiment=167,contract=dict(experiment=167,batch=8,protocol='exp167-lr-sweep-summary-v1'),
        status='completed' if matched else 'partial',branches=rows,
        pending_slots=[name+':'+slot for name,row in rows.items() for slot in row['pending_slots']],
        matched_complete=matched,ranking_by_edu_E=ranking,
        caveat='one seed, inspected diagnostic panels; compare D/A/B/C and EN/RU before selecting a recipe')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default=str(Path(__file__).resolve().parent.parent/'engine-core'))
    p.add_argument('--max-wall-hours',type=float,default=9.)
    p.add_argument('--save-reserve-minutes',type=float,default=18.)
    p.add_argument('--pilot-updates',type=int,default=4096)
    p.add_argument('--leave-free-gib',type=float,default=5.)
    p.add_argument('--gpu',default='0')
    p.add_argument('--max-local-gb',type=float,default=220.)
    p.add_argument('--min-free-gb',type=float,default=20.)
    p.add_argument('--max-new-hf-gb',type=float,default=96.)
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--clear-stop',action='store_true')
    args=p.parse_args(argv)
    if not math.isfinite(args.max_wall_hours) or args.max_wall_hours<=0 or args.pilot_updates<1:
        raise ValueError('positive finite time and positive pilot-updates required')
    if args.pilot_updates*8>390625-204673:raise ValueError('pilot exceeds remaining fresh Edu stream')
    if args.save_reserve_minutes<18 or args.max_wall_hours*60<3*(args.save_reserve_minutes+15):
        raise ValueError('need >=18min save reserve per arm and at least15min work per arm')
    if args.max_new_hf_gb<72:raise ValueError('three full checkpoints need >=72GB declared HF budget')
    root=Path(args.work_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    from extent.campaign_checkpoint import write_json_atomic
    with FileLock(str(root/'lr-sweep.lock'),timeout=0):
        # Never clear STOP belonging to a live training process.
        with FileLock(str(root/'server.lock'),timeout=0):
            if args.clear_stop:(root/'STOP').unlink(missing_ok=True)
            if (root/'STOP').exists():raise ValueError('STOP exists; use --clear-stop after prior training exits')
        from extent.server_control import storage_status
        storage=storage_status(root,args.max_local_gb,args.min_free_gb)
        existing=summarize(root,args.pilot_updates)
        todo=[m for m in MULTIPLIERS if not existing['branches'].get(f'LR-x{m}',{}).get('complete')]
        # All active generations retained; atomic replacement also needs one full temporary generation.
        additional=(len(todo)+1)*22e9 if todo else 0
        if storage['free_bytes']<additional+args.min_free_gb*1e9 or storage['work_bytes']+additional>args.max_local_gb*1e9:
            raise OSError('Need space for retained LR currents plus atomic temporary generation; no automatic deletion')
        wall=time.time()+args.max_wall_hours*3600;child=None;stopping=[False]
        def stop(signum,frame):
            stopping[0]=True;(root/'STOP').write_text('EXP167 soft stop')
            if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
        old={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            for index,multiplier in enumerate(todo):
                if stopping[0] or (root/'STOP').exists():break
                remaining=wall-time.time();arm_hours=remaining/(len(todo)-index)/3600
                if arm_hours*60<=args.save_reserve_minutes+15:break
                command=[sys.executable,'-u','-m','engine_full_lr','--work-dir',str(root),
                    '--lr-multiplier',str(multiplier),'--pilot-updates',str(args.pilot_updates),
                    '--max-wall-hours',str(arm_hours),'--save-reserve-minutes',str(args.save_reserve_minutes),
                    '--leave-free-gib',str(args.leave_free_gib),'--gpu',args.gpu,
                    '--max-local-gb',str(args.max_local_gb),'--min-free-gb',str(args.min_free_gb),
                    '--max-new-hf-gb',str(args.max_new_hf_gb/3),
                    '--telegram' if args.telegram else '--no-telegram']
                print('EXP167: starting LR x'+str(multiplier)+'; equal target inputs='+str(args.pilot_updates*2048),flush=True)
                child=subprocess.Popen(command,cwd=Path(__file__).resolve().parent)
                code=child.wait();child=None
                report=summarize(root,args.pilot_updates)
                write_json_atomic(root/'results/runtime-lr.json',report)
                if code:raise RuntimeError(f'LR x{multiplier} failed; inspect runtime-lr-x{multiplier}.json and runtime-lr.log')
                if not report['branches'].get(f'LR-x{multiplier}',{}).get('complete'):break
            result=summarize(root,args.pilot_updates);write_json_atomic(root/'results/runtime-lr.json',result)
            print('EXP167 comparison:',json.dumps(dict(status=result['status'],ranking=result['ranking_by_edu_E'],
                arms={k:dict(tokens=v.get('accepted_input_tokens'),panels=v['final_panels']) for k,v in result['branches'].items()})),flush=True)
            if args.telegram:
                from extent.status_autostart import ensure_listener
                ensure_listener(root,'results/runtime-lr.json')
            return result
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)


if __name__=='__main__':main()
