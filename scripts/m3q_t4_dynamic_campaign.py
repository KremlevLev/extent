"""EXP125--127: dynamic token-wise bounds and MLP adaptation, Kaggle T4x2."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from scripts.m3q_t4_campaign import write,notify,sha,baseline,Session as ReportSession
from scripts.m3q_t4_scale_campaign import background,context_report,reconcile

ROOT=Path(__file__).resolve().parents[1]
NAMES={125:'dynamic-mamba-bound',126:'absolute-residual-bound',127:'bounded-mlp-adaptation'}
FILES=('scripts/m3q_t4_dynamic_campaign.py','extent/dynamic_recovery.py',
    'scripts/m3q_t4_scale_campaign.py','extent/t4_scale_recovery.py','extent/t4_scale_data.py',
    'scripts/m3q_t4_campaign.py','extent/t4_patient.py','extent/plateau_forensics.py',
    'extent/layers/mamba3.py','extent/layers/common.py','extent/qwen3_teacher.py',
    'extent/stable_gradient_clip.py','extent/recovery_durability.py','extent/chunked_checkpoint.py',
    'extent/hf_artifact_sync.py','extent/recovery_subspace.py','extent/efficiency_sources.py',
    'extent/calibration_data.py','extent/pinned_pg19_manifest.py','requirements-t4.txt')

def contract_for(experiment,worker,workers):
    from extent.dynamic_recovery import TARGET_STEPS,RAMP_STEPS,EVAL_STEPS,arm_for
    from extent.t4_scale_data import PANELS
    return dict(experiment=experiment,name=NAMES[experiment],seed=123,worker=worker,workers=workers,
        source_pin_sha=sha(ROOT/'results/EXP-114-source-pin.json'),source='EXP111 MAIN32768',
        source_replay_tolerance=.03,hook_background_tolerance=.01,inspection_relative_output_tolerance=.02,
        steps=TARGET_STEPS,ramp_steps=RAMP_STEPS,eval_steps=list(EVAL_STEPS),train_panels=PANELS,
        arm=arm_for(experiment,worker),
        limiter='per-token hidden-axis stable log-RMS; relative Mamba cap rho*max(RMS(input),1e-3); absolute cap after mixer addition AND MLP addition, all28 layers; geometric coefficient ramp2048',
        gradient='full chain rule through BOTH input/output RMS and caps; no detached or straight-through gate',
        optimizer='fresh Adam .9/.999 eps1e-8; joint stable clip1; Mamba-R32 LR3e-4 / optional MLP down-proj rank32 LR1e-4; warmup256; no learned gains',
        checkpoint='local128; remote2048 or120min and deadline; current+Adam+best parameters;4GB total/invocation default',
        selection='source A64/B32/C64; A16 monitor before full ramp, A64 selects best ONLY at >=2048 fixed endpoints; B/C never select; source+last+best context/profile',
        interpretation='single seed exploratory; A/B/C exposed; no superiority/plateau cause established; compare equal steps AND intervention strength; cross-session CPU/GPU rounding recorded',
        code={f:sha(ROOT/f) for f in FILES})

class Session(ReportSession):
    def __init__(self,args,hub):
        from extent.hf_artifact_sync import restore_artifact
        import jax
        from importlib.metadata import version
        self.args,self.hub=args,hub
        self.prefix=f'experiments/exp{args.experiment}-t4/worker-{args.worker}'
        self.path=Path(args.output_dir)/f'extent-m3q-t4-{args.experiment}-worker-{args.worker}.json'
        self.small=self.path.with_name(self.path.stem+'-summary.txt')
        self.started=time.monotonic();self.next_sync=0.;self.uploaded=0
        self.hard_deadline=args.deadline+args.save_reserve_minutes*60
        if not self.path.exists():restore_artifact(self.path,f'{self.prefix}/latest.json',hub)
        contract=json.loads(json.dumps(contract_for(args.experiment,args.worker,args.workers)))
        self.result=json.loads(self.path.read_text()) if self.path.exists() else dict(contract=contract,tasks={},fits={},
            sessions=[],cost=dict(attempted_updates=0,training_input_tokens=0,teacher_input_tokens=0,rejected_updates=0))
        self.previous_status=self.result.get('status')
        if self.result['contract']!=contract:raise ValueError('dynamic GPU resume contract changed')
        runtime=dict(versions={n:version(n) for n in ('jax','jaxlib','flax','optax')},
            device=jax.devices('gpu')[0].device_kind,matmul_precision=str(jax.config.jax_default_matmul_precision))
        if self.result.get('runtime',runtime)!=runtime:raise ValueError('dynamic GPU runtime changed')
        self.result['runtime']=runtime
        self.invocation=dict(started_at=time.time(),max_wall_hours=args.max_wall_hours,uploaded_bytes=0)
        self.result['sessions'].append(self.invocation);self.result['status']='preflight';self.result.pop('error',None)

    def finalize_budget(self):
        self.args.deadline=self.hard_deadline-180

    def task(self,key,call):
        if key in self.result['tasks']:return self.result['tasks'][key]['value']
        self.check();begun=time.monotonic()
        try:value=call()
        finally:
            cost=self.result['cost'];cost['attempted_tasks']=cost.get('attempted_tasks',0)+1
            cost['task_seconds']=cost.get('task_seconds',0)+time.monotonic()-begun
        self.result['tasks'][key]=dict(seconds=time.monotonic()-begun,value=value)
        print(f'EXP{self.args.experiment} GPU{self.args.worker} {key}: saved',flush=True)
        self.persist();return value



def run_training(s,p,tokens):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from extent.dynamic_recovery import Engine,initialize,optimizer,TARGET_STEPS,RAMP_STEPS,EVAL_STEPS
    from extent.t4_scale_recovery import make_update
    from extent.t4_scale_data import training_index
    from extent.recovery_durability import RecoveryCheckpointStore
    from extent.efficiency_sources import tree_digest
    spec=s.result['contract']['arm'];e=Engine(p,spec)
    with jax.default_device(p.device):q=initialize(p,spec)
    background(s,p,e,q,tokens)
    # Explicit source controls on ALL final panels: do not compare A with C.
    source=s.result.setdefault('source_controls',{})
    for panel in ('A','B','C'):
        source[panel]=s.task(f'source-control/{panel}',lambda panel=panel:e.evaluate(q,tokens[panel]))
    source['profile']=s.task('source-control/profile',lambda:e.inspect(q,tokens['A'][:1]))
    source['context']=context_report(s,p,e,q,tokens,'source-control')
    # Dry diagnosis uses training windows, not B/C or a selection sweep.
    e.set_step(RAMP_STEPS)
    s.result['full_bound_initial']=s.task('initial-full-bound/FIT8',lambda:e.evaluate(q,tokens['FIT'][:8]))
    s.result['full_bound_initial_profile']=s.task('initial-full-bound/profile',lambda:e.inspect(q,tokens['FIT'][:1]))
    if not all(r['accepted'] and r['relative_bound_ok'] and r['absolute_bound_ok'] for r in s.result['full_bound_initial_profile']['layers']):
        raise ValueError('full-strength bound/profile engineering gate failed')
    tx=optimizer(q,spec);state=tx.init(q);step_update=make_update(tx)
    cursor=0;best=q;best_step=-1
    store=RecoveryCheckpointStore(Path(s.args.cache_dir)/f'dynamic-states-{s.args.experiment}-{s.args.worker}',
        f'{s.prefix}/states',s.hub)
    cc=s.result['contract'];slot='MAIN'
    template=dict(parameters=q,optimizer=state,best=best,best_step=jnp.array(-1,jnp.int32))
    restored=store.restore(slot,cc,template)
    row=s.result['fits'].setdefault('MAIN',dict(step=0,durable_step=0,best_step=-1,history=[],evaluations=[]))
    saved_cursor=None;saved_best_step=None
    if restored:
        payload,meta=restored;q=p.placed(payload['parameters']);state=p.placed(payload['optimizer'])
        best=p.placed(payload['best']);cursor=meta['step'];best_step=int(payload['best_step'])
        if best_step>cursor or (best_step!=-1 and best_step<RAMP_STEPS):raise ValueError('invalid post-ramp best cursor')
        if row.get('durable_step',0)>cursor:raise ValueError('durable report cursor ahead of checkpoint')
        saved_cursor=cursor;saved_best_step=best_step
    elif row.get('durable_step',0):raise FileNotFoundError('claimed durable state unavailable')
    reconcile(row,cursor,best_step)
    for key in list(s.result['tasks']):
        if key.startswith('endpoint-') and int(key.split('/')[0].split('-')[1])>cursor:
            s.result.setdefault('discarded_future_tasks',[]).append(key);s.result['tasks'].pop(key)
    e.set_step(cursor)
    best_nll=None
    if best_step>=RAMP_STEPS:
        e.set_step(best_step);best_nll=e.evaluate(best,tokens['A'])['nll'];e.set_step(cursor)
    row['best_nll']=best_nll
    if not row['evaluations']:row['evaluations'].append(dict(step=cursor,A16=e.evaluate(q,tokens['A'][:16])['nll'],ramp_strength=float(e.phase),selection_eligible=False))
    row['coordinates']=sum(x.size for x in jax.tree.leaves(q))
    row['checkpoint_payload_bytes']=sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(dict(parameters=q,optimizer=state,best=best)))
    s.result['status']='training';s.persist()
    last_remote_step=cursor;last_remote_time=time.monotonic()
    def checkpoint(remote=False):
        nonlocal saved_cursor,saved_best_step
        if remote and row.get('durable_step')==cursor and row.get('durable_best_step')==best_step and not row.get('checkpoint_sync_error'):return True
        if saved_cursor!=cursor or saved_best_step!=best_step:
            meta=store.save(slot,cc,dict(parameters=q,optimizer=state,best=best,best_step=jnp.array(best_step,jnp.int32)),cursor)
            saved_cursor=cursor;saved_best_step=best_step
        else:meta=json.loads((store.directory(slot)/'checkpoint.json').read_text())
        if not remote:return True
        receipt=store.directory(slot)/f"{meta['generation']}.uploaded.json"
        before=set(json.loads(receipt.read_text())['files']) if receipt.exists() else set()
        meta_bytes=(store.directory(slot)/'checkpoint.json').stat().st_size
        need=sum(c['bytes'] for r in meta['leaves'] for c in r['chunks'] if c['file'] not in before)+meta_bytes
        if need+s.uploaded>s.args.max_new_hf_gb*1e9:
            row['checkpoint_sync_error']='invocation upload cap';s.persist();return False
        success=store.sync(slot,force=True,deadline=s.hard_deadline-120)
        after=set(json.loads(receipt.read_text())['files']) if receipt.exists() else set()
        s.uploaded+=sum(c['bytes'] for r in meta['leaves'] for c in r['chunks'] if c['file'] in after-before)+meta_bytes
        if success:row['durable_step']=cursor;row['durable_best_step']=best_step;row.pop('checkpoint_sync_error',None)
        else:row['checkpoint_sync_error']='HF pending; local snapshot retained'
        s.persist();return success
    stopped='completed';next_notice=time.monotonic()+1800
    try:
        while cursor<TARGET_STEPS:
            s.check();e.set_step(cursor);t=tokens['FIT'][training_index(cursor)][None]
            begun=time.monotonic();cost=s.result['cost']
            cost['attempted_updates']+=1;cost['training_input_tokens']+=t.size
            try:
                health,gradient=e.loss_and_grad(q,t)
                proposed,after,gradient_health=step_update(q,state,gradient);health.update(gradient_health)
            except FloatingPointError:cost['rejected_updates']+=1;raise
            finally:cost['training_seconds']=cost.get('training_seconds',0)+time.monotonic()-begun
            q,state=proposed,after;cursor+=1;row['step']=cursor;e.set_step(cursor)
            cost['accepted_updates']=cost.get('accepted_updates',0)+1
            cost['accepted_training_input_tokens']=cost.get('accepted_training_input_tokens',0)+t.size
            cost['accepted_ce_targets']=cost.get('accepted_ce_targets',0)+t.shape[0]*(t.shape[1]-1)
            if cursor in (1,2,4,8,16) or cursor%32==0:
                row['history'].append(dict(step=cursor,ramp_strength=float(e.phase),loss_before_update=True,**health))
                if cursor==1:row['first_step']=dict(seconds=time.monotonic()-begun,device_memory=p.device.memory_stats())
                s.persist()
            if cursor in EVAL_STEPS:
                validation=e.evaluate(q,tokens['A'] if cursor>=RAMP_STEPS else tokens['A'][:16])
                record=dict(step=cursor,ramp_strength=float(e.phase),selection_eligible=cursor>=RAMP_STEPS,
                    B32=e.evaluate(q,tokens['B']),activation_penalty=validation['activation_penalty'])
                if cursor>=RAMP_STEPS:
                    record['A64']=validation;record['C64']=e.evaluate(q,tokens['C'])
                    record['profile']=e.inspect(q,tokens['A'][:1])
                    if best_nll is None or validation['nll']<best_nll:
                        best=q;best_step=cursor;best_nll=validation['nll'];row.update(best_step=best_step,best_nll=best_nll)
                else:record['A16']=validation['nll']
                row['evaluations'].append(record);s.persist()
            if cursor%128==0:checkpoint()
            if cursor-last_remote_step>=2048 or time.monotonic()-last_remote_time>=7200:
                if not checkpoint(remote=True):stopped='storage_partial';break
                last_remote_step=cursor;last_remote_time=time.monotonic()
            if time.monotonic()>=next_notice:
                notify(s.args,f'Extent T4 EXP{s.args.experiment} GPU{s.args.worker}: step={cursor}/{TARGET_STEPS}; durable={row.get("durable_step")}; ramp={float(e.phase):.3f}; best A64={best_nll}')
                next_notice=time.monotonic()+1800
    except TimeoutError:stopped='deadline_partial'
    finally:
        # A clean timeout may interrupt a streamed backward; only accepted q/state saved.
        checkpoint(remote=True)
    s.finalize_budget();row['finalization_status']='running';s.persist()
    try:
        final={}
        candidates=[('last',q,cursor)]
        if best_step>=RAMP_STEPS:candidates.append(('best',best,best_step))
        for label,params,at in candidates:
            e.set_step(at);state_sha=tree_digest(jax.device_get(params))
            prefix=f'endpoint-{cursor}/{label}/state-{state_sha}/phase-{at}'
            final[label]=dict(step=at,ramp_strength=float(e.phase),panels={},state_sha256=state_sha)
            for panel in ('A','B','C'):
                final[label]['panels'][panel]=s.task(f'{prefix}/{panel}',lambda params=params,panel=panel:e.evaluate(params,tokens[panel],profile=True))
            final[label]['layer_profile']=s.task(f'{prefix}/layer-profile',lambda params=params:e.inspect(params,tokens['A'][:1]))
            final[label]['context']=context_report(s,p,e,params,tokens,prefix)
        row['final']=final;row['finalization_status']='completed'
    except TimeoutError:
        row['finalization_status']='deadline_partial'
        if stopped=='completed':stopped='finalization_partial'
    return stopped


def worker(args):
    import jax
    from extent.hf_artifact_sync import artifact_config_from_env
    from extent.t4_patient import Patient
    from extent.t4_scale_data import load_data
    devices=jax.devices('gpu')
    if len(devices)!=1:raise ValueError('worker dispatch requires one visible GPU')
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    s=Session(args,hub)
    try:
        if not s.persist(force=True):raise RuntimeError('HF report write preflight failed')
        if s.previous_status=='completed' and not s.result.get('pending_fit_states'):
            s.result['status']='completed';s.persist(force=True)
            return s.result
        tokens,meta=load_data(Path(args.cache_dir)/f'long-data-{args.worker}',s.check)
        s.result['data']=meta;s.persist()
        p=Patient(Path(args.cache_dir)/f'patient-{args.worker}',hub,devices[0],s.check)
        baseline(s,p,tokens);s.result['status']='running';s.persist()
        s.result['status']=run_training(s,p,tokens)
    except TimeoutError:s.result['status']='deadline_partial'
    except Exception as exc:
        s.result['status']='failed';s.result['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        s.result['pending_fit_states']=[k for k,v in s.result['fits'].items() if v.get('checkpoint_sync_error')]
        s.result['device_memory']=devices[0].memory_stats();s.persist(force=True)
        notify(args,f'Extent T4 EXP{args.experiment} GPU{args.worker}: {s.result["status"]}; progress={[(k,v.get("step"),v.get("durable_step")) for k,v in s.result["fits"].items()]}; pending={len(s.result["pending_fit_states"])}; HF={s.result.get("report_sync_error")}')
    if s.result['status']=='failed':raise RuntimeError('long GPU worker failed; inspect JSON')
    return s.result


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=int,choices=NAMES,required=True)
    p.add_argument('--max-wall-hours',type=float,default=11.5)
    p.add_argument('--save-reserve-minutes',type=float,default=60)
    p.add_argument('--max-new-hf-gb',type=float,default=4)
    p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--cache-dir',default='/kaggle/working/extent-t4-cache')
    p.add_argument('--workers',type=int,default=2,choices=(2,))
    p.add_argument('--worker',type=int)
    p.add_argument('--deadline',type=float)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if not all(math.isfinite(v) and v>0 for v in (args.max_wall_hours,args.save_reserve_minutes,args.max_new_hf_gb)):
        raise ValueError('positive finite limits required')
    if args.max_wall_hours*60<=args.save_reserve_minutes:raise ValueError('reserve consumes work budget')
    if args.plan_only:
        result=[contract_for(args.experiment,w,args.workers) for w in range(args.workers)]
        print(json.dumps(result,indent=2));return result
    if args.worker is not None:
        if args.deadline is None or not 0<=args.worker<args.workers:raise ValueError('invalid worker dispatch')
        return worker(args)
    started=time.time();deadline=started+args.max_wall_hours*3600-args.save_reserve_minutes*60
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    children=[];handles=[]
    notify(args,f'Extent T4 EXP{args.experiment}: started; wall={args.max_wall_hours}h; long work/resumable')
    try:
        for index in range(args.workers):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(index),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
            cmd=[sys.executable,'-u','-m','scripts.m3q_t4_dynamic_campaign','--experiment',str(args.experiment),
                '--worker',str(index),'--workers',str(args.workers),'--deadline',str(deadline),
                '--max-wall-hours',str(args.max_wall_hours),'--save-reserve-minutes',str(args.save_reserve_minutes),
                '--max-new-hf-gb',str(args.max_new_hf_gb/args.workers),'--output-dir',str(output),'--cache-dir',args.cache_dir]
            if not args.telegram:cmd.append('--no-telegram')
            handle=(output/f'exp{args.experiment}-gpu{index}.log').open('a',encoding='utf-8');handles.append(handle)
            children.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
        progress_at=0
        while any(c.poll() is None for c in children):
            if time.time()-started>=args.max_wall_hours*3600:raise TimeoutError('registered session wall deadline')
            if time.time()-progress_at>=60:
                progress=[]
                for i,c in enumerate(children):
                    path=output/f'extent-m3q-t4-{args.experiment}-worker-{i}.json'
                    r=json.loads(path.read_text()) if path.exists() else {}
                    progress.append(dict(gpu=i,status=r.get('status','loading'),tasks=len(r.get('tasks',{})),
                        fits={k:v.get('step') for k,v in r.get('fits',{}).items()},exit=c.poll()))
                print(f'EXP{args.experiment} {(time.time()-started)/3600:.2f}h {progress}',flush=True);progress_at=time.time()
            time.sleep(5)
        return aggregate(args,children,started)
    except Exception as exc:
        notify(args,f'Extent T4 EXP{args.experiment}: runner stopped {type(exc).__name__}; inspect durable checkpoints and logs')
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        try:aggregate(args,children,started)
        except RuntimeError:pass  # aggregate is written before reporting failed workers
        raise
    finally:
        for c in children:
            if c.poll() is None:
                c.terminate()
                try:c.wait(timeout=15)
                except subprocess.TimeoutExpired:c.kill();c.wait()
        for handle in handles:handle.close()


def aggregate(args,children,started):
    output=Path(args.output_dir);workers=[]
    for i,child in enumerate(children):
        path=output/f'extent-m3q-t4-{args.experiment}-worker-{i}.json'
        r=json.loads(path.read_text()) if path.exists() else dict(status='failed',error='no worker report')
        workers.append(dict(worker=i,exit=child.returncode,report=r))
    failed=any(w['exit']!=0 or w['report']['status']=='failed' for w in workers)
    statuses={w['report']['status'] for w in workers}
    status=('failed' if failed else 'completed' if statuses=={'completed'} else
        'storage_partial' if 'storage_partial' in statuses else
        'finalization_partial' if 'finalization_partial' in statuses else 'deadline_partial')
    result=dict(experiment=args.experiment,status=status,wall_hours=(time.time()-started)/3600,workers=workers)
    path=output/f'extent-m3q-t4-{args.experiment}.json';write(path,result)
    summary=dict(experiment=args.experiment,status=status,wall_hours=result['wall_hours'],workers=[dict(worker=w['worker'],
        status=w['report']['status'],fits={k:dict(step=v.get('step'),durable_step=v.get('durable_step'),best_nll=v.get('best_nll'))
            for k,v in w['report'].get('fits',{}).items()},error=w['report'].get('error')) for w in workers])
    write(path.with_name(path.stem+'-summary.txt'),summary)
    print(json.dumps(summary,indent=2),flush=True)
    pending=sum(len(w['report'].get('pending_fit_states',[])) for w in workers)
    sync_errors=sum(bool(w['report'].get('report_sync_error')) for w in workers)
    notify(args,f'Extent T4 EXP{args.experiment}: {status}; pending states={pending}; report sync errors={sync_errors}; download {path.name}')
    if failed:raise RuntimeError('long GPU experiment failed; inspect aggregate JSON')
    return result


if __name__=='__main__':main()
