"""EXP120--123: long causal scale probes and streamed full-model recovery."""
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from scripts.m3q_t4_campaign import write,notify,sha,baseline,Session as ReportSession

ROOT=Path(__file__).resolve().parents[1]
NAMES={120:'scale-interventions',121:'learn-scales',122:'rank32-scale-control',123:'scale-logit-distillation'}
FILES=('scripts/m3q_t4_scale_campaign.py','extent/t4_scale_recovery.py','extent/t4_scale_data.py',
       'scripts/m3q_t4_campaign.py','extent/t4_patient.py','extent/plateau_forensics.py',
       'extent/layers/mamba3.py','extent/layers/common.py','extent/qwen3_teacher.py',
       'extent/stable_gradient_clip.py','extent/recovery_durability.py','extent/chunked_checkpoint.py',
       'extent/hf_artifact_sync.py','requirements-t4.txt')


def probe_jobs(order):
    order=tuple(order);spikes=(0,1,3,11,16,25,26)
    groups=[('all',order),('spikes',spikes),('early',tuple(j for j in order if j<8)),
            ('middle',tuple(j for j in order if 8<=j<20)),('late',tuple(j for j in order if j>=20))]
    gains=(.75,.5,.25,.1,.03,.01,.003,.001,1e-4,1e-5,1e-6,1e-8,0.)
    jobs=[]
    # Round-robin by strength: broad coverage before an interrupted sweep.
    for gain in gains:
        for name,layers in groups+[(f'layer-{j}',(j,)) for j in spikes]+[(f'layer-{j}',(j,)) for j in order if j not in spikes]:
            jobs.append(dict(key=f'{name}/gain-{gain:g}',layers=list(layers),gain=gain))
    for pair in itertools.combinations(spikes,2):
        for gain in (.5,.1,.01,0.):
            jobs.append(dict(key=f'pair-{pair[0]}-{pair[1]}/gain-{gain:g}',layers=list(pair),gain=gain))
    return jobs


def contract_for(experiment,worker,workers):
    from extent.t4_scale_recovery import TARGET_STEPS,EVAL_STEPS,arm_for
    from extent.t4_patient import patient_config
    from extent.t4_scale_data import PANELS
    pin=json.loads((ROOT/'results/EXP-114-source-pin.json').read_text());cfg=patient_config(pin)
    return dict(experiment=experiment,name=NAMES[experiment],seed=123,worker=worker,workers=workers,
        source_pin_sha=sha(ROOT/'results/EXP-114-source-pin.json'),source='EXP111 MAIN32768',
        source_replay_tolerance=.03,hook_background_tolerance=.01,inspection_relative_output_tolerance=.02,
        practical_nll_delta=.1,
        steps=TARGET_STEPS,eval_steps=list(EVAL_STEPS),train_panels=PANELS,
        arm=arm_for(experiment,worker) if experiment!=120 else None,
        probes=probe_jobs(cfg.mamba_layer_indices) if experiment==120 else None,
        optimizer='fresh Adam b1=.9/b2=.999 eps1e-8, joint stable clip1; LR gain .01 / old-rank32 3e-4;256 warmup',
        checkpoint='local128; remote4096 or90min, phase end and deadline; current+Adam+best parameters; no full-weight copies',
        selection='training best A16 among fixed endpoints; terminal A64/B32/C64/context; probe top3 by A64 then B32/C64/context',
        limitations='A/B/C reused; no independent test, causal mechanism and rank+protection superiority not assumed',
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
        if self.result['contract']!=contract:raise ValueError('long GPU resume contract changed')
        runtime=dict(versions={n:version(n) for n in ('jax','jaxlib','flax','optax')},
            device=jax.devices('gpu')[0].device_kind,matmul_precision=str(jax.config.jax_default_matmul_precision))
        if self.result.get('runtime',runtime)!=runtime:raise ValueError('long GPU runtime changed')
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


def background(s,p,e,q,tokens):
    import numpy as np
    reference=s.result['tasks']['source-replay']['value']['windows']
    value=s.task('hook-background',lambda:e.evaluate(q,tokens['A']))
    error=float(np.max(np.abs(np.asarray(value['windows'])-reference)))
    s.result['hook_background']=dict(max_abs_per_window=error,tolerance=.01,passed=error<=.01)
    s.persist()
    if error>.01:raise ValueError('gain1/zero-change hook differs from same-GPU source')


def context_report(s,p,e,q,tokens,key):
    import numpy as np
    windows=tokens['B'].reshape(8,1024)
    report={}
    for length in (128,256,512,1024):
        t=windows[:,-length:]
        current=s.task(f'{key}/context-{length}',lambda t=t:e.evaluate(q,t,tail=127,profile=True))
        def source(t=t):
            v=p.evaluate(t);a=np.asarray(v['token_nll'])[:,-127:]
            return dict(nll=float(a.mean()),windows=a.mean(1).tolist())
        base=s.task(f'baseline-context-{length}',source)
        report[str(length)]=dict(candidate=current,baseline=base,delta_nll=current['nll']-base['nll'])
    return report


def run_probes(s,p,tokens):
    import jax
    import jax.numpy as jnp
    from extent.t4_scale_recovery import Engine
    spec=dict(scales=True,lora=False,penalty=0.,alpha=0.)
    e=Engine(p,spec,direct=True)
    identity={str(j):({'gain':jnp.array(1.,jnp.float32)} if j in p.cfg.mamba_layer_indices else {}) for j in range(p.cfg.num_layers)}
    background(s,p,e,identity,tokens)
    jobs=s.result['contract']['probes']
    complete=True
    try:
        for index,job in enumerate(jobs):
            if index%s.args.workers!=s.args.worker:continue
            q={j:dict(v) for j,v in identity.items()}
            for j in job['layers']:q[str(j)]['gain']=jnp.array(job['gain'],jnp.float32)
            def run(q=q,job=job):
                value=e.evaluate(q,tokens['A'],profile=True)
                return dict(**job,validation=value,delta_nll=value['nll']-s.result['tasks']['hook-background']['value']['nll'])
            s.task('probe/'+job['key'],run)
    except TimeoutError:complete=False
    s.finalize_budget()
    # A deadline-truncated search is explicitly labelled; interim selections
    # are exploratory, not a claim about unmeasured grid points.
    values=[t['value'] for k,t in s.result['tasks'].items() if k.startswith('probe/')]
    selected=sorted(values,key=lambda v:(v['validation']['nll'],v['key']))[:3]
    s.result['probe_coverage']=dict(completed=len(values),planned=sum(i%s.args.workers==s.args.worker for i in range(len(jobs))),complete=complete)
    s.result['selected_probes']=[r['key'] for r in selected];s.persist(force=True)
    for row in selected:
        q={j:dict(v) for j,v in identity.items()}
        for j in row['layers']:q[str(j)]['gain']=jnp.array(row['gain'],jnp.float32)
        key='verify/'+row['key']
        for panel in ('B','C'):
            s.task(f'{key}/{panel}',lambda q=q,panel=panel:e.evaluate(q,tokens[panel],profile=True))
        s.task(f'{key}/layer-profile',lambda q=q:e.inspect(q,tokens['A'][:1]))
        context_report(s,p,e,q,tokens,key)
    return 'completed' if complete else 'deadline_partial'


def reconcile(row,cursor,best_step):
    if row.get('step',0)!=cursor:
        row.setdefault('rollbacks',[]).append(dict(reported=row.get('step',0),restored=cursor))
    row['step']=cursor;row['best_step']=best_step
    row['history']=[v for v in row.get('history',[]) if v['step']<=cursor]
    row['evaluations']=[v for v in row.get('evaluations',[]) if v['step']<=cursor]
    row.pop('final',None)


def run_training(s,p,tokens):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from extent.t4_scale_recovery import Engine,initialize,optimizer,make_update,gain_report,TARGET_STEPS,EVAL_STEPS
    from extent.t4_scale_data import training_index
    from extent.recovery_durability import RecoveryCheckpointStore
    spec=s.result['contract']['arm'];e=Engine(p,spec)
    with jax.default_device(p.device):q=initialize(p,spec)
    background(s,p,e,q,tokens)
    tx=optimizer(q,spec);state=tx.init(q);step_update=make_update(tx);best=q;cursor=0;best_step=0
    store=RecoveryCheckpointStore(Path(s.args.cache_dir)/f'long-states-{s.args.experiment}-{s.args.worker}',
        f'{s.prefix}/states',s.hub)
    cc=s.result['contract'];slot='MAIN'
    template=dict(parameters=q,optimizer=state,best=best,best_step=jnp.array(0,jnp.int32))
    restored=store.restore(slot,cc,template)
    row=s.result['fits'].setdefault('MAIN',dict(step=0,durable_step=0,best_step=0,history=[],evaluations=[]))
    if restored:
        payload,meta=restored;q=p.placed(payload['parameters']);state=p.placed(payload['optimizer'])
        best=p.placed(payload['best']);cursor=meta['step'];best_step=int(payload['best_step'])
        if best_step>cursor:raise ValueError('best state is ahead of checkpoint cursor')
        if row.get('durable_step',0)>cursor:raise ValueError('durable report cursor ahead of restored checkpoint')
    elif row.get('durable_step',0):raise FileNotFoundError('claimed durable training state missing')
    reconcile(row,cursor,best_step)
    for key in list(s.result['tasks']):
        if key.startswith('endpoint-') and int(key.split('/')[0].split('-')[1])>cursor:
            s.result.setdefault('discarded_future_tasks',[]).append(key)
            s.result['tasks'].pop(key)
    # The restored best's A16 score is recomputed; never trust a future report
    # metric that survived while its local checkpoint was lost.
    best_nll=e.evaluate(best,tokens['A'][:16])['nll'];row['best_nll']=best_nll
    if not row['evaluations']:row['evaluations'].append(dict(step=cursor,A16=best_nll,gains=gain_report(q)))
    s.result['status']='training';s.persist()
    last_remote_step=cursor;last_remote_time=time.monotonic()
    def checkpoint(remote=False):
        if remote and row.get('durable_step')==cursor and row.get('durable_best_step')==best_step and not row.get('checkpoint_sync_error'):
            return True
        meta=store.save(slot,cc,dict(parameters=q,optimizer=state,best=best,best_step=jnp.array(best_step,jnp.int32)),cursor)
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
    stopped='completed'
    try:
        while cursor<TARGET_STEPS:
            s.check();t=tokens['FIT'][training_index(cursor)][None]
            begun=time.monotonic()
            # Count attempted work even if a nonfinite proposal is rejected.
            s.result['cost']['attempted_updates']+=1;s.result['cost']['training_input_tokens']+=t.size
            if spec['alpha']:s.result['cost']['teacher_input_tokens']+=t.size
            try:
                health,gradient=e.loss_and_grad(q,t)
                q_after,state_after,gradient_health=step_update(q,state,gradient)
                health.update(gradient_health)
            except FloatingPointError:
                s.result['cost']['rejected_updates']+=1;raise
            finally:
                s.result['cost']['training_seconds']=s.result['cost'].get('training_seconds',0)+time.monotonic()-begun
            q,state=q_after,state_after;cursor+=1;row['step']=cursor
            cost=s.result['cost']
            cost['accepted_updates']=cost.get('accepted_updates',0)+1
            cost['accepted_training_input_tokens']=cost.get('accepted_training_input_tokens',0)+t.size
            cost['accepted_ce_targets']=cost.get('accepted_ce_targets',0)+t.shape[0]*(t.shape[1]-1)
            if cursor in (1,2,4,8,16) or cursor%32==0:
                row['history'].append(dict(step=cursor,**health))
                if cursor==1:
                    row['first_step']=dict(seconds=time.monotonic()-begun,device_memory=p.device.memory_stats())
                s.persist()
            if cursor in EVAL_STEPS:
                validation=e.evaluate(q,tokens['A'][:16])
                monitor=e.evaluate(q,tokens['B'])
                row['evaluations'].append(dict(step=cursor,A16=validation['nll'],gains=gain_report(q),
                    B32=monitor,activation_penalty=validation['activation_penalty']))
                if validation['nll']<best_nll:
                    best=q;best_step=cursor;best_nll=validation['nll']
                    row.update(best_step=best_step,best_nll=best_nll)
                s.persist()
            if cursor%128==0:checkpoint()
            if cursor-last_remote_step>=4096 or time.monotonic()-last_remote_time>=5400:
                if not checkpoint(remote=True):stopped='storage_partial';break
                last_remote_step=cursor;last_remote_time=time.monotonic()
    except TimeoutError:stopped='deadline_partial'
    finally:
        # Endpoint as well as last safe state survive a clean deadline. A
        # physical notebook kill before upload still loses local-only updates.
        checkpoint(remote=True)
    # Reserve time is used for durable saves and endpoint measurements.
    s.finalize_budget()
    row['finalization_status']='running';s.persist()
    try:
        final={}
        for label,params in (('last',q),('best',best)):
            from extent.efficiency_sources import tree_digest
            state_sha=tree_digest(jax.device_get(params));prefix=f'endpoint-{cursor}/{label}/state-{state_sha}'
            final[label]=dict(step=cursor if label=='last' else best_step,gains=gain_report(params),panels={},state_sha256=state_sha)
            for panel in ('A','B','C'):
                final[label]['panels'][panel]=s.task(f'{prefix}/{panel}',
                    lambda params=params,panel=panel:e.evaluate(params,tokens[panel],profile=True))
            final[label]['layer_profile']=s.task(f'{prefix}/layer-profile',
                lambda params=params:e.inspect(params,tokens['A'][:1]))
        final['best']['context']=context_report(s,p,e,best,tokens,f'endpoint-{cursor}/best/state-{final["best"]["state_sha256"]}')
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
        if args.experiment==120:s.result['status']=run_probes(s,p,tokens)
        else:s.result['status']=run_training(s,p,tokens)
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
    p.add_argument('--save-reserve-minutes',type=float,default=45)
    p.add_argument('--max-new-hf-gb',type=float,default=2)
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
            cmd=[sys.executable,'-u','-m','scripts.m3q_t4_scale_campaign','--experiment',str(args.experiment),
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
    status='failed' if failed else ('completed' if all(w['report']['status']=='completed' for w in workers) else 'deadline_partial')
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
