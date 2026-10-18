"""EXP147/148: EXP145 exact continuation/LR screen and repaired paired control with immutable sources and quiet parent notifications.
GPU dispatch uses fresh subprocesses; TPU executes in the notebook process.
"""
import argparse,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from dataclasses import replace
from importlib.metadata import version
from types import SimpleNamespace
import numpy as np
from extent.fold_continue_recovery import recipes
ROOT=Path(__file__).resolve().parents[1]
GPU_EXPERIMENTS=(147,148)
TPU_EXPERIMENTS=()
EVAL_STEPS=(0,128,512,1024,2048,2560,3072,3434,3584,4096,4608,5120,6144,8192,10240,12288,16384,20480,24576,28672,32768,40960,49152,65536,81920,98304,114688,131072)


def contract_for(experiment,worker=None,max_steps=65536):
    from scripts.m3q_reset_long_campaign import contract_for as original
    if experiment not in GPU_EXPERIMENTS:raise ValueError('EXP147/148 only')
    c=original(145,worker,max_steps)
    specs=recipes(experiment);specs=specs if worker is None else [specs[worker]]
    c.update(protocol='exp147-148-fold-continue-v1',experiment=experiment,recipes=specs,seed=123,
        horizon=max_steps,first_goal_new_steps=8192,
        clock='step GLOBAL original136 data cursor; NEW=step-source_step;145 current24965 with Adam12379; control13612586 with freshAdam0',
        gpu='two independent one-GPU workers/account;147 LR1+LR/3,148 repaired145 freshAdam control+LR/10',
        checkpoint='local128 NEW updates; remote120min+exit; own fold-continue-v1 namespace;4GB TOTAL/account,2GB/worker; current+Adam+selectedbest, frozen base never uploaded',
        optimizer='continuations restore all moments/counts incl Adam12379, no warmup restart; LR divisor applies to ALL groups; control freshAdam warm256',
        preparation='145 base is136 current12586 folded ONCE;145 current q+Adam from global24965; no extra folding/reset for continuations',
        continuation='147 LR1 vsLR/3;148 repaired original145 SAME-FACTORS control vs continuation LR/10;148 arms have DIFFERENT sources and are not a direct paired treatment comparison',
        comparison='three continuations compare matched NEW counts;148 control compares archived145 treatment at matched NEW counts from original136 source12586; retain145 prior12379-stage cost for continuing arms',
        execution_namespace='experiments/expID-fold-continue-v1; originals immutable; exact resume OWN state only',
        numerical_gate='source per-window NLL replay<=.03 on all160 A/B/C windows BEFORE updates; all raw chunk SHA checked; empty frozen GQA nodes retained',
        selection='source CURRENT baseline eligible, earlier inherited best not used for new selection; A64 selects registered endpoints, B/C descriptive; final separate',
        source_pin_sha=hashlib.sha256((ROOT/'results/EXP-147-148-source-pin.json').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        ancestor_pin_sha=hashlib.sha256((ROOT/'results/EXP-143-146-source-pin.json').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        limitations='single123 continuation screen, shared exposed panels; no proof of globally best LR, capacity causality or teacher recovery')
    for file in ('scripts/m3q_fold_continue_campaign.py','extent/fold_continue_recovery.py','extent/reset_long_fixed_sources.py'):
        c['code'][file]=hashlib.sha256((ROOT/file).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return c


def session_budget(experiment,max_wall_hours,save_reserve_minutes):
    """Validate this suite's hardware limits, including GPU worker invocations."""
    if experiment not in GPU_EXPERIMENTS:raise ValueError('EXP147/148 only')
    limit=7.
    if not np.isfinite(max_wall_hours) or not 0 < max_wall_hours <= limit:
        raise ValueError(f'EXP{experiment} wall budget must be >0 and <={limit:g}h')
    if not np.isfinite(save_reserve_minutes) or not 30 <= save_reserve_minutes < max_wall_hours*60:
        raise ValueError('save reserve must be >=30min and less than the wall budget')
    return max_wall_hours*3600-save_reserve_minutes*60,save_reserve_minutes


def session(args,contract,hub):
    # Reuse audited checkpoint/report transport, without the old 133 source recipe.
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from extent.recovery_durability import RecoveryCheckpointStore
    from extent.hf_artifact_sync import restore_artifact
    import jax
    s=Transport.__new__(Transport);s.args=args;s.contract=contract;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    suffix=f'/worker-{args.worker}' if args.worker is not None else ''
    s.prefix=f'experiments/exp{args.experiment}-fold-continue-v1'+suffix
    s.stem=f'extent-m3q-fold-continue-{args.experiment}'+(f'-worker-{args.worker}' if args.worker is not None else '')
    work,_=session_budget(args.experiment,args.max_wall_hours,args.save_reserve_minutes);s.deadline=s.begun+work;s.wall=s.begun+args.max_wall_hours*3600
    s.output=Path(args.output_dir);s.output.mkdir(parents=True,exist_ok=True);s.path=s.output/(s.stem+'.json');s.small=s.output/(s.stem+'-summary.txt')
    if not s.path.exists():restore_artifact(s.path,s.prefix+'/latest.json',hub)
    s.result=json.loads(s.path.read_text()) if s.path.exists() else dict(contract=contract,branches={},sessions=[],pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0))
    if s.result['contract']!=contract:raise ValueError('follow-up resume contract differs; do not overwrite old state')
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    if s.result.get('runtime',runtime)!=runtime:raise ValueError('follow-up math runtime changed')
    s.result['runtime']=runtime;s.result.pop('error',None)
    s.store=RecoveryCheckpointStore(Path(args.checkpoint_dir)/(s.stem),s.prefix+'/checkpoints',hub)
    s.next_sync=0.;s.invocation=dict(max_wall_hours=args.max_wall_hours,save_reserve_minutes=args.save_reserve_minutes,uploaded_bytes=0)
    s.result['sessions'].append(s.invocation)
    def emit(event):
        progress={name:r.get('step',0) for name,r in s.result['branches'].items()}
        message=f'Extent EXP{args.experiment}: {event}; status={s.result.get("status")}; steps={progress}; HF pending={len(s.result["pending_slots"])}'
        s.result.setdefault('events',[]).append(dict(event=event,telegram_sent=False))
        print(message,flush=True);s.persist(local_only=True)
    s.emit=emit
    return s


def run(args):
    import jax
    import jax.numpy as jnp
    from extent.config import load_config
    from extent.reset_followup import block_forward,Inspector
    from extent.fold_continue_recovery import prepare,restore_source,base_coordinates
    from extent.competitor_recovery import optimizer
    from extent.competitor_tpu import token_nll,training_batch,reconcile
    from extent.reset_streamed_gpu import make_step
    from extent.plateau_forensics import GuardedJit,plain,state_digest
    from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding,replicated_sharding
    from extent.reset_long_patient import Patient
    from extent.t4_scale_data import load_data
    from extent.hf_artifact_sync import artifact_config_from_env
    from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer
    cfg,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    c=contract_for(args.experiment,args.worker,args.max_steps);horizon=c['horizon'];eval_steps=c['eval_steps'];hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    s=session(args,c,hub);payloads={};bases={};layouts={};kernels={};evaluators={};specs={};teacher_params=None
    if s.result.get('status')=='completed':s.persist(force=True);return s.result
    s.result['status']='preflight';s.emit('started')
    try:
        if not s.persist(force=True):raise RuntimeError('HF report preflight failed')
        devices=jax.devices('gpu' if args.experiment in GPU_EXPERIMENTS else 'tpu')
        expected=1
        if len(devices)!=expected or jax.process_count()!=1:raise ValueError(f'expected{expected} local accelerator devices')
        mesh=create_v5e_mesh(devices);s.result['devices']=[dict(id=d.id,kind=d.device_kind,platform=d.platform) for d in devices]
        tokens,meta=load_data(Path(args.cache_dir)/'reset-data',s.check);s.result['data']=meta
        # Host source loading, not full model initialization on accelerator.
        patient=Patient(Path(args.cache_dir)/'reset-patient',hub,jax.devices('cpu')[0],s.check,seed=c['seed'])
        if patient.cfg!=cfg:raise ValueError('patient/config mismatch')
        columns=patient.columns;tc=patient.tc
        ancestor136=None
        if any(spec['source']=='145-fold' for spec in c['recipes']):ancestor136=base_coordinates(hub,Path(args.cache_dir)/'immutable',s.check)
        teacher_params={}
        for spec in c['recipes']:
            s.check();name=spec['name'];specs[name]=spec
            source_payload=pin=None
            source_payload,pin=restore_source(spec['source'],hub,Path(args.cache_dir)/'immutable',s.check)
            if s.result['runtime']!=pin['runtime']:raise ValueError('source runtime differs; refusing nominal exact resume')
            with jax.default_device(jax.devices('cpu')[0]):base,q,source_cursor=prepare(patient,spec,source_payload,pin,ancestor136)
            base=jax.device_put(base,named_sharding_tree(base,mesh));bases[name]=base
            layout=coordinate_layout(q,mesh);q=jax.device_put(q,layout);tx=optimizer(q,spec)
            state,state_layout=sharded_optimizer(tx,q,layout,mesh)
            if spec['mode']=='exact':state=jax.device_put(source_payload['optimizer'],state_layout)
            best=q;best_step=source_cursor if source_cursor else -1
            template=dict(parameters=q,optimizer=state,best=best,best_step=jnp.asarray(best_step,jnp.int32))
            layouts[name]=dict(parameters=layout,optimizer=state_layout,best=layout,best_step=replicated_sharding(mesh))
            saved=s.store.restore(name,dict(c,arm=name),template,layouts[name]);payload=saved[0] if saved else template
            cursor=saved[1]['step'] if saved else source_cursor
            row=s.result['branches'].setdefault(name,dict(step=cursor,source_step=source_cursor,history=[],evaluations=[]))
            if not saved and row.get('durable_step',source_cursor)>source_cursor:raise FileNotFoundError('durable continuation missing')
            if saved and row.get('durable_step',source_cursor)>cursor:raise ValueError('durable report cursor ahead of restored checkpoint')
            reconcile(row,cursor,int(payload['best_step']));payloads[name]=payload
            row['invocation_updates']=0;row['timing_samples']=[]
            row['source_adam_step']=pin.get('adam_step',source_cursor) if spec['mode']=='exact' else 0
            row['initialization']='restored moments/counts' if spec['mode']=='exact' else 'fresh Adam ONLY; source factors retained'
            row['coordinates']=sum(a.size for a in jax.tree.leaves(q));row['payload_bytes']=sum(a.size*a.dtype.itemsize for a in jax.tree.leaves(template))
            kernels[name]=make_step(cfg,tc,columns,spec,tx,block_fn=block_forward,return_gradients=False)
            kernels[name].check=s.check
            evaluators[name]=GuardedJit(lambda b,q,t,step,spec=spec:token_nll(cfg,b,q,t,columns,spec,step,block_fn=block_forward))
            def evaluate(q,windows,step,tail=None,function=evaluators[name],weights=base):
                losses=[]
                for t in windows:
                    s.check();v=np.asarray(function(weights,q,jax.device_put(t[None],batch_sharding(mesh)),jnp.asarray(step,jnp.float32)))[0]
                    if not np.all(np.isfinite(v)):raise FloatingPointError('nonfinite ordinary held-out NLL')
                    losses.append(float(np.mean(v if tail is None else v[-tail:])))
                return dict(nll=float(np.mean(losses)),windows=losses,targets_per_window=255 if tail is None else tail)
            if 'source_controls' not in row:
                control={p:evaluate(q,tokens[p],source_cursor) for p in ('A','B','C')}
                if pin is not None:
                    errors={p:float(np.max(np.abs(np.asarray(control[p]['windows'])-pin['reference'][p]['windows']))) for p in ('A','B','C')}
                    if max(errors.values())>.03:raise ValueError('immutable source NLL replay differs; refusing update')
                    row['source_replay']=dict(passed=True,per_panel_max_error=errors,tolerance=.03)
                row['source_controls']=control
            row['best_nll']=evaluate(payload['best'],tokens['A'],int(payload['best_step']))['nll'] if int(payload['best_step'])>=0 else None
            # Python callables are kept OUTSIDE JSON/checkpoint payloads.
            evaluators[name]=SimpleNamespace(kernel=evaluators[name],evaluate=evaluate,inspector=Inspector(cfg,base,columns,spec,s.check))
            if 'source_profile' not in row:row['source_profile']=evaluators[name].inspector.profile(q,tokens['A'][:1],source_cursor)
            s.persist()
            del template,q,state;gc.collect()
        del patient,source_payload,ancestor136;gc.collect()
        rows=s.result['branches'];s.result['status']='running';s.persist();last_sync=time.monotonic();last_notice=time.monotonic()
        def evaluate_endpoint(name,force=False):
            row=rows[name];payload=payloads[name];spec=specs[name];at=row['step']
            if not force and ((at-row['source_step'] not in eval_steps and at not in c['global_eval_steps']) or any(e['step']==at for e in row['evaluations'])):return
            ready=spec['ramp_steps']==0 or at>=spec['ramp_steps'];point=dict(step=at,new_steps=at-row['source_step'],selection_eligible=ready,panels={})
            if ready and at==row['source_step']:
                point['panels']=row['source_controls'];point['reuses_source_control']=True
            else:
                for panel in ('A','B','C') if ready else ('A',):
                    point['panels'][panel]=evaluators[name].evaluate(payload['parameters'],tokens[panel] if ready else tokens[panel][:16],at)
            if ready and (row['best_nll'] is None or point['panels']['A']['nll']<row['best_nll']):
                payload['best']=payload['parameters'];payload['best_step']=jnp.asarray(at,jnp.int32)
                row.update(best_nll=point['panels']['A']['nll'],best_step=at)
            if at-row['source_step'] in (4096,8192,16384,32768,65536,98304,131072):point['profile']=evaluators[name].inspector.profile(payload['parameters'],tokens['A'][:1],at)
            if not force:row['evaluations'].append(point)
            s.persist();return point
        try:
            while any(r['step']-r['source_step']<horizon and not r.get('failed') for r in rows.values()):
                eligible=[n for n,r in rows.items() if r['step']-r['source_step']<horizon and not r.get('failed')]
                name=min(eligible,key=lambda n:rows[n]['step']-rows[n]['source_step']);row=rows[name];payload=payloads[name];spec=specs[name]
                cursor=row['step'];block_size=128;origin=row['source_step'];target=min(origin+horizon,origin+(((cursor-origin)//block_size)+1)*block_size)
                try:
                    evaluate_endpoint(name)
                    while cursor<target:
                        s.check();t=training_batch(tokens['FIT'],cursor,133)
                        begun=time.monotonic();cost=s.result['cost'];cost['attempted_updates']+=1;cost['input_tokens']+=t.size;cost['ce_targets']+=t.shape[1]-1
                        if spec['alpha']:cost['teacher_input_tokens']+=t.size
                        outputs=kernels[name](payload['parameters'],payload['optimizer'],bases[name],teacher_params,
                            jax.device_put(t,batch_sharding(mesh)),jnp.asarray(cursor,jnp.float32))
                        proposed,after,health=outputs[:3]
                        seconds=time.monotonic()-begun;cost['training_seconds']=cost.get('training_seconds',0)+seconds;h=plain(health)
                        if 'first_update_health' not in row or not h['finite']:
                            row['first_update_health' if 'first_update_health' not in row else 'failed_health']=dict(step=cursor,seconds=seconds,health=h)
                            # The real proposal has not been accepted yet.
                            s.persist(force=True,checkpoints=False)
                        if not h['finite']:
                            row['failed']=dict(step=cursor,reason='nonfinite proposal rejected',failed_stages=[k for k,v in h['stages'].items() if not v['finite']]);break
                        payload['parameters']=jax.device_put(proposed,layouts[name]['parameters']);payload['optimizer']=jax.device_put(after,layouts[name]['optimizer'])
                        cursor+=1;row['step']=cursor;row.pop('final',None)
                        cost['accepted_updates']=cost.get('accepted_updates',0)+1;cost['accepted_input_tokens']=cost.get('accepted_input_tokens',0)+t.size;cost['accepted_ce_targets']=cost.get('accepted_ce_targets',0)+t.shape[0]*(t.shape[1]-1)
                        row['accepted_new_updates']=row.get('accepted_new_updates',0)+1;row['invocation_updates']+=1
                        row.setdefault('timing_samples',[]).append(seconds);row['timing_samples']=row['timing_samples'][-16:]
                        if row['invocation_updates']>=8:
                            median=float(np.median(row['timing_samples'][-6:]));remaining=max(0,c['first_goal_new_steps']-(cursor-row['source_step']))*median/3600
                            row['throughput']=dict(seconds_per_update=median,goal_new_steps=c['first_goal_new_steps'],estimated_training_hours_to_goal=remaining,estimated_training_hours_to_ceiling=max(0,horizon-(cursor-row['source_step']))*median/3600,ceiling=horizon,remaining_work_hours=max(0,s.deadline-time.monotonic())/3600,
                                caveat='recent training throughput only; evaluations/saves excluded; not a completion promise')
                        if row['invocation_updates']==8:s.emit(name+' throughput checked: '+str(row.get('throughput')))
                        if row['invocation_updates']<=8 or cursor%32==0:
                            row['history'].append(dict(step=cursor,seconds=seconds,**h));s.persist(local_only=True)
                        evaluate_endpoint(name)
                        if cursor-row['source_step']==c['first_goal_new_steps']:
                            row['first_goal_reached']=True
                            s.emit(str(c['first_goal_new_steps'])+' additional goal reached'+('; continuing within wall budget' if horizon>c['first_goal_new_steps'] else ''))
                        if (cursor-row['source_step'])%block_size==0:s.checkpoint(name,payload,cursor)
                    s.checkpoint(name,payload,cursor)
                except TimeoutError:raise
                if time.monotonic()-last_sync>=7200:
                    s.persist(force=True);last_sync=time.monotonic()
                    if s.result.get('storage_error'):raise TimeoutError('upload cap; progress saved locally')
                if time.monotonic()-last_notice>=1800:s.emit('progress');last_notice=time.monotonic()
            s.result['status']='arm_failed' if any(r.get('failed') for r in rows.values()) else 'completed'
        except TimeoutError as exc:s.result['status']='deadline_partial';s.result['stop_reason']=str(exc)
        # Source/current/Adam are saved BEFORE final diagnostics.
        for name,payload in payloads.items():s.checkpoint(name,payload,rows[name]['step'])
        s.persist(force=True);s.deadline=s.wall-180
        try:
            for name,row in rows.items():
                payload=payloads[name];ev=evaluators[name].evaluate;at=row['step']
                current_hash=state_digest(payload['parameters']);best_hash=state_digest(payload['best']);best_at=int(payload['best_step'])
                signature=(current_hash,best_hash,at,best_at)
                if tuple(row.get('final_signature',()))!=signature:row.pop('final',None)
                row['final_signature']=signature;final=row.setdefault('final',{})
                for label,q,step in [('last',payload['parameters'],at)]+([('best',payload['best'],best_at)] if best_at>=0 else []):
                    point=final.setdefault(label,dict(step=step,panels={},context={}))
                    for panel in ('A','B','C'):
                        if panel not in point['panels']:point['panels'][panel]=ev(q,tokens[panel],step);s.persist()
                    if 'profile' not in point:point['profile']=evaluators[name].inspector.profile(q,tokens['A'][:1],step);s.persist()
                    for length in (128,256,512,1024):
                        if str(length) not in point['context']:
                            windows=tokens['B'].reshape(8,1024)[:,-length:]
                            point['context'][str(length)]=ev(q,windows,step,tail=127);s.persist()
                row['finalization_status']='completed'
        except TimeoutError:
            s.result['finalization_status']='deadline_partial'
            if s.result['status']=='completed':s.result['status']='finalization_partial'
        s.result['accelerator_memory']=[plain(d.memory_stats() or {}) for d in devices]
    except TimeoutError as exc:s.result.update(status='deadline_partial',stop_reason=str(exc))
    except Exception as exc:
        s.result.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
    finally:
        for name,payload in payloads.items():s.checkpoint(name,payload,s.result['branches'][name]['step'])
        s.persist(force=True)
        if s.result.get('storage_error'):s.result['status']='storage_partial';s.persist(local_only=True)
        s.emit('finished and saved' if not s.result['pending_slots'] and not s.result.get('report_sync_error') else 'finished; HF pending')
    if s.result['status']=='failed':raise RuntimeError('follow-up stopped; see saved report')
    return s.result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=int,choices=GPU_EXPERIMENTS,required=True)
    p.add_argument('--worker',type=int)
    p.add_argument('--max-wall-hours',type=float)
    p.add_argument('--max-steps',type=int,default=65536,help='Ceiling in ADDITIONAL updates after pinned source')
    p.add_argument('--save-reserve-minutes',type=float,default=30.)
    p.add_argument('--max-new-hf-gb',type=float)
    p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--cache-dir',default='/kaggle/working/extent-reset-cache')
    p.add_argument('--checkpoint-dir',default='/kaggle/working/extent-reset-states')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.max_wall_hours is None:args.max_wall_hours=6.8
    if args.max_steps<32768:raise ValueError('ceiling must be at least32768')
    if args.max_new_hf_gb is None:args.max_new_hf_gb=4.
    session_budget(args.experiment,args.max_wall_hours,args.save_reserve_minutes)
    if args.save_reserve_minutes<30 or not np.isfinite(args.max_new_hf_gb) or args.max_new_hf_gb<=0:raise ValueError('invalid reserve or upload cap')
    if args.worker is not None and (args.experiment not in GPU_EXPERIMENTS or not 0<=args.worker<len(recipes(args.experiment))):raise ValueError('invalid GPU worker')
    if args.plan_only:
        value=contract_for(args.experiment,args.worker,args.max_steps);print(json.dumps(value,indent=2));return value
    if args.worker is not None:
        args.telegram=False
        return run(args)
    from extent.reset_quiet import ParentNotifier
    from scripts.m3q_t4_campaign import write
    notice=ParentNotifier(args.telegram,args.experiment)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);started=time.monotonic();children=[];handles=[]
    notice.emit('started','started; two GPU workers; quiet notifications')
    try:
        for worker in range(2):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(worker),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
            cmd=[sys.executable,'-u','-m','scripts.m3q_fold_continue_campaign','--experiment',str(args.experiment),'--worker',str(worker),
                '--max-wall-hours',str(args.max_wall_hours),'--max-steps',str(args.max_steps),'--save-reserve-minutes',str(args.save_reserve_minutes),'--max-new-hf-gb',str(args.max_new_hf_gb/2),
                '--output-dir',args.output_dir,'--cache-dir',str(Path(args.cache_dir)/f'worker-{worker}'),'--checkpoint-dir',args.checkpoint_dir,'--no-telegram']
            handle=(output/f'exp{args.experiment}-fold-continue-worker-{worker}.log').open('w');handles.append(handle)
            children.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
        last_check=0.
        while any(child.poll() is None for child in children):
            now=time.monotonic()
            if now-last_check>=60:
                progress={};bad=False
                for worker,child in enumerate(children):
                    path=output/f'extent-m3q-fold-continue-{args.experiment}-worker-{worker}.json'
                    try:r=json.loads(path.read_text()) if path.exists() else {}
                    except (OSError,ValueError):r={}
                    progress[worker]={n:dict(step=b.get('step'),new_steps=b.get('step',0)-b.get('source_step',0),best=b.get('best_nll')) for n,b in r.get('branches',{}).items()}
                    bad=bad or child.poll() not in (None,0) or r.get('status') in ('failed','arm_failed')
                if bad:notice.emit('failure','worker stopped; inspect aggregate/log; '+str(progress))
                notice.emit('progress',str(progress));last_check=now
            time.sleep(5)
    except Exception as exc:
        notice.emit('failure',type(exc).__name__+': '+str(exc));raise
    finally:
        for child in children:child.wait()
        for handle in handles:handle.close()
    rows=[]
    for worker,child in enumerate(children):
        path=output/f'extent-m3q-fold-continue-{args.experiment}-worker-{worker}.json'
        report=json.loads(path.read_text()) if path.exists() else dict(status='missing_report')
        log=output/f'exp{args.experiment}-fold-continue-worker-{worker}.log'
        tail=log.read_text(encoding='utf-8',errors='replace')[-16000:] if log.exists() else ''
        bad=child.returncode!=0 or report.get('status') in ('failed','arm_failed')
        rows.append(dict(worker=worker,exit=child.returncode,report=report,log_tail=tail if bad else ''))
        if bad:print(f'worker{worker} exit={child.returncode}; '+str(report.get('error'))+'\n'+tail,flush=True)
    statuses={w['report']['status'] for w in rows}
    status='failed' if any(w['exit']!=0 for w in rows) or statuses&{'failed','arm_failed','missing_report'} else 'completed' if statuses=={'completed'} else 'storage_partial' if 'storage_partial' in statuses else 'finalization_partial' if 'finalization_partial' in statuses else 'deadline_partial'
    result=dict(experiment=args.experiment,status=status,wall_hours=(time.monotonic()-started)/3600,workers=rows,notifications=notice.events,notification_policy='parent-only,4h progress,max5')
    notice.emit('finished',status+'; download extent-m3q-fold-continue-'+str(args.experiment)+'.json')
    write(output/f'extent-m3q-fold-continue-{args.experiment}.json',result)
    if status=='failed':raise RuntimeError('worker failure/rejected arm; see original traceback in aggregate/log')
    return result


if __name__=='__main__':main()
