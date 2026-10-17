"""EXP135-141: reset follow-ups; dual-GPU continuation and eight-chip TPU controls.
GPU dispatch uses fresh subprocesses; TPU executes in the notebook process.
"""
import argparse,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from dataclasses import replace
from importlib.metadata import version
from types import SimpleNamespace
import numpy as np
from extent.reset_followup import recipes
ROOT=Path(__file__).resolve().parents[1]
GPU_EXPERIMENTS=(135,136,137,138)
TPU_EXPERIMENTS=(139,140,141)
EVAL_STEPS=(0,128,512,1024,2048,2560,3072,3434,3584,4096,4608,5120,6144,8192,10240,12288,16384,20480,24576,28672,32768)


def contract_for(experiment,worker=None):
    from dataclasses import asdict
    from extent.config import load_config
    model,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    specs=recipes(experiment)
    if worker is not None:specs=[specs[worker]]
    files=('scripts/m3q_reset_followup_campaign.py','extent/reset_followup.py','extent/competitor_tpu.py',
        'extent/competitor_recovery.py','extent/pregate_recovery.py','extent/dynamic_recovery.py',
        'extent/t4_patient.py','extent/stable_gradient_clip.py','extent/layers/mamba3.py','extent/layers/common.py',
        'scripts/m3q_peer_tpu_campaign.py','extent/mamba3_transplant.py','extent/sharding.py',
        'extent/recovery_durability.py','extent/chunked_checkpoint.py','extent/t4_scale_data.py','config/hybrid_1_7b_gqa_v5e8.yaml')
    value=dict(protocol='exp135-141-reset-v1',experiment=experiment,worker=worker,seed=123,model=asdict(model),
        recipes=specs,horizon=32768,eval_steps=list(EVAL_STEPS),sequence_length=256,batch_size=1,
        gradient='one unchanged training window/update; full-chain gradient, no data-parallel batch increase',
        checkpoint='local128(R32)/512(dense); remote120min+exit, also4096 updates in R32-only sessions; current,Adam,best; frozen base reconstructed, never uploaded',
        optimizer='stableclip1; Adam .9/.999 eps1e-8; Mamba3e-4,norm1e-4; warmup256; fixed LR thereafter; no MLP in new comparisons',
        continuation='135 restores current3434,Adam,schedule and data cursor; 138/139 fold current3434 but RESET Adam in ALL paired arms',
        preparation='140/141 compare clean balanced QKVO lift vs EXP072 ONPOLICY, both BEFORE EXP111 corrections; zero-effect R32 init or dense Mamba; same fully-active norm+tanh+combined reset initialization; no ramp, dense dynamics not overwritten during training',
        selection='A64 at registered full-strength steps; B/C never select; arbitrary last reported separately',
        gpu='135 single trajectory model-sharded across BOTH GPUs;136-138 independent paired workers; resident frozen weights; tiled vocabulary; full rematerialized decoder',
        evaluation='source,last,best A/B/C; context128/256/512/1024 with same last127 targets; source replay tolerance.03 per window; first real proposal health persisted before acceptance',
        limitations='single-seed screening, exposed panels; physical throughput and full-scale HBM not locally established; partial sessions resume; no promise of32k in one session',
        source130_pin=hashlib.sha256((ROOT/'results/EXP-135-source-pin.json').read_bytes()).hexdigest(),
        source111_pin=hashlib.sha256((ROOT/'results/EXP-114-source-pin.json').read_bytes()).hexdigest(),
        code={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    return json.loads(json.dumps(value))


def session(args,contract,hub):
    # Reuse audited checkpoint/report transport, without the old 133 source recipe.
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from scripts.m3q_batch_recovery_campaign import session_budget
    from extent.recovery_durability import RecoveryCheckpointStore
    from extent.hf_artifact_sync import restore_artifact
    import jax
    s=Transport.__new__(Transport);s.args=args;s.contract=contract;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    suffix=f'/worker-{args.worker}' if args.worker is not None else ''
    s.prefix=f'experiments/exp{args.experiment}-reset'+suffix
    s.stem=f'extent-m3q-reset-{args.experiment}'+(f'-worker-{args.worker}' if args.worker is not None else '')
    work,_=session_budget(args.max_wall_hours,args.save_reserve_minutes);s.deadline=s.begun+work;s.wall=s.begun+args.max_wall_hours*3600
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
        from scripts.qwen_extended_horizon_campaign import _safe_notify
        progress={name:r.get('step',0) for name,r in s.result['branches'].items()}
        message=f'Extent EXP{args.experiment}: {event}; status={s.result.get("status")}; steps={progress}; HF pending={len(s.result["pending_slots"])}'
        s.result.setdefault('notifications',[]).append(dict(event=event,**_safe_notify(args.telegram,message)))
        print(message,flush=True);s.persist(local_only=True)
    s.emit=emit
    return s


def run(args):
    import jax
    import jax.numpy as jnp
    from extent.config import load_config
    from extent.reset_followup import source130,clean_transplant,prepare,block_forward,Inspector
    from extent.competitor_recovery import optimizer
    from extent.competitor_tpu import make_step,token_nll,training_batch,reconcile
    from extent.plateau_forensics import GuardedJit,plain,state_digest
    from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding,replicated_sharding
    from extent.t4_patient import Patient
    from extent.t4_scale_data import load_data
    from extent.hf_artifact_sync import artifact_config_from_env
    from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer
    from extent.context_recovery import tiled_nll
    cfg,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    c=contract_for(args.experiment,args.worker);hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    s=session(args,c,hub);payloads={};bases={};layouts={};kernels={};evaluators={};specs={};teacher_params=None
    if s.result.get('status')=='completed':s.persist(force=True);return s.result
    s.result['status']='preflight';s.emit('started')
    try:
        if not s.persist(force=True):raise RuntimeError('HF report preflight failed')
        devices=jax.devices('gpu' if args.experiment in GPU_EXPERIMENTS else 'tpu')
        expected=2 if args.experiment==135 else 1 if args.experiment in GPU_EXPERIMENTS else 8
        if len(devices)!=expected or jax.process_count()!=1:raise ValueError(f'expected{expected} local accelerator devices')
        mesh=create_v5e_mesh(devices);s.result['devices']=[dict(id=d.id,kind=d.device_kind,platform=d.platform) for d in devices]
        tokens,meta=load_data(Path(args.cache_dir)/'reset-data',s.check);s.result['data']=meta
        # Host source loading, not full model initialization on accelerator.
        patient=Patient(Path(args.cache_dir)/'reset-patient',hub,jax.devices('cpu')[0],s.check)
        if patient.cfg!=cfg:raise ValueError('patient/config mismatch')
        columns=patient.columns;tc=patient.tc
        source_payload=pin=None
        if any(v['start'].startswith('130') for v in c['recipes']):source_payload,pin=source130(patient,hub,Path(args.cache_dir)/'immutable',s.check)
        if args.experiment==135:
            source_runtime=pin['source_runtime'];runtime=s.result['runtime']
            if runtime['matmul_precision'] not in (None,'None') or any(runtime[k]!=v for k,v in source_runtime['versions'].items()):raise ValueError('EXP130 runtime mismatch')
        clean=reports=None
        if any(v['start']=='clean' for v in c['recipes']):clean,reports=clean_transplant(patient);s.result['clean_initialization']=reports
        teacher_params=jax.device_put(patient.teacher,named_sharding_tree(patient.teacher,mesh)) if any(v['alpha'] for v in c['recipes']) else {}
        for spec in c['recipes']:
            s.check();name=spec['name'];specs[name]=spec
            with jax.default_device(jax.devices('cpu')[0]):base,q,source_cursor=prepare(patient,spec,source_payload,clean)
            base=jax.device_put(base,named_sharding_tree(base,mesh));bases[name]=base
            layout=coordinate_layout(q,mesh);q=jax.device_put(q,layout);tx=optimizer(q,spec)
            state,state_layout=sharded_optimizer(tx,q,layout,mesh)
            if spec['start']=='130-exact':state=jax.device_put(source_payload['optimizer'],state_layout)
            best=q;best_step=source_cursor if source_cursor else -1
            if spec['start']=='130-exact':best=jax.device_put(source_payload['best'],layout);best_step=int(source_payload['best_step'])
            template=dict(parameters=q,optimizer=state,best=best,best_step=jnp.asarray(best_step,jnp.int32))
            layouts[name]=dict(parameters=layout,optimizer=state_layout,best=layout,best_step=replicated_sharding(mesh))
            saved=s.store.restore(name,dict(c,arm=name),template,layouts[name]);payload=saved[0] if saved else template
            cursor=saved[1]['step'] if saved else source_cursor
            row=s.result['branches'].setdefault(name,dict(step=cursor,source_step=source_cursor,history=[],evaluations=[]))
            if not saved and row.get('durable_step',source_cursor)>source_cursor:raise FileNotFoundError('durable continuation missing')
            reconcile(row,cursor,int(payload['best_step']));payloads[name]=payload
            row['invocation_updates']=0;row['timing_samples']=[]
            row['coordinates']=sum(a.size for a in jax.tree.leaves(q));row['payload_bytes']=sum(a.size*a.dtype.itemsize for a in jax.tree.leaves(template))
            kernels[name]=GuardedJit(make_step(cfg,tc,columns,spec,tx,block_fn=block_forward))
            evaluators[name]=GuardedJit(lambda q,t,step,base=base,spec=spec:token_nll(cfg,base,q,t,columns,spec,step,block_fn=block_forward))
            def evaluate(q,windows,step,tail=None,function=evaluators[name]):
                losses=[]
                for t in windows:
                    s.check();v=np.asarray(function(q,jax.device_put(t[None],batch_sharding(mesh)),jnp.asarray(step,jnp.float32)))[0]
                    if not np.all(np.isfinite(v)):raise FloatingPointError('nonfinite ordinary held-out NLL')
                    losses.append(float(np.mean(v if tail is None else v[-tail:])))
                return dict(nll=float(np.mean(losses)),windows=losses,targets_per_window=255 if tail is None else tail)
            if 'source_controls' not in row:
                control={p:evaluate(q,tokens[p],source_cursor) for p in ('A','B','C')}
                if pin is not None:
                    errors={p:float(np.max(np.abs(np.asarray(control[p]['windows'])-pin['reference'][p]['windows']))) for p in ('A','B','C')}
                    if max(errors.values())>.03:raise ValueError('EXP130 replay differs; refusing update')
                    row['source_replay']=dict(passed=True,per_panel_max_error=errors,tolerance=.03)
                row['source_controls']=control
            row['best_nll']=evaluate(payload['best'],tokens['A'],int(payload['best_step']))['nll'] if int(payload['best_step'])>=0 else None
            # Python callables are kept OUTSIDE JSON/checkpoint payloads.
            evaluators[name]=SimpleNamespace(kernel=evaluators[name],evaluate=evaluate,inspector=Inspector(cfg,base,columns,spec,s.check))
            if 'source_profile' not in row:row['source_profile']=evaluators[name].inspector.profile(q,tokens['A'][:1],source_cursor)
            s.persist()
            del template,q,state;gc.collect()
        del patient,clean,source_payload;gc.collect()
        rows=s.result['branches'];s.result['status']='running';s.persist();last_sync=time.monotonic();last_notice=time.monotonic();dense_campaign=any(v['dense'] for v in specs.values())
        def evaluate_endpoint(name,force=False):
            row=rows[name];payload=payloads[name];spec=specs[name];at=row['step']
            if not force and (at not in EVAL_STEPS or any(e['step']==at for e in row['evaluations'])):return
            ready=spec['ramp_steps']==0 or at>=spec['ramp_steps'];point=dict(step=at,selection_eligible=ready,panels={})
            if ready and at==row['source_step']:
                point['panels']=row['source_controls'];point['reuses_source_control']=True
            else:
                for panel in ('A','B','C') if ready else ('A',):
                    point['panels'][panel]=evaluators[name].evaluate(payload['parameters'],tokens[panel] if ready else tokens[panel][:16],at)
            if ready and (row['best_nll'] is None or point['panels']['A']['nll']<row['best_nll']):
                payload['best']=payload['parameters'];payload['best_step']=jnp.asarray(at,jnp.int32)
                row.update(best_nll=point['panels']['A']['nll'],best_step=at)
            if at in (4096,8192,16384,32768):point['profile']=evaluators[name].inspector.profile(payload['parameters'],tokens['A'][:1],at)
            if not force:row['evaluations'].append(point)
            s.persist();return point
        try:
            while any(r['step']<32768 and not r.get('failed') for r in rows.values()):
                eligible=[n for n,r in rows.items() if r['step']<32768 and not r.get('failed')]
                name=min(eligible,key=lambda n:rows[n]['step']-rows[n]['source_step']);row=rows[name];payload=payloads[name];spec=specs[name]
                cursor=row['step'];block_size=512 if spec['dense'] else 128;target=min(32768,((cursor//block_size)+1)*block_size)
                try:
                    evaluate_endpoint(name)
                    while cursor<target:
                        s.check();t=training_batch(tokens['FIT'],cursor,133)
                        begun=time.monotonic();cost=s.result['cost'];cost['attempted_updates']+=1;cost['input_tokens']+=t.size;cost['ce_targets']+=t.shape[1]-1
                        if spec['alpha']:cost['teacher_input_tokens']+=t.size
                        proposed,after,health=kernels[name](payload['parameters'],payload['optimizer'],bases[name],teacher_params,
                            jax.device_put(t,batch_sharding(mesh)),jnp.asarray(cursor,jnp.float32))
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
                            median=float(np.median(row['timing_samples'][-6:]));remaining=(32768-cursor)*median/3600
                            row['throughput']=dict(seconds_per_update=median,estimated_training_hours_to32768=remaining,remaining_work_hours=max(0,s.deadline-time.monotonic())/3600,
                                caveat='recent training throughput only; evaluations/saves excluded; not a completion promise')
                        if row['invocation_updates']==8:s.emit(name+' throughput checked: '+str(row.get('throughput')))
                        if row['invocation_updates']<=8 or cursor%32==0:
                            row['history'].append(dict(step=cursor,seconds=seconds,**h));s.persist(local_only=True)
                        evaluate_endpoint(name)
                        if cursor%block_size==0:s.checkpoint(name,payload,cursor)
                    s.checkpoint(name,payload,cursor)
                except TimeoutError:raise
                if time.monotonic()-last_sync>=7200 or (not dense_campaign and any(r['step']-r.get('durable_step',r['source_step'])>=4096 for r in rows.values())):
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
    p.add_argument('--experiment',type=int,choices=range(135,142),required=True)
    p.add_argument('--worker',type=int)
    p.add_argument('--max-wall-hours',type=float)
    p.add_argument('--save-reserve-minutes',type=float,default=60.)
    p.add_argument('--max-new-hf-gb',type=float)
    p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--cache-dir',default='/kaggle/working/extent-reset-cache')
    p.add_argument('--checkpoint-dir',default='/kaggle/working/extent-reset-states')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.max_wall_hours is None:args.max_wall_hours=11.5 if args.experiment in GPU_EXPERIMENTS else 8.5
    if args.max_new_hf_gb is None:args.max_new_hf_gb=40. if args.experiment==141 else 24. if args.experiment==139 else 4.
    from scripts.m3q_batch_recovery_campaign import session_budget
    session_budget(args.max_wall_hours,args.save_reserve_minutes)
    if args.save_reserve_minutes<30 or not np.isfinite(args.max_new_hf_gb) or args.max_new_hf_gb<=0:raise ValueError('invalid reserve or upload cap')
    if args.worker is not None and (args.experiment not in GPU_EXPERIMENTS or not 0<=args.worker<len(recipes(args.experiment))):raise ValueError('invalid GPU worker')
    if args.plan_only:
        value=contract_for(args.experiment,args.worker);print(json.dumps(value,indent=2));return value
    if args.experiment in TPU_EXPERIMENTS or args.worker is not None:return run(args)
    # Fresh processes see exactly the assigned GPUs, regardless of notebook JAX state.
    from scripts.m3q_t4_campaign import notify,write
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);started=time.time();children=[];handles=[]
    workers=1 if args.experiment==135 else 2
    notify(args,f'Extent EXP{args.experiment}: started; '+('both GPUs on one trajectory' if workers==1 else 'paired GPU workers'))
    for worker in range(workers):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1' if args.experiment==135 else str(worker),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
        cmd=[sys.executable,'-u','-m','scripts.m3q_reset_followup_campaign','--experiment',str(args.experiment),'--worker',str(worker),
            '--max-wall-hours',str(args.max_wall_hours),'--save-reserve-minutes',str(args.save_reserve_minutes),'--max-new-hf-gb',str(args.max_new_hf_gb/workers),
            '--output-dir',args.output_dir,'--cache-dir',str(Path(args.cache_dir)/f'worker-{worker}'),'--checkpoint-dir',args.checkpoint_dir,
            '--telegram' if args.telegram else '--no-telegram']
        handle=(output/f'exp{args.experiment}-worker-{worker}.log').open('w');handles.append(handle)
        children.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
    try:
        for child in children:child.wait()
    finally:
        for handle in handles:handle.close()
    rows=[]
    for worker,child in enumerate(children):
        path=output/f'extent-m3q-reset-{args.experiment}-worker-{worker}.json'
        report=json.loads(path.read_text()) if path.exists() else dict(status='missing_report')
        rows.append(dict(worker=worker,exit=child.returncode,report=report))
    statuses={w['report']['status'] for w in rows}
    status='failed' if any(w['exit']!=0 for w in rows) or 'failed' in statuses else 'completed' if statuses=={'completed'} else 'storage_partial' if 'storage_partial' in statuses else 'arm_failed' if 'arm_failed' in statuses else 'finalization_partial' if 'finalization_partial' in statuses else 'deadline_partial'
    result=dict(experiment=args.experiment,status=status,wall_hours=(time.time()-started)/3600,workers=rows)
    write(output/f'extent-m3q-reset-{args.experiment}.json',result)
    notify(args,f'Extent EXP{args.experiment}: {status}; download extent-m3q-reset-{args.experiment}.json')
    if status=='failed':raise RuntimeError('GPU worker failed; aggregate report and logs saved')
    return result


if __name__=='__main__':main()
