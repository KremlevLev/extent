"""EXP111-113: recovery efficiency, LR pilots and read-only layer diagnostics."""
import argparse,gc,hashlib,json,sys,time,traceback
from importlib.metadata import version
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from extent import HybridForCausalLM
from extent.config import load_config
from extent.campaign_checkpoint import write_json_atomic
from extent.recovery_durability import RecoveryCheckpointStore,cloud_roundtrip
from extent.hf_artifact_sync import artifact_config_from_env,restore_artifact,upload_artifacts_together
from extent.decoder_recovery import split_decoder,forward_parameters
from extent.recovery_subspace import coordinates_finite
from extent.initialization import initialize_sharded_optimizer_state
from extent.sharding import create_v5e_mesh,batch_sharding,named_sharding_tree,replicated_sharding
from jax.sharding import NamedSharding,PartitionSpec as P
from extent.efficiency_recovery import TOTAL_STEPS,PILOT_STEPS,SEEDS,PEAKS,DATA,PROBE_STEPS,load_tokens,token_manifest,schedule,optimizer,make_step,choose_lr,crossing_intervals,cost_totals,pilot_parity
from extent.efficiency_sources import load_sources
from extent.efficiency_diagnostics import Probes,update_metrics,json_safe
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout,memory_estimate,evaluate
from scripts.m3q_batch_recovery_campaign import session_budget
from scripts.qwen_extended_horizon_campaign import _safe_notify
ROOT=Path(__file__).resolve().parents[1]

def contract_for(config,experiment):
    files=('scripts/m3q_efficiency_campaign.py','extent/efficiency_recovery.py','extent/efficiency_sources.py',
        'extent/efficiency_diagnostics.py','extent/batch_recovery.py','extent/decoder_recovery.py',
        'extent/recovery_subspace.py','extent/calibration_data.py','extent/stable_gradient_clip.py',
        'extent/recovery_durability.py','extent/chunked_checkpoint.py','extent/config.py',
        'extent/model.py','extent/layers/mamba3.py','extent/layers/common.py',
        'extent/layers/mla.py','extent/layers/rorope_bkv.py','extent/qwen3_teacher.py',
        'extent/qwen_source.py','extent/full_model_distillation.py','extent/initialization.py',
        'extent/sharding.py','extent/layer_clip_recovery.py',
        'scripts/m3q_subspace_engine.py','scripts/m3q_full_depth_sequential_confirmation.py',
        'scripts/m3q_sequential_recovery_campaign.py','scripts/m3q_paper_decoder_recovery_campaign.py',
        'scripts/qwen17_full_model_distill_campaign.py')
    return json.loads(json.dumps(dict(experiment=experiment,model=asdict(config),seeds=list(SEEDS),
        source=json.loads((ROOT/'results/EXP-111-113-source-pin.json').read_text()),
        random_init='native PRNG seed, only Mamba' if experiment==113 else None,
        data=[list(x) for x in DATA],pg19=[393216,128],
        data_manifest=json.loads((ROOT/'results/EXP-111-113-data-preflight.json').read_text()),
        total_steps=TOTAL_STEPS,pilot_steps=0 if experiment==111 else PILOT_STEPS,
        lr_candidates={'MAIN':3e-4} if experiment==111 else PEAKS,
        optimizer=dict(name='Adam',betas=[.9,.999],eps=1e-8,decay=0,global_clip=1,warmup=0 if experiment==111 else 128),
        selection='mean validation BOTH seeds; no test selection; selected pilots replayed and charged',
        pipeline_preparation_cost='UNKNOWN; no total-pipeline speed assertion',
        implementation={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in files})))

def expected_tasks(result,experiment):
    if experiment!=111 and not result.get('selected_lr'):
        return [(seed,arm,PILOT_STEPS,True) for arm in PEAKS for seed in SEEDS]
    return [(seed,'MAIN',target,False) for target in range(4096,TOTAL_STEPS+1,4096)
            for seed in (SEEDS if (target//4096)%2 else SEEDS[::-1])]

def completed(result):
    return all(result.get('branches',{}).get(str(s),{}).get('MAIN',{}).get('step')==TOTAL_STEPS and 'final' in result['branches'][str(s)]['MAIN'] and not result['branches'][str(s)]['MAIN'].get('failed') for s in SEEDS)

def summarize(result):
    branches=result.get('branches',{})
    gaps={}
    for seed,rows in branches.items():
        final=rows.get('MAIN',{}).get('final',{})
        gaps[seed]={n:m['nll']-result['original_qwen'][n]['nll'] for n,m in final.items() if n in result.get('original_qwen',{})}
    return dict(status=result.get('status'),gate=None,complete=completed(result),selected_lr=result.get('selected_lr'),
        progress={s:{a:dict(step=r.get('step',0),durable_step=r.get('durable_step'),failed=r.get('failed'),metrics_only_pilot=r.get('metrics_only_pilot',False)) for a,r in rows.items()} for s,rows in branches.items()},
        crossings={s:{str(t):crossing_intervals(rows.get('MAIN',{}).get('diagnostics',[]),t) for t in (12,10,8,7.5,7)} for s,rows in branches.items()},
        final_gap_to_qwen=gaps,pending=result.get('pending_slots',[]),cost=cost_totals(branches),preparation_cost='UNKNOWN')

def upload_requirement(store,slots):
    missing=[slot for slot in slots if not (store.directory(slot)/'checkpoint.json').exists()]
    if missing:raise FileNotFoundError(f'pending local states unavailable: {missing}; sync-only needs the original runtime')
    return sum(json.loads((store.directory(slot)/'checkpoint.json').read_text())['payload_bytes'] for slot in slots)

def main(argv=None,*,experiment=111):
    if experiment not in (111,112,113):raise ValueError('unregistered experiment')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='/kaggle/working/output')
    parser.add_argument('--state-dir',default=f'/kaggle/working/extent-exp{experiment}-state')
    parser.add_argument('--checkpoint-dir',default=f'/dev/shm/extent-exp{experiment}-checkpoints')
    parser.add_argument('--dataset-cache-dir',default='/kaggle/working/extent-dataset-cache')
    parser.add_argument('--qwen-cache-dir',default='/kaggle/working/qwen3-exp104-weights')
    parser.add_argument('--max-wall-hours',type=float,default=8.8)
    parser.add_argument('--save-reserve-minutes',type=float,default=90)
    parser.add_argument('--max-new-hf-gb',type=float,default=2 if experiment==111 else 35)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--data-preflight-only',action='store_true')
    parser.add_argument('--sync-only',action='store_true')
    parser.add_argument('--fp32-probe',action='store_true')
    parser.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args(sys.argv[1:] if argv is None else argv)
    if not np.isfinite(args.max_new_hf_gb) or args.max_new_hf_gb<=0:raise ValueError('positive upload budget required')
    training_seconds,reserve=session_budget(args.max_wall_hours,args.save_reserve_minutes)
    config,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    if args.data_preflight_only:
        m=token_manifest(load_tokens(args.dataset_cache_dir));write_json_atomic(ROOT/'results/EXP-111-113-data-preflight.json',m);return m
    contract=contract_for(config,experiment)
    if args.plan_only:
        plan=dict(contract=contract,wall_hours=args.max_wall_hours,training_deadline_hours=training_seconds/3600,
            max_new_hf_bytes=int(args.max_new_hf_gb*1e9),checkpoint_policy='local during training; at most one remotely uploaded generation per active slot per invocation',
            fp32_probe=args.fp32_probe,pilot_replay_charged=True)
        print(json.dumps(plan,indent=2));return plan
    begun=time.monotonic();deadline=begun+training_seconds;wall_deadline=begun+args.max_wall_hours*3600
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    prefix=f'experiments/exp{experiment}-recovery-efficiency';stem=f'extent-m3q-efficiency-{experiment}'
    path=output/f'{stem}.json';summary_path=output/f'{stem}-summary.txt'
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF checkpoint secrets required')
    if not path.exists():restore_artifact(path,f'{prefix}/latest.json',hub)
    result=json.loads(path.read_text()) if path.exists() else dict(contract=contract,branches={},sessions=[],pending_slots=[])
    if result['contract']!=contract:raise ValueError('scientific resume contract mismatch')
    runtime=dict(versions={name:version(name) for name in ('jax','jaxlib','flax','optax')},
        matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    if not args.sync_only and result.get('runtime') is not None and result['runtime']!=runtime:
        raise ValueError('optimizer/math runtime changed; continuation not authorized by scientific contract')
    result.setdefault('runtime',runtime)
    session=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),max_wall_hours=args.max_wall_hours,
        save_reserve_minutes=reserve,max_new_hf_gb=args.max_new_hf_gb,uploaded_bytes=0)
    result['sessions'].append(session);result.pop('error',None)
    store=RecoveryCheckpointStore(args.checkpoint_dir,f'{prefix}/checkpoints',hub)
    sync_next=0.
    def persist(remote=False):
        nonlocal sync_next
        session['hours']=(time.monotonic()-begun)/3600
        result['aggregate']=summarize(result)
        write_json_atomic(path,result);summary_path.write_text(json.dumps(result['aggregate'],indent=2),encoding='utf-8')
        if remote and time.monotonic()>=sync_next:
            try:upload_artifacts_together([(path,f'{prefix}/latest.json'),(summary_path,f'{prefix}/latest-summary.txt')],hub,commit_message=f'EXP{experiment} metrics')
            except Exception as e:
                status=getattr(getattr(e,'response',None),'status_code',None)
                if status in (401,403):raise
                result['summary_sync_error']=type(e).__name__;sync_next=time.monotonic()+(3600 if status==429 else 120)
            else:result.pop('summary_sync_error',None);sync_next=time.monotonic()+600
    def notify(event):
        result.setdefault('notifications',[]).append(dict(event=event,**_safe_notify(args.telegram,
            f"Extent EXP-{experiment}: {event}; status={result.get('status')}; HF pending={len(result['pending_slots'])}")));persist()
    def sync_states():
        # Reserve entire pending payloads before uploading any bytes. No quota claim.
        remaining=args.max_new_hf_gb*1e9-session['uploaded_bytes']
        required=upload_requirement(store,result['pending_slots'])
        if required>remaining:
            result['storage_error']=dict(required_bytes=required,remaining_invocation_bytes=remaining);persist();return False
        for slot in list(result['pending_slots']):
            if store.sync(slot,force=True,deadline=wall_deadline-60):
                receipt=json.loads((store.directory(slot)/'synced.json').read_text())
                session['uploaded_bytes']+=receipt['payload_bytes'];result['pending_slots'].remove(slot)
                seed,arm=slot.split('/');result['branches'][seed][arm]['durable_step']=receipt['step']
            else:result['checkpoint_sync_errors']=dict(store.sync_errors)
        result['checkpoint_upload_status']='pending' if result['pending_slots'] else 'durable';persist(remote=True)
        return not result['pending_slots']
    # Metrics can be newer than cloud state. Reconcile before skipping tasks.
    if not args.sync_only:
        for seed,rows in result['branches'].items():
            for arm,row in rows.items():
                if not row.get('step') or row.get('metrics_only_pilot'):continue
                slot=f'{seed}/{arm}'
                metadata=store.metadata(slot,dict(contract,seed=int(seed),arm=arm))
                safe=0 if metadata is None else metadata['step']
                if metadata is None and row.get('durable_step',0):raise FileNotFoundError('durable pointer unavailable')
                if safe!=row['step']:
                    row.setdefault('rollbacks',[]).append(dict(reported=row['step'],restored=safe))
                    row['step']=safe;row.pop('final',None)
                    row['diagnostics']=[d for d in row['diagnostics'] if d['step']<=safe]
                    for field in ('milestones','probes'):row[field]={k:v for k,v in row.get(field,{}).items() if int(k)<=safe}
        # Pending paths from a destroyed runtime cannot be uploaded without files.
        result['pending_slots']=[slot for slot in result['pending_slots'] if (store.directory(slot)/'checkpoint.json').exists()]
    if args.sync_only or completed(result):
        sync_states();notify('sync finished' if not result['pending_slots'] else 'HF pending');return result
    notify('started')
    try:
        result['status']='preflight';persist(remote=True)
        if result.get('summary_sync_error'):raise RuntimeError('HF write preflight failed before models')
        tokens=load_tokens(args.dataset_cache_dir)
        if token_manifest(tokens)!=contract['data_manifest']:raise ValueError('data SHA mismatch')
        result['cloud_roundtrip']=cloud_roundtrip(Path(args.checkpoint_dir),prefix,hub,deadline)
        devices=list(jax.devices())
        if jax.process_count()!=1 or len(devices)!=8 or any(d.platform!='tpu' for d in devices):raise ValueError('single-host TPU v5e8 required')
        mesh=create_v5e_mesh(devices);batch=batch_sharding(mesh);micro=NamedSharding(mesh,P('data',None,None))
        import psutil
        if psutil.virtual_memory().available<80*(1<<30):raise MemoryError('source caches/local checkpoints require >=80GiB available host RAM')
        model,order,sources,teacher_source=load_sources(config,mesh,args,hub,tokens,result,persist,deadline,experiment=experiment)
        teacher,teacher_host,source_config=teacher_source
        probes=Probes(model,teacher,config,source_config,mesh,batch)
        columns=2*int(config.hidden_size*config.mamba.expand)+2*config.mamba.mimo_rank*config.mamba.groups*config.mamba.d_state
        adapter=experiment==111;kind='ADAPTER-CE' if adapter else 'DECODER-CE'
        compiled={};result['status']='running'
        report_update=jax.jit(lambda a,b,c,d:update_metrics(a,b,c,d,config))
        # Pilots are metrics-only once finished. MAIN replays chosen pilot from
        # pinned start; all pilot/replay updates are counted, never free.
        while True:
            tasks=expected_tasks(result,experiment)
            for seed,arm,target,pilot in tasks:
                row=result['branches'].setdefault(str(seed),{}).setdefault(arm,dict(step=0,diagnostics=[]))
                if row.get('failed') or row['step']>target:continue
                if row['step']==target and (row.get('metrics_only_pilot') if pilot else
                    str(target) in row.get('milestones',{}) and (target not in PROBE_STEPS or str(target) in row.get('probes',{}))):continue
                if time.monotonic()>=deadline:raise TimeoutError('registered training deadline')
                peak=3e-4 if adapter else PEAKS[arm if pilot else result['selected_lr']]
                base,c=sources[seed];base=jax.device_put(base,named_sharding_tree(base,mesh));c=jax.device_put(c,trainable_layout(c,mesh))
                if adapter:parameters,fixed=c,base
                else:parameters,fixed=split_decoder(base)
                pl,fl=trainable_layout(parameters,mesh),named_sharding_tree(fixed,mesh)
                parameters,fixed=jax.device_put(parameters,pl),jax.device_put(fixed,fl)
                initial=forward_parameters(kind,parameters,fixed,columns,config.mamba.head_dim)
                if not all(bool(jnp.all(a==b)) for a,b in zip(jax.tree.leaves(initial),jax.tree.leaves(base))):raise ValueError('initial BF16 identity failed')
                if 'start' not in row:
                    row['start']={n:evaluate(model,initial,tokens[n],batch) for n in ('validation','locked_test','pg19_test','reproduction')}
                    row['diagnostics'].append(dict(step=0,reproduction=row['start']['reproduction'],validation=row['start']['validation']))
                plan=memory_estimate(parameters,fixed,pl,fl);row['memory_policy']=plan
                limit=(devices[0].memory_stats() or {}).get('bytes_limit')
                if limit is None:limit=(devices[0].memory_stats() or {}).get('bytes_reservable_limit')
                if limit is None or plan['conservative_bytes_per_device']>limit:raise MemoryError('unsupported HBM policy budget')
                tx=optimizer(parameters,peak=peak,adapter=adapter)
                abstract=jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),parameters)
                sharded=initialize_sharded_optimizer_state(tx,parameters,abstract,pl,mesh);state,sl=sharded.opt_state,sharded.layout
                del sharded
                slot=f'{seed}/{arm}';cc=dict(contract,seed=seed,arm=arm)
                saved=store.restore(slot,cc,dict(parameters=parameters,optimizer=state),dict(parameters=pl,optimizer=sl))
                if saved is not None:
                    payload,meta=saved;parameters,state=payload['parameters'],payload['optimizer'];row['step']=meta['step'];del payload,saved
                    row['diagnostics']=[d for d in row['diagnostics'] if d['step']<=row['step']]
                    row['milestones']={k:v for k,v in row.get('milestones',{}).items() if int(k)<=row['step']}
                elif row['step']:raise FileNotFoundError('active trajectory lacks compatible state')
                if int(state[1].count)!=row['step'] or int(state[3].count)!=row['step']:raise ValueError('Adam/schedule cursor mismatch')
                if not bool(coordinates_finite(parameters)) or not bool(coordinates_finite(state)):raise FloatingPointError('nonfinite restored state')
                if peak not in compiled:
                    step=make_step(model,tx,kind,columns,config.mamba.head_dim,(pl,sl,fl,micro,replicated_sharding(mesh)),accumulation=1)
                    compiled[peak]=step.lower(parameters,state,fixed,jax.device_put(tokens['train'][:1,None,:],micro)).compile()
                step=compiled[peak];analysis=step.memory_analysis()
                if analysis is not None:
                    measured=analysis.argument_size_in_bytes+analysis.output_size_in_bytes+analysis.temp_size_in_bytes-analysis.alias_size_in_bytes
                    row['compiler_memory_bytes']=int(measured)
                    if measured>limit:raise MemoryError('compiled step exceeds HBM')
                def current():return forward_parameters(kind,parameters,fixed,columns,config.mamba.head_dim)
                def save():
                    meta=store.save(slot,cc,dict(parameters=parameters,optimizer=state),row['step']);row['checkpoint_bytes']=meta['payload_bytes'];persist()
                def diagnostic(n):
                    if time.monotonic()>=deadline:raise TimeoutError('diagnostics deferred at safe cursor')
                    clock=time.monotonic()
                    row.setdefault('probes',{})[str(n)]=probes.run(current(),initial,teacher_host,tokens['validation'][:8],rollback=n in (4096,8192,16384),fp32=args.fp32_probe)
                    row['diagnostic_seconds']=row.get('diagnostic_seconds',0.)+time.monotonic()-clock
                p2,s2=parameters,state
                try:
                    if not pilot and row['step'] in PROBE_STEPS and str(row['step']) not in row.get('probes',{}):diagnostic(row['step'])
                    while row['step']<target and time.monotonic()<deadline:
                        clock=time.monotonic();p2,s2,h=step(parameters,state,fixed,jax.device_put(tokens['train'][row['step']:row['step']+1,None,:],micro))
                        h=jax.tree.map(lambda x:np.asarray(x).item(),jax.device_get(h))
                        row['training_seconds']=row.get('training_seconds',0.)+time.monotonic()-clock
                        row['attempted_updates']=row.get('attempted_updates',0)+1
                        if not h['finite']:row.update(failed=True,error='nonfinite proposal rejected');break
                        update=None
                        if (row['step']+1)%1024==0:
                            clock=time.monotonic()
                            update=json_safe(report_update(parameters,p2,current(),forward_parameters(kind,p2,fixed,columns,config.mamba.head_dim)))
                            row['diagnostic_seconds']=row.get('diagnostic_seconds',0.)+time.monotonic()-clock
                        parameters,state=p2,s2;row['step']+=1;row['last_health']=h;row['last_learning_rate']=float(schedule(row['step']-1,peak,adapter))
                        if row['step']%256==0:
                            d=dict(step=row['step'],health=h,validation=evaluate(model,current(),tokens['validation'],batch),training_seconds=row['training_seconds'])
                            if row['step']%1024==0:
                                d['actual_update']=update
                                d['reproduction']=evaluate(model,current(),tokens['reproduction'],batch)
                                d['locked_test']=evaluate(model,current(),tokens['locked_test'],batch)
                                print(f"EXP{experiment} {seed}/{arm} step={row['step']} val={d['validation']['nll']:.5f}",flush=True)
                            row['diagnostics'].append(d);persist(remote=True)
                            if not pilot and row['step']==PILOT_STEPS and not adapter:
                                row['pilot_replay_parity']=pilot_parity(d,result['branches'][str(seed)][result['selected_lr']])
                        if not pilot and row['step'] in PROBE_STEPS:diagnostic(row['step'])
                        if row['step']%4096==0:save()
                    if pilot and row.get('failed'):row['metrics_only_pilot']=True
                    if pilot and row['step']==PILOT_STEPS and not row.get('failed'):
                        row['pilot_validation']=evaluate(model,current(),tokens['validation'],batch);row['metrics_only_pilot']=True
                        # Completed trials need no remote state: immutable source
                        # + fixed seeds reconstruct selected pilot, charged below.
                    elif not pilot and row['step']==target:
                        row.setdefault('milestones',{})[str(target)]={n:evaluate(model,current(),tokens[n],batch) for n in ('locked_test','pg19_test','reproduction')}
                        if target==TOTAL_STEPS:row['final']=row['milestones'][str(target)]
                finally:
                    if not row.get('metrics_only_pilot'):
                        save()
                        if slot not in result['pending_slots']:result['pending_slots'].append(slot)
                if not pilot and row.get('failed'):raise FloatingPointError('MAIN failed; last safe checkpoint retained')
                # Completed pilots are replayable metrics, not claimed durable states.
                if pilot and row.get('metrics_only_pilot') and slot in result['pending_slots']:result['pending_slots'].remove(slot)
                del parameters,state,fixed,initial,base,c,step,current,p2,s2;gc.collect();persist(remote=True)
            if experiment!=111 and not result.get('selected_lr'):
                selected,scores=choose_lr(result['branches']);result['selected_lr']=selected;result['lr_scores']=scores;persist(remote=True);continue
            break
        result['status']='completed' if completed(result) else 'branch_failure'
    except TimeoutError as e:result.update(status='deadline_partial',error=str(e))
    except Exception as e:
        result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());notify('failed')
        sync_states();persist(remote=True);raise
    notify('training finished' if completed(result) else 'training stopped')
    sync_states()
    while result['pending_slots'] and time.monotonic()<wall_deadline-60 and not result.get('storage_error'):
        wait=max(1.,store.next_sync-time.monotonic())
        if wait>=wall_deadline-time.monotonic()-60:break
        time.sleep(min(wait,60));sync_states()
    result['checkpoint_upload_status']='pending' if result['pending_slots'] else 'durable'
    notify('finished and saved' if not result['pending_slots'] else 'finished; HF states pending')
    sync_next=0.;persist(remote=True);return result

if __name__=='__main__':main()
