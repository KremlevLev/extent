"""EXP160: identical output KD on old cyclic Wiki versus fresh non-cycling C4."""
import argparse,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from dataclasses import replace
from importlib.metadata import version
from types import SimpleNamespace
import numpy as np
from extent.corpus_recovery_160 import recipes,TPU_EXPERIMENTS,GPU_EXPERIMENTS,LIMITS
ROOT=Path(__file__).resolve().parents[1]
EVAL_STEPS=(0,128,512,1024,2048,2560,3072,3434,3584,4096,4608,5120,6144,8192,10240,12288,16384,20480,24576,28672,32768,40960,49152,65536,81920,98304,114688,131072,163840,196608,229376,262144)


def contract_for(experiment,worker=None,max_steps=262144,corpus_dir=None,execution='t4-pair'):
    from extent.broad_corpus import PackedCorpus,C4_REVISION
    corpus=PackedCorpus(corpus_dir)
    from extent.qwen_source import QWEN3_1_7B_BASE as donor
    data=corpus.manifest['contract']
    if (data.get('dataset')!='allenai/c4' or data.get('revision')!=C4_REVISION or data.get('config')!='en' or
        data['sequence_length']!=256 or data['tokenizer']!=donor.repo_id or
        data['tokenizer_revision']!=donor.revision or data['vocab_size']!=donor.vocab_size or
        data['validation_tokens']<128*256):
        raise ValueError('C4 tokenizer/context/validation contract incompatible with EXP160')
    from scripts.m3q_rank_growth_153_campaign import contract_for as previous
    from extent.corpus_recovery_160 import source_pin
    server=execution=='server-balanced'
    if execution not in ('t4-pair','server-balanced'):raise ValueError('unknown execution mode')
    if experiment!=160 or (worker is not None if server else worker not in (0,1)):
        raise ValueError('server needs no worker; T4 needs worker0/1')
    c=previous(153,1,max_steps)
    c.update(protocol='exp160-corpus-recovery-v1',experiment=160,worker=worker,recipes=recipes(160) if server else [recipes(160)[worker]],
        horizon=max_steps,first_goal_new_steps=8192,global_eval_steps=[],eval_steps=[0,128,256,512,1024,2048,3072,4096,6144,8192,12288,16384,24576,32768,49152,65536,98304,131072,196608,262144],
        preparation='same immutable155 CURRENT118304 and validated136 folded base; new data experiment, no reset/growth/fold',
        optimizer='fresh Adam both workers, warmup512, mixer3e-4/pregate1e-4, stable global clip1',
        continuation='GPU0 .5CE+.5KL old cyclic Wiki; GPU1 identical .5CE+.5KL fresh C4; hidden_beta0 both',
        comparison='data intervention at matched updates; old Wiki versus C4 changes domain, diversity and repeat rate together; not a pure repetition ablation',
        clock='GLOBAL118304; NEW=global-118304; optimizer=NEW; C4 window=NEW, never source global or modulo',
        corpus_identity=corpus.identity,corpus_manifest=corpus.manifest,
        gradient='streamed full chain rule; auxiliary cotangents injected BEFORE VJP of corresponding block; frozen teacher, finite proposal gate',
        execution_namespace='experiments/exp160-corpus-recovery-v1; source155 read-only',
        gpu='one independent worker perT4; frozen teacher layers host->GPU one at a time',
        runtime_policy='exact155 pinned math runtime and all160source-window replay required; CPU gradient audit against whole-chain objective',
        selection='source A selects best for historical comparability; C4 D128 and old B/C, EN/RU descriptive; final separate',
        numerical_gate='no parameter update before per-window source replay<=.03; finite teacher/student objectives and gradients',
        limitations='single seed; fixed KD; different data domains and diversity; limited tokens; no claim of MOHAWK reproduction',
        source_pin_sha=hashlib.sha256((ROOT/'results/EXP-157-source-pin.json').read_bytes().replace(b'\r\n',b'\n')).hexdigest())
    for file in ('scripts/m3q_corpus_recovery_160.py','extent/corpus_recovery_160.py','extent/broad_corpus.py','extent/hidden_distillation_159.py','extent/decoder_recovery_157.py','extent/quality_benchmark_156.py'):
        c['code'][file]=hashlib.sha256((ROOT/file).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if server:
        c.update(protocol='exp160-corpus-server-v1',execution_namespace='experiments/exp160-corpus-server-v1',
                 execution=execution,gpu='one visible GPU; both branches resident; teacher resident; balanced sequential updates',
                 checkpoint='local at most once per arm per15min plus exit; atomic current generation; remote4h plus exit; explicit upload cap')
        c['eval_steps']+=list(range(294912,min(max_steps,data['train_tokens']//256)+1,32768))
        c['code']['extent/server_control.py']=hashlib.sha256((ROOT/'extent/server_control.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return c


def session_budget(experiment,max_wall_hours,save_reserve_minutes,execution='t4-pair'):
    """Validate this suite's hardware limits, including GPU worker invocations."""
    if experiment not in LIMITS:raise ValueError('EXP160 only')
    limit=24*365 if execution=='server-balanced' else LIMITS[experiment]
    if not np.isfinite(max_wall_hours) or not 0 < max_wall_hours <= limit:
        raise ValueError(f'EXP{experiment} wall budget must be >0 and <={limit:g}h')
    if not np.isfinite(save_reserve_minutes) or not 18 <= save_reserve_minutes < max_wall_hours*60:
        raise ValueError('save reserve must be >=18min and less than the wall budget')
    return max_wall_hours*3600-save_reserve_minutes*60,save_reserve_minutes


def session(args,contract,hub):
    # Reuse audited checkpoint/report transport, without the old 133 source recipe.
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from extent.recovery_durability import RecoveryCheckpointStore
    from extent.hf_artifact_sync import restore_artifact
    import jax
    s=Transport.__new__(Transport);s.args=args;s.contract=contract;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    server=getattr(args,'execution','t4-pair')=='server-balanced'
    suffix=f'/worker-{args.worker}' if args.worker is not None else ''
    s.prefix=(f'experiments/exp{args.experiment}-corpus-server-v1' if server else f'experiments/exp{args.experiment}-corpus-recovery-v1')+suffix
    s.stem='runtime-report' if server else f'extent-m3q-corpus-recovery-{args.experiment}'+(f'-worker-{args.worker}' if args.worker is not None else '')
    work,_=session_budget(args.experiment,args.max_wall_hours,args.save_reserve_minutes,getattr(args,'execution','t4-pair'));s.deadline=s.begun+work;s.wall=s.begun+args.max_wall_hours*3600
    if getattr(args,'deadline_unix',None) is not None:
        s.deadline=min(s.deadline,time.monotonic()+max(0,args.deadline_unix-time.time()));s.wall=s.deadline+args.save_reserve_minutes*60
    s.output=Path(args.output_dir);s.output.mkdir(parents=True,exist_ok=True);s.path=s.output/(s.stem+'.json');s.small=s.output/(s.stem+'-summary.txt')
    if not s.path.exists():restore_artifact(s.path,s.prefix+'/latest.json',hub)
    s.result=json.loads(s.path.read_text()) if s.path.exists() else dict(contract=contract,branches={},sessions=[],pending_slots=[],cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0))
    if s.result['contract']!=contract:raise ValueError('follow-up resume contract differs; do not overwrite old state')
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    if s.result.get('runtime',runtime)!=runtime:raise ValueError('follow-up math runtime changed')
    s.result['runtime']=runtime;s.result.pop('error',None)
    s.result['storage_preflight']={'scope':'adapter states only; no dense checkpoints or frozen-base uploads'}
    s.store=RecoveryCheckpointStore(Path(args.checkpoint_dir)/(s.stem),s.prefix+'/checkpoints',hub)
    s.next_sync=0.;s.invocation=dict(max_wall_hours=args.max_wall_hours,save_reserve_minutes=args.save_reserve_minutes,uploaded_bytes=0)
    s.result['sessions'].append(s.invocation)
    def emit(event):
        progress={name:r.get('step',0) for name,r in s.result['branches'].items()}
        message=f'Extent EXP{args.experiment}: {event}; status={s.result.get("status")}; steps={progress}; HF pending={len(s.result["pending_slots"])}'
        s.result.setdefault('events',[]).append(dict(event=event,telegram_sent=False))
        if getattr(args,'_notice',None):args._notice.emit('progress',message)
        print(message,flush=True);s.persist(local_only=True)
    s.emit=emit
    if server:
        from extent.server_control import install_session_control
        install_session_control(s,args)
    return s


def run(args):
    import jax
    import jax.numpy as jnp
    from extent.config import load_config
    from extent.corpus_recovery_160 import block_forward,Inspector,replay_report
    from extent.corpus_recovery_160 import prepare,restore_source,base_coordinates,make_kernel,adam_counts
    from extent.corpus_recovery_160 import optimizer
    from extent.competitor_tpu import token_nll,training_batch,reconcile
    from extent.reset_streamed_gpu import make_step
    from extent.plateau_forensics import GuardedJit,plain,state_digest
    from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding,replicated_sharding
    from extent.reset_long_patient import Patient
    from extent.t4_scale_data import load_data
    from extent.hf_artifact_sync import artifact_config_from_env
    from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer
    cfg,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    execution=getattr(args,'execution','t4-pair')
    c=contract_for(args.experiment,args.worker,args.max_steps,args.corpus_dir,execution);horizon=c['horizon'];eval_steps=c['eval_steps'];hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    s=session(args,c,hub);payloads={};bases={};layouts={};kernels={};evaluators={};specs={};teacher_params=None
    if s.result.get('status')=='completed':s.persist(force=True);return s.result
    s.result['status']='preflight';s.emit('started')
    try:
        if not s.persist(force=True):raise RuntimeError('HF report preflight failed')
        devices=jax.devices('gpu' if args.experiment in GPU_EXPERIMENTS else 'tpu')
        expected=1 if args.experiment in GPU_EXPERIMENTS else 8
        if len(devices)!=expected or jax.process_count()!=1:raise ValueError(f'expected{expected} local accelerator devices')
        mesh=create_v5e_mesh(devices);s.result['devices']=[dict(id=d.id,kind=d.device_kind,platform=d.platform) for d in devices]
        tokens,meta=load_data(Path(args.cache_dir)/'reset-data',s.check);s.result['data']=meta
        from extent.corpus_recovery_160 import source_pin
        from extent.broad_corpus import PackedCorpus,CorpusExhausted
        corpus=PackedCorpus(args.corpus_dir)
        if json.loads(json.dumps(meta))!=source_pin()['data']:raise ValueError('source training/validation data identity changed')
        tokens['D']=corpus.validation()[:128]
        s.result['corpus_identity']=corpus.identity
        from extent.quality_benchmark_156 import prepare_data
        from transformers import AutoTokenizer
        from extent.qwen_source import QWEN3_1_7B_BASE
        tokenizer=AutoTokenizer.from_pretrained(QWEN3_1_7B_BASE.repo_id,revision=QWEN3_1_7B_BASE.revision,cache_dir=Path(args.cache_dir)/'tokenizer')
        fresh_dir=Path(args.cache_dir)/'benchmark-data'
        prepare_data(fresh_dir,tokenizer,s.check)
        fresh=dict(np.load(fresh_dir/'fresh-panels.npz',allow_pickle=False))
        s.result['fresh_data_manifest']=json.loads((fresh_dir/'data-manifest.json').read_text())
        # Host source loading, not full model initialization on accelerator.
        patient=Patient(Path(args.cache_dir)/'reset-patient',hub,jax.devices('cpu')[0],s.check,seed=c['seed'])
        if patient.cfg!=cfg:raise ValueError('patient/config mismatch')
        columns=patient.columns;tc=patient.tc
        ancestor136=None
        ancestor136=base_coordinates(hub,Path(args.cache_dir)/'immutable',s.check)
        teacher_params=patient.teacher
        if execution=='server-balanced':
            teacher_params=jax.device_put(teacher_params,named_sharding_tree(teacher_params,mesh))
            s.result['teacher_execution']='resident GPU weights; no per-step host teacher weight copies'
            s.result['teacher_resident_bytes']=sum(a.size*a.dtype.itemsize for a in jax.tree.leaves(teacher_params))
        for spec in c['recipes']:
            s.check();name=spec['name'];specs[name]=spec
            source_payload=pin=None
            source_payload,pin=restore_source(spec['source'],hub,Path(args.cache_dir)/'immutable',s.check)
            if s.result['runtime']!=pin['runtime']:raise ValueError('source runtime differs; refusing nominal exact resume')
            with jax.default_device(jax.devices('cpu')[0]):base,q,source_cursor=prepare(patient,spec,source_payload,pin,ancestor136)
            for key in ('embed_tokens','norm'):
                for left,right in zip(jax.tree.leaves(base[key]),jax.tree.leaves(patient.teacher[key])):
                    if not np.array_equal(np.asarray(left),np.asarray(right)):raise ValueError('teacher/student shared frozen embedding/head differ')
            s.result['teacher_shared_head_verified']=True
            base=jax.device_put(base,named_sharding_tree(base,mesh));bases[name]=base
            layout=coordinate_layout(q,mesh)
            from scripts.m3q_paper_decoder_recovery_campaign import memory_estimate
            resource=memory_estimate(q,base,layout,named_sharding_tree(base,mesh))
            s.result.setdefault('resource_estimates',{})[name]=resource
            s.persist(local_only=True)
            q=jax.device_put(q,layout);tx=optimizer(q,spec)
            state,state_layout=sharded_optimizer(tx,q,layout,mesh)
            if spec['mode']=='exact':state=jax.device_put(source_payload['optimizer'],state_layout)
            best=q;best_step=pin['source_step']
            template=dict(parameters=q,optimizer=state,best=best,best_step=jnp.asarray(best_step,jnp.int32))
            layouts[name]=dict(parameters=layout,optimizer=state_layout,best=layout,best_step=replicated_sharding(mesh))
            saved=s.store.restore(name,dict(c,arm=name),template,layouts[name]);payload=saved[0] if saved else template
            cursor=saved[1]['step'] if saved else source_cursor
            row=s.result['branches'].setdefault(name,dict(step=cursor,source_step=source_cursor,history=[],evaluations=[]))
            if not saved and row.get('durable_step',source_cursor)>source_cursor:raise FileNotFoundError('durable continuation missing')
            if saved and row.get('durable_step',source_cursor)>cursor:raise ValueError('durable report cursor ahead of restored checkpoint')
            reconcile(row,cursor,int(payload['best_step']));payloads[name]=payload
            row['invocation_updates']=0;row['timing_samples']=[]
            row['source_adam_step']=pin['adam_step']
            row['immutable_source_adam_count']=pin['adam_step']
            row['initialization']='CURRENT118304, unchanged trained R256, freshAdam, own registered data stream; NEW index is checkpoint global minus source118304'
            row['trainable_scope']='protected rank256 and pregate; teacher and donor backbone frozen'
            row['adam_count_at_restore']=adam_counts(payload['optimizer'])
            expected_count=(pin['adam_step'] if spec['mode']=='exact' else 0)+(cursor-source_cursor)
            if row['adam_count_at_restore']!=[expected_count]*6:raise ValueError('own restored Adam/global clocks disagree')
            row['coordinates']=sum(a.size for a in jax.tree.leaves(q));row['payload_bytes']=sum(a.size*a.dtype.itemsize for a in jax.tree.leaves(template))
            kernels[name]=make_kernel(cfg,tc,columns,spec,tx,tpu=args.experiment in TPU_EXPERIMENTS)
            kernels[name].check=s.check
            evaluators[name]=GuardedJit(lambda b,q,t,step,spec=spec:token_nll(cfg,b,q,t,columns,spec,step,block_fn=block_forward))
            def evaluate(q,windows,step,tail=None,function=evaluators[name],weights=base):
                losses=[]
                for t in windows:
                    s.check();v=np.asarray(function(weights,q,jax.device_put(t[None],batch_sharding(mesh)),jnp.asarray(step,jnp.float32)))[0]
                    if not np.all(np.isfinite(v)):raise FloatingPointError('nonfinite ordinary held-out NLL')
                    losses.append(float(np.mean(v if tail is None else v[-tail:])))
                return dict(nll=float(np.mean(losses)),windows=losses,targets_per_window=255 if tail is None else tail)
            if 'source_replay' not in row or not row['source_replay'].get('passed'):
                # Source CURRENT lacks a recorded NLL. Replay its independently
                # stored SELECTED BEST against the pinned per-window reference.
                control={p:evaluate(best,tokens[p],pin['best_step']) for p in ('A','B','C')}
                row['source_best_controls']=control
                row['source_replay']=dict(replay_report(control,pin),reference_step=pin['best_step'],scope='pinned CURRENT118304 before updates')
                # Measurements survive even when the scientific gate rejects startup.
                s.persist(local_only=True)
                s.persist(force=True,checkpoints=False)
                if not row['source_replay']['passed']:
                    raise ValueError('TPU inherited-best NLL replay differs; refusing update; diagnostics persisted')
            if 'source_controls' not in row:
                row['source_controls']={p:evaluate(q,tokens[p],source_cursor) for p in ('A','B','C','D')}
                s.persist(local_only=True)
            if 'teacher_c4_reference' not in s.result:
                from extent.corpus_recovery_160 import teacher_validation
                s.result['teacher_c4_reference']=teacher_validation(cfg,kernels[name],base,teacher_params,tokens['D'],s.check)
                s.persist(local_only=True)
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
                for panel in ('A','B','C','D') if ready else ('A',):
                    point['panels'][panel]=evaluators[name].evaluate(payload['parameters'],tokens[panel] if ready else tokens[panel][:16],at)
            if (at-row['source_step'])%1024==0 or force:
                point['fresh_nll']={key:evaluators[name].evaluate(payload['parameters'],arr[:4096].reshape(16,256),at) for key,arr in fresh.items()}
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
                cursor=row['step'];origin=row['source_step']
                block_size=8 if min(r.get('invocation_updates',0) for r in rows.values())<8 else 256
                target=min(origin+horizon,cursor+block_size)
                try:
                    evaluate_endpoint(name)
                    while cursor<target:
                        s.check();t=(corpus.batch(cursor-row['source_step']) if spec['data_stream']=='c4' else training_batch(tokens['FIT'],cursor,133))
                        begun=time.monotonic();cost=s.result['cost'];cost['attempted_updates']+=1;cost['input_tokens']+=t.size;cost['ce_targets']+=t.shape[1]-1
                        if spec['alpha']:cost['teacher_input_tokens']+=t.size
                        outputs=kernels[name](payload['parameters'],payload['optimizer'],bases[name],teacher_params,
                            jax.device_put(t,batch_sharding(mesh)),jnp.asarray(cursor,jnp.float32))
                        proposed,after,health=outputs[:3]
                        seconds=time.monotonic()-begun;cost['training_seconds']=cost.get('training_seconds',0)+seconds;h=plain(health)
                        row['training_seconds']=row.get('training_seconds',0)+seconds
                        if 'engine_gate' in h:
                            audit=dict(step=cursor,**h['engine_gate'])
                            row.setdefault('execution_checks',[]).append(audit)
                            extra=max(0,audit['gradient_evaluations']-1)
                            cost['engine_audit_extra_input_tokens']=cost.get('engine_audit_extra_input_tokens',0)+extra*t.size
                            cost['engine_audit_extra_ce_targets']=cost.get('engine_audit_extra_ce_targets',0)+extra*t.shape[0]*(t.shape[1]-1)
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
                                input_tokens_per_second=t.size/median,teacher_input_tokens_per_second=t.size/median if spec['alpha'] else 0,
                                c4_windows_remaining=max(0,len(corpus.arrays['train'])//256-(cursor-row['source_step'])) if spec['data_stream']=='c4' else None,
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
                if time.monotonic()-last_sync>=14400:
                    s.persist(force=True);last_sync=time.monotonic()
                    if s.result.get('storage_error'):raise TimeoutError('upload cap; progress saved locally')
                if time.monotonic()-last_notice>=14400:s.emit('progress');last_notice=time.monotonic()
            s.result['status']='arm_failed' if any(r.get('failed') for r in rows.values()) else 'completed'
        except CorpusExhausted as exc:s.result['status']='data_exhausted';s.result['stop_reason']=str(exc)
        except TimeoutError as exc:s.result['status']='deadline_partial';s.result['stop_reason']=str(exc)
        # Stop-file checks cease during bounded save/finalization, not during training.
        s.finalizing=True
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
                    for panel in ('A','B','C','D'):
                        if panel not in point['panels']:point['panels'][panel]=ev(q,tokens[panel],step);s.persist()
                    if label=='last' and 'fresh_nll' not in point:
                        point['fresh_nll']={key:ev(q,arr[:32768].reshape(128,256),step) for key,arr in fresh.items()};s.persist()
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
        s.finalizing=True
        for name,payload in payloads.items():s.checkpoint(name,payload,s.result['branches'][name]['step'])
        s.persist(force=True)
        if s.result.get('storage_error'):s.result['status']='storage_partial';s.persist(local_only=True)
        s.emit('finished and saved' if not s.result['pending_slots'] and not s.result.get('report_sync_error') else 'finished; HF pending')
    if s.result['status']=='failed':raise RuntimeError('follow-up stopped; see saved report')
    return s.result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-dir',required=True,help='completed pinned CPU-prepared C4 directory')
    p.add_argument('--execution',choices=('t4-pair','server-balanced'),default='t4-pair')
    p.add_argument('--stop-file')
    p.add_argument('--work-root')
    p.add_argument('--max-local-gb',type=float,default=80.)
    p.add_argument('--min-free-gb',type=float,default=20.)
    p.add_argument('--experiment',type=int,choices=(160,),default=160)
    p.add_argument('--max-wall-hours',type=float,default=11.8)
    p.add_argument('--max-steps',type=int,default=262144,help='NEW updates after118304; ceiling not early stop target')
    p.add_argument('--save-reserve-minutes',type=float,default=18.)
    p.add_argument('--max-new-hf-gb',type=float,default=8.)
    p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--cache-dir',default='/kaggle/working/extent-160-cache')
    p.add_argument('--checkpoint-dir',default='/kaggle/working/extent-160-states')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--worker',type=int,choices=(0,1))
    p.add_argument('--deadline-unix',type=float)
    return p



def aggregate(output,contracts):
    value=dict(protocol='exp160-two-gpu-distillation-v1',contracts=contracts,workers={})
    for i in (0,1):
        path=Path(output)/f'extent-m3q-corpus-recovery-160-worker-{i}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row['contract']!=contracts[str(i)]:raise ValueError('worker report contract differs')
            value['workers'][str(i)]=row
    statuses=[r.get('status') for r in value['workers'].values()]
    value['status']='failed' if any(s in ('failed','arm_failed') for s in statuses) else 'completed' if len(statuses)==2 and all(s=='completed' for s in statuses) else 'deadline_partial'
    return value


def main(argv=None):
    args=parser().parse_args(argv)
    session_budget(args.experiment,args.max_wall_hours,args.save_reserve_minutes,args.execution)
    if args.max_steps<32768 or not np.isfinite(args.max_new_hf_gb) or args.max_new_hf_gb<=0:raise ValueError('positive upload cap and high step ceiling required')
    if args.plan_only:return contract_for(160,None,args.max_steps,args.corpus_dir,args.execution) if args.execution=='server-balanced' else {str(i):contract_for(160,i,args.max_steps,args.corpus_dir) for i in (0,1)}
    if args.execution=='server-balanced':return run(args)
    if args.worker is not None:return run(args)
    started=time.time();deadline=started+args.max_wall_hours*3600-args.save_reserve_minutes*60;wall=started+args.max_wall_hours*3600
    os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(deadline)
    os.environ['EXTENT_HF_SAVE_DEADLINE_UNIX']=str(wall-60)
    from extent.hf_download_retry import install_download_retries
    install_download_retries()
    from extent.hf_artifact_sync import artifact_config_from_env,upload_artifact
    from extent.reset_quiet import ParentNotifier
    from extent.campaign_checkpoint import write_json_atomic
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    contracts={str(i):contract_for(160,i,args.max_steps,args.corpus_dir) for i in (0,1)}
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    path=output/'extent-m3q-corpus-recovery-160.json';prefix='experiments/exp160-corpus-recovery-v1'
    notice=ParentNotifier(args.telegram,160);notice.emit('started','same outputKD; old Wiki versus fresh C4;2T4;source118304')
    processes=[];next_sync=0;result=dict(protocol='exp160-two-gpu-distillation-v1',contracts=contracts,workers={},status='starting')
    uploaded=0;cap=2_000_000_000
    def collect():
        nonlocal result
        result=aggregate(output,contracts);write_json_atomic(path,result)
    def sync(force=False,status_override=None,parent_error=None):
        nonlocal next_sync,uploaded
        if not force and time.time()<next_sync:return
        collect()
        if status_override:result['status']=status_override
        if parent_error:result['parent_error']=parent_error
        write_json_atomic(path,result)
        if uploaded+path.stat().st_size>cap:result['report_sync_error']='parent report upload cap';write_json_atomic(path,result);return
        try:
            upload_artifact(path,prefix+'/latest.json',hub,commit_message='EXP160 corpus comparison metrics')
            uploaded+=path.stat().st_size;result['report_sync_error']=None;next_sync=time.time()+3600
        except Exception as exc:
            headers=getattr(getattr(exc,'response',None),'headers',{})
            try:delay=max(120.,float(headers.get('Retry-After',120)))
            except (TypeError,ValueError):delay=120.
            next_sync=time.time()+delay;result['report_sync_error']=dict(type=type(exc).__name__,retry_after_seconds=delay)
        write_json_atomic(path,result)
    try:
        for i in (0,1):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(i),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            log=open(output/f'exp160-gpu{i}.log','a',encoding='utf-8')
            cmd=[sys.executable,'-m','scripts.m3q_corpus_recovery_160','--worker',str(i),
                '--corpus-dir',str(Path(args.corpus_dir).resolve()),'--deadline-unix',str(deadline),'--max-wall-hours',str(args.max_wall_hours),
                '--save-reserve-minutes',str(args.save_reserve_minutes),'--max-steps',str(args.max_steps),
                '--max-new-hf-gb',str(args.max_new_hf_gb),'--output-dir',str(output),
                '--cache-dir',str(Path(args.cache_dir)/f'worker-{i}'),
                '--checkpoint-dir',args.checkpoint_dir,'--no-telegram']
            processes.append((i,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        while True:
            collect();sync()
            crashed=any(p.poll() not in (None,0) for _,p,_ in processes)
            if result['status']=='failed' or crashed:
                result['status']='failed';write_json_atomic(path,result);break
            if all(p.poll() is not None for _,p,_ in processes):break
            if time.time()>=wall-60:break
            time.sleep(15)
    except Exception as exc:
        result['status']='failed';result['parent_error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());write_json_atomic(path,result);raise
    finally:
        parent_error=result.get('parent_error');failed=result.get('status')=='failed'
        for i,proc,log in processes:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            log.close()
        sync(True,'failed' if failed else None,parent_error)
        if failed:result['status']='failed'
        if parent_error:result['parent_error']=parent_error
        result['uploaded_report_bytes']=uploaded
        if result['status']=='failed':
            for i,_,_ in processes:
                f=output/f'exp160-gpu{i}.log'
                if f.exists():print(f.read_text(encoding='utf-8',errors='replace')[-6000:],flush=True)
        write_json_atomic(path,result)
        notice.emit('finished',result['status']+';download '+path.name)
    if result['status']=='failed':raise RuntimeError('EXP160 failed; actual worker traceback printed and aggregate saved')
    return result

if __name__=='__main__':main()
