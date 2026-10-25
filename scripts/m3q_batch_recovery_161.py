"""EXP161: optimizer-preserving batch6 continuation of the saved server160 arms."""
import argparse,gc,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from dataclasses import replace
from importlib.metadata import version
from types import SimpleNamespace
import numpy as np
from extent.batch_recovery_161 import recipes,TPU_EXPERIMENTS,GPU_EXPERIMENTS,LIMITS
ROOT=Path(__file__).resolve().parents[1]
EVAL_STEPS=(0,128,512,1024,2048,2560,3072,3434,3584,4096,4608,5120,6144,8192,10240,12288,16384,20480,24576,28672,32768,40960,49152,65536,81920,98304,114688,131072,163840,196608,229376,262144)


def contract_for(experiment,worker=None,max_steps=1000000000,corpus_dir=None,execution='server-balanced',source_plan=None):
    from extent.batch_recovery_161 import contract
    return contract(experiment,worker,max_steps,corpus_dir,execution,source_plan)


def session(args,c,hub):
    from extent.batch_recovery_161 import make_session
    return make_session(args,c,hub)


def run(args):
    import jax
    import jax.numpy as jnp
    from extent.config import load_config
    from extent.batch_recovery_161 import block_forward,Inspector,replay_report
    from extent.batch_recovery_161 import prepare,restore_source,base_coordinates,make_kernel,adam_counts
    from extent.batch_recovery_161 import optimizer
    from extent.competitor_tpu import token_nll,reconcile
    from extent.batch_recovery_161 import training_windows,windows_seen
    from extent.reset_streamed_gpu import make_step
    from extent.plateau_forensics import GuardedJit,plain,state_digest
    from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding,replicated_sharding
    from extent.reset_long_patient import Patient
    from extent.t4_scale_data import load_data
    from extent.hf_artifact_sync import artifact_config_from_env
    from scripts.m3q_peer_tpu_campaign import coordinate_layout,sharded_optimizer
    cfg,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    execution=getattr(args,'execution','t4-pair')
    c=contract_for(args.experiment,args.worker,args.max_steps,args.corpus_dir,execution,args.source_plan);horizon=c['horizon'];eval_steps=c['eval_steps'];hub=artifact_config_from_env()
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
        from extent.batch_recovery_161 import source_pin
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
            row['inherited_parent_adam_count']=c['parent']['branches'][name]['adam_count']
            row['initialization']='parent160 CURRENT and Adam preserved; batch6, separate window and optimizer clocks'
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
            if 'source_controls' not in row:
                row['source_controls']={p:evaluate(payload['parameters'],tokens[p],cursor) for p in ('A','B','C','D')}
                from extent.batch_recovery_161 import parent_replay
                row['parent_replay']=parent_replay(row['source_controls'],c['parent']['branches'][name])
                s.persist(local_only=True)
                if row['parent_replay'].get('passed') is False:
                    raise ValueError('parent CURRENT NLL replay failed before batch6 updates')
            if 'teacher_c4_reference' not in s.result:
                from extent.batch_recovery_161 import teacher_validation
                s.result['teacher_c4_reference']=teacher_validation(cfg,kernels[name],base,teacher_params,tokens['D'],s.check)
                s.persist(local_only=True)
            row['best_nll']=evaluate(payload['best'],tokens['A'],int(payload['best_step']))['nll'] if int(payload['best_step'])>=0 else None
            # Python callables are kept OUTSIDE JSON/checkpoint payloads.
            evaluators[name]=SimpleNamespace(kernel=evaluators[name],evaluate=evaluate,inspector=Inspector(cfg,base,columns,spec,s.check))
            if 'source_profile' not in row:row['source_profile']=evaluators[name].inspector.profile(payload['parameters'],tokens['A'][:1],cursor)
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
                        s.check();t=training_windows(tokens['FIT'],corpus,c,name,cursor)
                        begun=time.monotonic();cost=s.result['cost'];cost['attempted_updates']+=1;cost['input_tokens']+=t.size;cost['ce_targets']+=t.shape[0]*(t.shape[1]-1)
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
                        cursor+=1;row['step']=cursor;row['training_windows_seen']=windows_seen(c,name,cursor);row['batch6_updates']=cursor-row['source_step'];row.pop('final',None)
                        cost['accepted_updates']=cost.get('accepted_updates',0)+1;cost['accepted_input_tokens']=cost.get('accepted_input_tokens',0)+t.size;cost['accepted_ce_targets']=cost.get('accepted_ce_targets',0)+t.shape[0]*(t.shape[1]-1)
                        row['accepted_new_updates']=row.get('accepted_new_updates',0)+1;row['invocation_updates']+=1
                        row.setdefault('timing_samples',[]).append(seconds);row['timing_samples']=row['timing_samples'][-16:]
                        if row['invocation_updates']>=8:
                            median=float(np.median(row['timing_samples'][-6:]));remaining=max(0,c['first_goal_new_steps']-(cursor-row['source_step']))*median/3600
                            row['throughput']=dict(seconds_per_update=median,goal_new_steps=c['first_goal_new_steps'],estimated_training_hours_to_goal=remaining,estimated_training_hours_to_ceiling=max(0,horizon-(cursor-row['source_step']))*median/3600,ceiling=horizon,remaining_work_hours=max(0,s.deadline-time.monotonic())/3600,
                                input_tokens_per_second=t.size/median,teacher_input_tokens_per_second=t.size/median if spec['alpha'] else 0,
                                c4_windows_remaining=max(0,len(corpus.arrays['train'])//256-windows_seen(c,name,cursor)) if spec['data_stream']=='c4' else None,
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
    p.add_argument('--source-plan',required=True)
    p.add_argument('--corpus-dir',required=True,help='completed pinned CPU-prepared C4 directory')
    p.add_argument('--execution',choices=('server-balanced',),default='server-balanced')
    p.add_argument('--stop-file')
    p.add_argument('--work-root')
    p.add_argument('--max-local-gb',type=float,default=80.)
    p.add_argument('--min-free-gb',type=float,default=20.)
    p.add_argument('--experiment',type=int,choices=(161,),default=161)
    p.add_argument('--max-wall-hours',type=float,default=11.8)
    p.add_argument('--max-steps',type=int,default=1000000000,help='batch6 optimizer updates after parent CURRENT; data end or wall limit stops')
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




def main(argv=None):
    args=parser().parse_args(argv)
    if args.worker is not None:raise ValueError('single-GPU batch6 only')
    if args.plan_only:return contract_for(161,None,args.max_steps,args.corpus_dir,args.execution,args.source_plan)
    return run(args)

if __name__=='__main__':main()
