"""EXP124: paired short/long-context CE recovery on one TPU v5e8."""
import argparse,gc,json,sys,time,traceback
from pathlib import Path
from dataclasses import asdict
from importlib.metadata import version
import jax
import jax.numpy as jnp
import numpy as np
from extent.config import load_config
from extent.context_recovery import (ARMS,HORIZON,EVAL_STEPS,SPEC,initial_coordinates,
    training_batch,hidden,tiled_nll,make_step,reconcile,optimizer,gain_report)
from extent.rank_growth import materialize,ARMS as GROWTH_ARMS
from extent.rank_growth_data import load_tokens
from extent.initialization import initialize_sharded_optimizer_state
from extent.sharding import create_v5e_mesh,batch_sharding,named_sharding_tree,replicated_sharding
from extent.recovery_durability import RecoveryCheckpointStore,cloud_roundtrip
from extent.hf_artifact_sync import artifact_config_from_env,restore_artifact,upload_artifacts_together
from extent.campaign_checkpoint import write_json_atomic
from extent.plateau_forensics import GuardedJit,plain,state_digest
from extent.t4_scale_recovery import block_forward
from scripts.m3q_rank_growth_115_campaign import load_patient,file_sha
from scripts.m3q_batch_recovery_campaign import session_budget
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout,evaluate
from scripts.qwen_extended_horizon_campaign import _safe_notify
ROOT=Path(__file__).resolve().parents[1]
PREFIX='experiments/exp124-context-recovery'
STEM='extent-m3q-context-124'

def contract_for(config):
    from scripts.m3q_rank_growth_115_campaign import contract_for as source_contract
    source=source_contract(config)
    files=('extent/context_recovery.py','scripts/m3q_context_recovery_124_campaign.py',
        'extent/t4_scale_recovery.py','extent/plateau_forensics.py',
        'extent/recovery_durability.py','extent/chunked_checkpoint.py',
        'extent/initialization.py','extent/hf_artifact_sync.py',
        'extent/stable_gradient_clip.py','extent/rank_growth_data.py')
    return json.loads(json.dumps(dict(experiment=124,seed=123,model=asdict(config),
        source_contract=source,source_runtime=source['source_runtime'],arms=ARMS,
        horizon=HORIZON,eval_steps=EVAL_STEPS,block_updates=256,
        train_input_tokens_per_update=1024,ce_targets_per_update=1020,
        long_omitted_loss_indices=[255,511,767],optimizer='fresh Adam both arms; R32 LR3e-4; gains LR.01; warmup256; joint stable clip1',
        context_lengths=[128,256,512,1024],context_tail_targets=127,
        panels='reused EXP115 validation, Wiki test and PG19; contiguous blocks; NOT independent tests',
        interpretation='single seed exploratory context intervention; comparisons only at matched updates',
        code={f:file_sha(ROOT/f) for f in files})))

def summarize(r):
    return dict(status=r.get('status'),progress={a:dict(step=b.get('step'),durable_step=b.get('durable_step'),
        failed=b.get('failed'),last_evaluation=(b.get('evaluations') or [None])[-1],final=b.get('final'))
        for a,b in r.get('branches',{}).items()},cost=r.get('cost'),pending=r.get('pending_slots'),
        error=r.get('error'),report_sync_error=r.get('report_sync_error'))

class Session:
    def __init__(self,args,contract,hub):
        self.args,self.contract,self.hub=args,contract,hub;self.begun=time.monotonic()
        work,_=session_budget(args.max_wall_hours,args.save_reserve_minutes)
        self.deadline=self.begun+work;self.wall=self.begun+args.max_wall_hours*3600
        self.output=Path(args.output_dir);self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/f'{STEM}.json';self.small=self.output/f'{STEM}-summary.txt'
        if not self.path.exists():restore_artifact(self.path,f'{PREFIX}/latest.json',hub)
        self.result=json.loads(self.path.read_text()) if self.path.exists() else dict(contract=contract,branches={},
            sessions=[],pending_slots=[],cost={'attempted_updates':0,'input_tokens':0,'ce_targets':0})
        if self.result['contract']!=contract:raise ValueError('EXP124 resume contract mismatch')
        runtime={n:version(n) for n in ('jax','jaxlib','flax','optax')}
        runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
        if 'source_runtime' in contract:
            source=contract['source_runtime']
            required=dict(source['versions'],matmul_precision=source['matmul_precision'],prng_impl=source['prng_impl'])
            if runtime!=required:raise ValueError('source EXP111 optimizer/math runtime differs')
        if self.result.get('runtime',runtime)!=runtime:raise ValueError('EXP124 optimizer/math runtime changed')
        self.result['runtime']=runtime;self.store=RecoveryCheckpointStore(args.checkpoint_dir,f'{PREFIX}/checkpoints',hub)
        self.next_sync=0.;self.invocation=dict(max_wall_hours=args.max_wall_hours,
            save_reserve_minutes=args.save_reserve_minutes,uploaded_bytes=0)
        self.result['sessions'].append(self.invocation);self.result.pop('error',None);self.result.pop('storage_error',None)
    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('registered work deadline')
    def emit(self,event):
        progress={a:r.get('step',0) for a,r in self.result['branches'].items()}
        self.result.setdefault('notifications',[]).append(dict(event=event,**_safe_notify(self.args.telegram,
            f'Extent EXP124: {event}; status={self.result.get("status")}; steps={progress}; HF pending={len(self.result["pending_slots"])}')))
        print(f'EXP124 {event}',flush=True);self.persist(local_only=True)
    def checkpoint(self,arm,payload,step):
        pointer=self.store.directory(arm)/'checkpoint.json'
        if pointer.exists() and json.loads(pointer.read_text())['step']==step:return
        self.store.save(arm,dict(self.contract,arm=arm),payload,step)
        if arm not in self.result['pending_slots']:self.result['pending_slots'].append(arm)
    def sync_states(self):
        for arm in list(self.result['pending_slots']):
            directory=self.store.directory(arm)
            if not (directory/'checkpoint.json').exists():
                self.result['pending_slots'].remove(arm);continue # destroyed local state; restore will use remote cursor
            meta=json.loads((directory/'checkpoint.json').read_text())
            receipt=directory/f'{meta["generation"]}.uploaded.json'
            before=set(json.loads(receipt.read_text())['files']) if receipt.exists() else set()
            needed=sum(c['bytes'] for r in meta['leaves'] for c in r['chunks'] if c['file'] not in before)
            if needed+self.invocation['uploaded_bytes']>self.args.max_new_hf_gb*1e9:
                self.result['storage_error']='invocation payload cap';return False
            success=self.store.sync(arm,deadline=self.wall-60)
            after=set(json.loads(receipt.read_text())['files']) if receipt.exists() else set()
            self.invocation['uploaded_bytes']+=sum(c['bytes'] for r in meta['leaves'] for c in r['chunks'] if c['file'] in after-before)
            if success:
                self.result['pending_slots'].remove(arm);self.result['branches'][arm]['durable_step']=meta['step']
        return not self.result['pending_slots']
    def persist(self,local_only=False,force=False,checkpoints=True):
        self.invocation['hours']=(time.monotonic()-self.begun)/3600
        def write():
            write_json_atomic(self.path,self.result);self.small.write_text(json.dumps(summarize(self.result),indent=2),encoding='utf-8')
        write()
        if local_only:return False
        if force and checkpoints:self.sync_states();write()
        if time.monotonic()<self.next_sync and (not force or self.result.get('report_sync_error')):return False
        size=self.path.stat().st_size+self.small.stat().st_size
        if size+self.invocation['uploaded_bytes']>self.args.max_new_hf_gb*1e9:
            self.result['report_sync_error']='invocation payload cap';write();return False
        try:upload_artifacts_together([(self.path,f'{PREFIX}/latest.json'),(self.small,f'{PREFIX}/latest-summary.txt')],
            self.hub,commit_message='EXP124 rank growth metrics')
        except Exception as e:
            status=getattr(getattr(e,'response',None),'status_code',None)
            if status in (401,403):raise
            self.result['report_sync_error']=dict(error_type=type(e).__name__,http_status=status)
            self.next_sync=time.monotonic()+(3600 if status==429 else 120);write();return False
        self.result.pop('report_sync_error',None);self.next_sync=time.monotonic()+600
        self.invocation['uploaded_bytes']+=size;write();return True


def init_payload(config,old,mesh):
    q=initial_coordinates(config,old);layout=trainable_layout(q,mesh)
    q=jax.device_put(q,layout);tx=optimizer(q,SPEC)
    initial=initialize_sharded_optimizer_state(tx,q,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),q),layout,mesh)
    return dict(parameters=q,optimizer=initial.opt_state),dict(parameters=layout,optimizer=initial.layout),tx

class Diagnostics:
    def __init__(self,config,base,columns,mesh,check):
        self.config,self.base,self.columns,self.mesh,self.check=config,base,columns,mesh,check
        self.batch=batch_sharding(mesh)
        self.loss=GuardedJit(lambda q,b,t:tiled_nll(config,b,hidden(config,b,q,t,columns),t))
        self.profiles={}
    def evaluate(self,q,windows,tail=None):
        found=[]
        for w in windows:
            self.check();loss=np.asarray(self.loss(q,self.base,jax.device_put(w[None],self.batch)))
            found.append(float(loss.mean() if tail is None else loss[:,-tail:].mean()))
        if not np.all(np.isfinite(found)):raise FloatingPointError('nonfinite evaluation')
        return dict(nll=float(np.mean(found)),windows=found,targets_per_window=tail or (windows.shape[1]-1))
    def context(self,q,windows):
        blocks=windows.reshape(-1,1024)
        return {str(length):self.evaluate(q,blocks[:,-length:],127) for length in (128,256,512,1024)}
    def profile(self,q,window):
        from dataclasses import replace
        cfg=replace(self.config,remat_policy='none')
        t=jax.device_put(window[None],self.batch)
        h=self.base['embed_tokens']['embedding'][t].astype(getattr(jnp,cfg.compute_dtype));rows=[]
        for layer in range(cfg.num_layers):
            self.check()
            if layer not in self.profiles:
                self.profiles[layer]=GuardedJit(lambda p,r,h,layer=layer:block_forward(cfg,layer,p,r,h,self.columns,details=True))
            h,_,report=self.profiles[layer](self.base[f'layers_{layer}'],q.get(str(layer),{}),h)
            rows.append(dict(layer=layer,**plain(report)))
        # Compare the instrumented executable with the actual training forward.
        ordinary=GuardedJit(lambda q,b,t:hidden(cfg,b,q,t,self.columns))(q,self.base,t)
        relative=float(jnp.max(jnp.abs(h.astype(jnp.float32)-ordinary.astype(jnp.float32)))/jnp.maximum(jnp.max(jnp.abs(ordinary)),1e-30))
        return dict(layers=rows,relative_output_discrepancy=relative,
            accepted=bool(np.isfinite(relative) and relative<=.02),tolerance=.02,
            interpretation='Instrumented BF16 profile; only accepted if <=.02 relative discrepancy; NOT recurrence trace')


def finalize(s,payloads,diagnostics,tokens,steps):
    # Fixed last-state diagnostics, no selection on previously exposed test panels.
    source=s.result.setdefault('source_diagnostics',{})
    work=[('source',s.initial_q,source)]
    for a in ARMS:
        row=s.result['branches'][a]
        signature=state_digest(jax.device_get(payloads[a]['parameters']))
        if row.get('final',{}).get('state_sha256')!=signature:row['final']=dict(state_sha256=signature)
    work += [(a,payloads[a]['parameters'],s.result['branches'][a].setdefault('final',{})) for a in ARMS]
    for name,q,out in work:
        if name!='source':out['step']=steps[name]
        for panel in ('validation','locked_wiki','locked_pg19'):
            key=panel+'_context'
            if key not in out:
                out[key]=diagnostics.context(q,tokens[panel]);s.persist()
        if 'profile1024' not in out:
            out['profile1024']=diagnostics.profile(q,tokens['validation'].reshape(-1,1024)[0]);s.persist()
        out['gains']=gain_report(q)
    s.result['comparison']=dict(matched_updates=steps['SHORT']==steps['LONG'],
        evaluated_steps=steps,latest_steps={a:s.result['branches'][a]['step'] for a in ARMS},
        limitation='One seed; reused validation/test panels; source preparation excluded; unequal-step results not a controlled comparison')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='/kaggle/working/output')
    p.add_argument('--state-dir',default='/dev/shm/extent-exp124-source')
    p.add_argument('--checkpoint-dir',default='/dev/shm/extent-exp124-checkpoints')
    p.add_argument('--dataset-cache-dir',default='/kaggle/working/extent-dataset-cache')
    p.add_argument('--qwen-cache-dir',default='/kaggle/working/qwen3-exp104-weights')
    p.add_argument('--max-wall-hours',type=float,default=8.8)
    p.add_argument('--save-reserve-minutes',type=float,default=60)
    p.add_argument('--max-new-hf-gb',type=float,default=2)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--sync-only',action='store_true')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args(sys.argv[1:] if argv is None else argv)
    session_budget(args.max_wall_hours,args.save_reserve_minutes)
    if args.save_reserve_minutes<30:raise ValueError('at least30min reserve required')
    if not np.isfinite(args.max_new_hf_gb) or args.max_new_hf_gb<=0:raise ValueError('positive finite upload cap required')
    config,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml');contract=contract_for(config)
    if args.plan_only:print(json.dumps(contract,indent=2));return contract
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    s=Session(args,contract,hub);payloads={};layouts={};diagnostics=None
    if args.sync_only or s.result.get('status')=='completed':s.persist(force=True);return s.result
    s.result['status']='preflight';s.emit('started')
    try:
        if not s.persist(force=True):raise RuntimeError('HF report write preflight failed')
        tokens,meta=load_tokens(args.dataset_cache_dir)
        if meta['panels']!=contract['source_contract']['data']:raise ValueError('pinned data SHA mismatch')
        s.result['data_provenance']=meta;s.check()
        s.result['roundtrip']=cloud_roundtrip(Path(args.checkpoint_dir),PREFIX,hub,s.deadline)
        devices=list(jax.devices())
        if jax.process_count()!=1 or len(devices)!=8 or any(d.platform!='tpu' for d in devices):raise ValueError('single-host TPU v5e8 required')
        import psutil
        if psutil.virtual_memory().available<80*(1<<30):raise MemoryError('source reconstruction needs80GiB host RAM')
        mesh=create_v5e_mesh(devices);model,base,old,source_state,teacher,tp=load_patient(config,mesh,s)
        del source_state;gc.collect()
        columns=2*int(config.hidden_size*config.mamba.expand)+2*config.mamba.mimo_rank*config.mamba.groups*config.mamba.d_state
        s.result['source_coordinates_hash']=state_digest(jax.device_get(old))
        original=materialize(base,{'old':old},columns,GROWTH_ARMS['R32'],0)
        ref=json.loads((ROOT/'results/EXP-115-source-validation.json').read_text())
        found=evaluate(model,original,tokens['validation'],batch_sharding(mesh))
        error=float(np.max(np.abs(np.asarray(found['windows'])-ref['windows'])))
        if error>1e-4:raise ValueError('canonical EXP111 source validation differs')
        s.result['canonical_source_parity']=error;del original
        diagnostics=Diagnostics(config,base,columns,mesh,s.check)
        s.initial_q=initial_coordinates(config,old)
        s.initial_q=jax.device_put(s.initial_q,trainable_layout(s.initial_q,mesh))
        hook=diagnostics.evaluate(s.initial_q,tokens['validation'])
        error=float(np.max(np.abs(np.asarray(hook['windows'])-found['windows'])))
        if error>.01:raise ValueError('gain1/tiled-head source NLL parity failed')
        s.result['gain1_hook_parity']=dict(max_window_nll_error=error,tolerance=.01)
        # Original Qwen context control evaluated on the SAME final127 targets.
        # These panels are already exposed historical validation, not locked confirmation.
        baseline=s.result.setdefault('qwen_context',{})
        teacher_params=jax.device_put(tp,named_sharding_tree(tp,mesh))
        from extent.plateau_forensics import losses
        teacher_loss=GuardedJit(lambda p,t:losses(teacher.apply({'params':p},t),t,127))
        blocks=tokens['validation'].reshape(-1,1024)
        for length in (128,256,512,1024):
            if str(length) in baseline:continue
            values=[]
            for w in blocks[:,-length:]:
                s.check();values.append(float(teacher_loss(teacher_params,jax.device_put(w[None],batch_sharding(mesh)))))
            if not np.all(np.isfinite(values)):raise FloatingPointError('Qwen baseline nonfinite')
            baseline[str(length)]=dict(nll=float(np.mean(values)),windows=values);s.persist()
        del teacher_params,teacher_loss,tp,teacher;gc.collect()
        for arm in ARMS:
            template,layouts[arm],tx=init_payload(config,old,mesh)
            saved=s.store.restore(arm,dict(contract,arm=arm),template,layouts[arm])
            row=s.result['branches'].setdefault(arm,dict(step=0,evaluations=[],history=[],attempted_updates=0))
            if saved is None and row.get('durable_step',0):raise FileNotFoundError('durable arm state unavailable')
            payloads[arm]=saved[0] if saved else template
            reconcile(row,saved[1]['step'] if saved else 0)
            del template,saved
        paired=None;paired_steps=None
        if len({s.result['branches'][a]['step'] for a in ARMS})==1:
            paired={a:dict(parameters=payloads[a]['parameters']) for a in ARMS}
            paired_steps={a:s.result['branches'][a]['step'] for a in ARMS}
        step=GuardedJit(make_step(config,columns,tx))
        last_sync=time.monotonic();s.result['status']='running';s.persist()
        try:
            while any(s.result['branches'][a]['step']<HORIZON and not s.result['branches'][a].get('failed') for a in ARMS):
                eligible=[a for a in ARMS if s.result['branches'][a]['step']<HORIZON and not s.result['branches'][a].get('failed')]
                arm=min(eligible,key=lambda a:s.result['branches'][a]['step'])
                row=s.result['branches'][arm];cursor=row['step'];target=min(HORIZON,((cursor//256)+1)*256)
                q,state=payloads[arm]['parameters'],payloads[arm]['optimizer']
                try:
                    while cursor<target:
                        s.check();w,mask=training_batch(tokens['train'],cursor,arm)
                        begun=time.monotonic()
                        proposal,after,health=step(q,state,base,jax.device_put(w,batch_sharding(mesh)),jax.device_put(mask,batch_sharding(mesh)))
                        elapsed=time.monotonic()-begun
                        row['attempted_updates']+=1;s.result['cost']['attempted_updates']+=1
                        s.result['cost']['input_tokens']+=1024;s.result['cost']['ce_targets']+=1020
                        row['training_seconds']=row.get('training_seconds',0)+elapsed
                        row['last_health']=plain(health)
                        if not bool(health['finite']):raise FloatingPointError('nonfinite proposal rejected')
                        row.pop('final',None)
                        q=jax.device_put(proposal,layouts[arm]['parameters']);state=jax.device_put(after,layouts[arm]['optimizer'])
                        cursor+=1;row['step']=cursor
                        if cursor%32==0:
                            row['history'].append(dict(step=cursor,seconds=elapsed,**plain(health)));s.persist(local_only=True)
                        if cursor%128==0:
                            payloads[arm]=dict(parameters=q,optimizer=state);s.checkpoint(arm,payloads[arm],cursor)
                    if cursor in EVAL_STEPS and not any(e['step']==cursor for e in row['evaluations']):
                        row['evaluations'].append(dict(step=cursor,input_tokens=1024*cursor,ce_targets=1020*cursor,
                            A16=diagnostics.evaluate(q,tokens['validation'][:16])))
                    row['gains']=gain_report(q)
                except (FloatingPointError,MemoryError) as e:
                    row['failed']=dict(step=cursor,error_type=type(e).__name__,reason=str(e));s.emit(f'{arm} stopped')
                finally:
                    payloads[arm]=dict(parameters=q,optimizer=state);s.checkpoint(arm,payloads[arm],cursor)
                if len({s.result['branches'][a]['step'] for a in ARMS})==1:
                    paired={a:dict(parameters=payloads[a]['parameters']) for a in ARMS}
                    paired_steps={a:s.result['branches'][a]['step'] for a in ARMS}
                s.emit(f'{arm} step={cursor}/{HORIZON}')
                if time.monotonic()-last_sync>=5400 or all(s.result['branches'][a]['step']%4096==0 for a in ARMS):
                    s.persist(force=True);last_sync=time.monotonic()
                    if s.result.get('storage_error'):raise TimeoutError('invocation upload cap; save local state')
                else:s.persist()
        except TimeoutError:
            s.result['status']='deadline_partial';s.emit('training deadline; final diagnostics and save')
        # Leave at least half the reserve for upload. Completed training can use
        # the remaining work time as well. Source/last panels cached incrementally.
        s.deadline=s.wall-max(900,args.save_reserve_minutes*30)
        s.result['status']='finalizing' if s.result.get('status')!='deadline_partial' else 'deadline_partial'
        try:
            final_payloads=paired if paired is not None else payloads
            final_steps=paired_steps if paired_steps is not None else {a:s.result['branches'][a]['step'] for a in ARMS}
            finalize(s,final_payloads,diagnostics,tokens,final_steps)
            done=all(s.result['branches'][a]['step']==HORIZON and not s.result['branches'][a].get('failed') for a in ARMS)
            failed=any(s.result['branches'][a].get('failed') for a in ARMS)
            s.result['status']='completed' if done else ('arm_failed' if failed else 'deadline_partial')
        except TimeoutError:s.result['status']='finalization_partial'
        s.result['accelerator_memory']=[plain(d.memory_stats() or {}) for d in devices]
    except TimeoutError as e:s.result.update(status='deadline_partial',error=str(e));s.emit('deadline')
    except Exception as e:
        s.result.update(status='failed',error=str(e),error_type=type(e).__name__,traceback=traceback.format_exc());s.emit('failed');raise
    finally:
        for arm,payload in payloads.items():
            s.checkpoint(arm,payload,s.result['branches'][arm]['step'])
        s.persist(force=True)
        while (s.result['pending_slots'] or s.result.get('report_sync_error')) and time.monotonic()<s.wall-60:
            if s.result.get('storage_error') or s.result.get('report_sync_error')=='invocation payload cap':break
            retry=[]
            if s.result['pending_slots']:retry.append(s.store.next_sync)
            if s.result.get('report_sync_error'):retry.append(s.next_sync)
            next_retry=min(retry)
            if next_retry>=s.wall-60:break
            time.sleep(min(60,max(1,next_retry-time.monotonic())))
            s.persist(force=True)
        if s.result.get('storage_error') or s.result.get('report_sync_error')=='invocation payload cap':
            s.result['status']='storage_partial';s.persist(local_only=True)
        s.emit('finished and saved' if not s.result['pending_slots'] and not s.result.get('report_sync_error') else 'finished; HF pending')
    return s.result

if __name__=='__main__':main()
