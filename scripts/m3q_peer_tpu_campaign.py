"""EXP133/134: competitor-informed sharded TPU campaigns in the notebook kernel."""
import argparse,gc,hashlib,json,re,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict,replace
from importlib.metadata import version
import numpy as np
import jax
import jax.numpy as jnp
from extent.config import load_config
from extent.competitor_recovery import arm_for,initialize,optimizer,prepare_base,phase_at,block_forward,EVAL_STEPS
from extent.competitor_tpu import make_step,token_nll,hidden,reconcile,training_batch
from extent.context_recovery import tiled_nll
from extent.plateau_forensics import GuardedJit,plain,state_digest
from extent.recovery_subspace import apply_corrections
from extent.initialization import initialize_sharded_optimizer_state
from extent.sharding import create_v5e_mesh,batch_sharding,named_sharding_tree,replicated_sharding
from extent.recovery_durability import RecoveryCheckpointStore,cloud_roundtrip
from extent.hf_artifact_sync import artifact_config_from_env,restore_artifact,upload_artifacts_together
from extent.campaign_checkpoint import write_json_atomic
from extent.t4_scale_data import load_data,training_index,PANELS
from scripts.m3q_rank_growth_115_campaign import load_patient,file_sha
from scripts.m3q_batch_recovery_campaign import session_budget
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout
from scripts.qwen_extended_horizon_campaign import _safe_notify
ROOT=Path(__file__).resolve().parents[1]
HORIZON=16384


def contract_for(config,experiment):
    arms=[arm_for(experiment,i) for i in range(3 if experiment==133 else 2)]
    files=('scripts/m3q_peer_tpu_campaign.py','extent/competitor_tpu.py','extent/competitor_recovery.py',
        'extent/pregate_recovery.py','extent/dynamic_recovery.py','extent/context_recovery.py',
        'extent/t4_scale_recovery.py','extent/plateau_forensics.py','extent/t4_scale_data.py',
        'extent/layers/mamba3.py','extent/layers/common.py','extent/recovery_durability.py',
        'extent/chunked_checkpoint.py','extent/stable_gradient_clip.py','extent/hf_artifact_sync.py',
        'scripts/m3q_rank_growth_115_campaign.py','extent/t4_patient.py',
        'extent/efficiency_sources.py','extent/recovery_subspace.py','extent/initialization.py','extent/sharding.py',
        'extent/qwen3_teacher.py','extent/pinned_pg19_manifest.py')
    return json.loads(json.dumps(dict(experiment=experiment,seed=123,model=asdict(config),arms=arms,
        source='EXP126 selected-best4096' if experiment==133 else 'EXP111 MAIN32768',
        source_pin=file_sha(ROOT/('results/EXP-133-source-pin.json' if experiment==133 else 'results/EXP-114-source-pin.json')),
        code={f:file_sha(ROOT/f) for f in files},horizon=HORIZON,eval_steps=EVAL_STEPS,
        panels=PANELS,train_input_tokens_per_update=256 if experiment==133 else 1024,ce_targets_per_update=255 if experiment==133 else 1023,block_updates=128,
        optimizer='freshAdam all arms; joint stableclip1;133 beta2.999/warmup256;134 beta2.95/warmup492/cosine floor.1; per-arm LR in recipe; source best has NO claimed inherited Adam',
        selection='A16 before full ramp; A64 after full ramp every registered endpoint; B/C report only; baseline step0 eligible for133; source/last/best contexts',
        checkpoint='current+Adam+best+best_step; local128; remote120min for133/240min134 and exit; total upload cap; no full frozen model copies',
        reference='arianraje b946069890837bbe9b5dd75e2d118a7e98d2d193: pre-gate norm, tanh angles, forward KL T1; adaptations NOT full 4B recipe reproduction',
        interpretation='single-seed exploratory, exposed panels; compare last common endpoint; source preparation and teacher passes separate')))


def summarize(r):
    return dict(status=r.get('status'),progress={a:dict(step=v.get('step'),durable_step=v.get('durable_step'),
        best_step=v.get('best_step'),throughput=v.get('throughput'),first_update_health=v.get('first_update_health'),failed=v.get('failed'),failed_health=v.get('failed_health'),first_update_diagnostic=v.get('first_update_diagnostic'),final=v.get('final')) for a,v in r.get('branches',{}).items()},
        cost=r.get('cost'),pending=r.get('pending_slots'),error=r.get('error'),report_sync_error=r.get('report_sync_error'))


class Session:
    def __init__(self,args,contract,hub):
        self.args,self.contract,self.hub=args,contract,hub;self.begun=time.monotonic()
        suffix='-'+args.run_tag if args.run_tag else ''
        self.prefix=f'experiments/exp{args.experiment}-peer-tpu'+suffix
        self.stem=f'extent-m3q-peer-{args.experiment}'+suffix
        self.checkpoint_signatures={}
        work,_=session_budget(args.max_wall_hours,args.save_reserve_minutes)
        self.deadline=self.begun+work;self.wall=self.begun+args.max_wall_hours*3600
        self.output=Path(args.output_dir);self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/f'{self.stem}.json';self.small=self.output/f'{self.stem}-summary.txt'
        if not self.path.exists():restore_artifact(self.path,f'{self.prefix}/latest.json',hub)
        self.result=json.loads(self.path.read_text()) if self.path.exists() else dict(contract=contract,branches={},
            sessions=[],pending_slots=[],cost={'attempted_updates':0,'input_tokens':0,'ce_targets':0,'teacher_input_tokens':0})
        if self.result['contract']!=contract:raise ValueError('peer TPU resume contract mismatch')
        runtime={n:version(n) for n in ('jax','jaxlib','flax','optax')}
        runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
        if 'source_runtime' in contract:
            source=contract['source_runtime']
            required=dict(source['versions'],matmul_precision=source['matmul_precision'],prng_impl=source['prng_impl'])
            if runtime!=required:raise ValueError('source EXP111 optimizer/math runtime differs')
        if self.result.get('runtime',runtime)!=runtime:raise ValueError('peer TPU optimizer/math runtime changed')
        self.result['runtime']=runtime;self.store=RecoveryCheckpointStore(args.checkpoint_dir,f'{self.prefix}/checkpoints',hub)
        self.next_sync=0.;self.invocation=dict(max_wall_hours=args.max_wall_hours,
            save_reserve_minutes=args.save_reserve_minutes,uploaded_bytes=0)
        self.result['sessions'].append(self.invocation);self.result.pop('error',None);self.result.pop('storage_error',None)
    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('registered work deadline')
    def emit(self,event):
        progress={a:r.get('step',0) for a,r in self.result['branches'].items()}
        self.result.setdefault('notifications',[]).append(dict(event=event,**_safe_notify(self.args.telegram,
            f'Extent peer TPU: {event}; status={self.result.get("status")}; steps={progress}; HF pending={len(self.result["pending_slots"])}')))
        print(f'peer TPU {event}',flush=True);self.persist(local_only=True)
    def checkpoint(self,arm,payload,step):
        if getattr(getattr(self,'args',None),'diagnostic_only',False):return
        pointer=self.store.directory(arm)/'checkpoint.json'
        signature=(int(step),int(payload['best_step']))
        if self.checkpoint_signatures.get(arm)==signature:return
        self.store.save(arm,dict(self.contract,arm=arm),payload,step)
        self.checkpoint_signatures[arm]=signature
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
        try:upload_artifacts_together([(self.path,f'{self.prefix}/latest.json'),(self.small,f'{self.prefix}/latest-summary.txt')],
            self.hub,commit_message='peer TPU rank growth metrics')
        except Exception as e:
            status=getattr(getattr(e,'response',None),'status_code',None)
            if status in (401,403):raise
            self.result['report_sync_error']=dict(error_type=type(e).__name__,http_status=status)
            self.next_sync=time.monotonic()+(3600 if status==429 else 120);write();return False
        self.result.pop('report_sync_error',None);self.next_sync=time.monotonic()+600
        self.invocation['uploaded_bytes']+=size;write();return True



def source126(s):
    from extent.t4_patient import read_parameter_chunks
    from extent.chunked_checkpoint import digest
    pin=json.loads((ROOT/'results/EXP-133-source-pin.json').read_text())
    hub=replace(s.hub,repo_id=pin['repo_id'],repo_type='dataset',revision=pin['revision'])
    directory=Path(s.args.state_dir)/'126-selected';directory.mkdir(parents=True,exist_ok=True)
    pointer=directory/'source.json'
    if not pointer.exists():restore_artifact(pointer,pin['path'],hub)
    if file_sha(pointer)!=pin['manifest_sha256']:raise ValueError('126 pinned manifest SHA differs')
    meta=json.loads(pointer.read_text())
    if meta!=pin['metadata']:raise ValueError('126 source metadata differs')
    # Re-label ONLY best parameters for existing audited chunk reader. No Adam
    # history is attached to best; all continuation arms get fresh Adam.
    subset=dict(meta,leaves=[dict(r,path=['parameters']+r['path'][1:]) for r in meta['leaves'] if r['path'][0]=='best'])
    prefix=pin['path'].rsplit('/',1)[0]
    step_row=next(r for r in meta['leaves'] if r['path']==['best_step'])
    if len(step_row['chunks'])!=1:raise ValueError('unexpected best-step layout')
    c=step_row['chunks'][0];file=directory/c['file']
    if not file.exists():restore_artifact(file,prefix+'/'+c['file'],hub)
    if digest(file)!=c['sha256'] or int(np.load(file).item())!=4096:raise ValueError('126 selected-best cursor differs')
    from extent.efficiency_sources import tree_digest
    q=read_parameter_chunks(subset,directory,prefix,hub,s.check)
    if tree_digest(q)!=pin['reference']['state_sha256']:raise ValueError('126 selected parameter tree SHA differs')
    return q,pin


class Diagnostics:
    def __init__(self,cfg,base,columns,spec,mesh,check):
        self.cfg,self.base,self.columns,self.spec,self.mesh,self.check=cfg,base,columns,spec,mesh,check
        self.loss=GuardedJit(lambda q,b,t,n:token_nll(cfg,b,q,t,columns,spec,n));self.profiles={}
    def evaluate(self,q,windows,step,tail=None):
        found=[]
        for w in windows:
            self.check();a=np.asarray(self.loss(q,self.base,jax.device_put(w[None],batch_sharding(self.mesh)),jnp.asarray(step,jnp.float32)))
            found.append(float(a.mean() if tail is None else a[:,-tail:].mean()))
        if not np.all(np.isfinite(found)):raise FloatingPointError('nonfinite NLL')
        return dict(nll=float(np.mean(found)),windows=found,targets_per_window=tail or windows.shape[1]-1)
    def context(self,q,windows,step):
        blocks=windows.reshape(-1,1024)
        return {str(n):self.evaluate(q,blocks[:,-n:],step,127) for n in (128,256,512,1024)}
    def profile(self,q,window,step):
        cfg=replace(self.cfg,remat_policy='none');t=jax.device_put(window[None],batch_sharding(self.mesh))
        h=self.base['embed_tokens']['embedding'][t].astype(getattr(jnp,cfg.compute_dtype));rows=[]
        phase=jnp.asarray(phase_at(self.spec,step),jnp.float32)
        for i in range(cfg.num_layers):
            self.check()
            if i not in self.profiles:self.profiles[i]=GuardedJit(lambda p,r,x,n,i=i:block_forward(cfg,i,p,r,x,self.columns,self.spec,n,details=True))
            h,_,row=self.profiles[i](self.base[f'layers_{i}'],q[str(i)],h,phase);rows.append(dict(layer=i,**plain(row)))
        ordinary=GuardedJit(lambda r,b,t,n:hidden(cfg,b,r,t,self.columns,self.spec,n))(q,self.base,t,jnp.asarray(step,jnp.float32))
        err=float(jnp.max(jnp.abs(h.astype(jnp.float32)-ordinary.astype(jnp.float32)))/jnp.maximum(jnp.max(jnp.abs(ordinary)),1e-30))
        if not np.isfinite(err) or err>.02:raise ValueError('profile/ordinary output mismatch')
        return dict(layers=rows,relative_output_discrepancy=err,tolerance=.02,accepted=True)


def coordinate_layout(q,mesh):
    # Legacy flatten/unflatten drops empty attention-coordinate dictionaries.
    # Rebuild through the original pytree, retaining every empty node.
    raw=trainable_layout(q,mesh)
    def lookup(path,_):
        node=raw
        for key in path:node=node[key.key]
        return node
    return jax.tree.map_with_path(lookup,q)


def sharded_optimizer(tx,q,layout,mesh):
    # multi_transform nests Adam states under PartitionState; the legacy helper
    # only handles top-level Adam and would REPLICATE their large mu/nu arrays.
    abstract=jax.eval_shape(tx.init,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),q))
    def leaf(path,_):
        for i,key in enumerate(path):
            if getattr(key,'name',None) in ('mu','nu'):
                node=layout
                for part in path[i+1:]:node=node[part.key]
                return node
        return replicated_sharding(mesh)
    state_layout=jax.tree.map_with_path(leaf,abstract)
    state=jax.jit(tx.init,in_shardings=(layout,),out_shardings=state_layout)(q)
    return state,state_layout


def initial_payload(cfg,base,current,old,columns,spec,mesh):
    q=initialize(SimpleNamespace(cfg=cfg,base=base,current=current,coordinates=old,columns=columns),spec)
    layout=coordinate_layout(q,mesh);q=jax.device_put(q,layout);tx=optimizer(q,spec)
    state,state_layout=sharded_optimizer(tx,q,layout,mesh)
    payload=dict(parameters=q,optimizer=state,best=q,best_step=jnp.asarray(-1,jnp.int32))
    layouts=dict(parameters=layout,optimizer=state_layout,best=layout,best_step=replicated_sharding(mesh))
    return payload,layouts,tx


def final_evaluations(s,rows,payloads,diags,tokens):
    # Cache by current AND selected-best hash/cursor: lost cloud tails cannot
    # retain endpoint measurements belonging to a future checkpoint.
    for name,row in rows.items():
        if name not in payloads:continue
        payload=payloads[name];diag=diags[name];cursor=row['step']
        signature=state_digest(payload['parameters'])
        best_step=int(payload['best_step']);best_sha=state_digest(payload['best'])
        final=row.setdefault('final',dict(current_step=cursor,current_sha256=signature,best_step=best_step,best_sha256=best_sha))
        if (final['current_sha256'],final.get('best_sha256'))!=(signature,best_sha):
            final=dict(current_step=cursor,current_sha256=signature,best_step=best_step,best_sha256=best_sha);row['final']=final
        candidates=[('last',payload['parameters'],cursor)]
        if best_step>=0:candidates.append(('best',payload['best'],best_step))
        for label,q,step in candidates:
            point=final.setdefault(label,dict(step=step,panels={}))
            for panel in ('A','B','C'):
                if panel not in point['panels']:point['panels'][panel]=diag.evaluate(q,tokens[panel],step);s.persist()
            if 'context' not in point:point['context']=diag.context(q,tokens['B'],step);s.persist()
            if 'profile' not in point:point['profile']=diag.profile(q,tokens['A'][0],step);s.persist()
        row['finalization_status']='completed';s.persist()


def record_endpoint(s,row,payload,diag,q,tokens,cursor,spec):
    if cursor not in EVAL_STEPS or any(v['step']==cursor for v in row['evaluations']):return
    ready=phase_at(spec,cursor)>=1.;point=dict(step=cursor,ramp_strength=phase_at(spec,cursor),selection_eligible=ready)
    panel=diag.evaluate(q,tokens['A'] if ready else tokens['A'][:16],cursor);point['A64' if ready else 'A16']=panel
    if ready:
        point['B32']=diag.evaluate(q,tokens['B'],cursor);point['C64']=diag.evaluate(q,tokens['C'],cursor)
        point['profile']=diag.profile(q,tokens['A'][0],cursor)
        if row['best_nll'] is None or panel['nll']<row['best_nll']:
            payload['best']=q;payload['best_step']=jnp.asarray(cursor,jnp.int32);row['best_step']=cursor;row['best_nll']=panel['nll']
    row['evaluations'].append(point);s.persist()


def diagnose_first_updates(s, rows, payloads, kernels, base, teacher_params, tokens, mesh, specs):
    """Replay a single proposal from each source without accepting ANY update."""
    for name, row in rows.items():
        if 'first_update_diagnostic' in row:continue
        s.check();payload=payloads[name];cursor=row['step'];spec=specs[name]
        t=training_batch(tokens['FIT'],cursor,s.args.experiment,spec['train_offset'])
        begun=time.monotonic();cost=s.result['cost']
        cost['attempted_updates']+=1;cost['input_tokens']+=t.size;cost['ce_targets']+=t.shape[0]*(t.shape[1]-1)
        if spec['alpha']:cost['teacher_input_tokens']+=t.size
        proposal,after,health=kernels[name](payload['parameters'],payload['optimizer'],base,teacher_params,
            jax.device_put(t,batch_sharding(mesh)),jnp.asarray(cursor,jnp.float32))
        health=plain(health)
        row['first_update_diagnostic']=dict(step=cursor,proposal_accepted=False,seconds=time.monotonic()-begun,
            training_batch_sha256=hashlib.sha256(t.tobytes()).hexdigest(),
            failed_stages=[k for k,v in health['stages'].items() if not v['finite']],health=health)
        del proposal,after
        # Persist diagnostics before notification or another arm compilation.
        s.persist(force=True,checkpoints=False);s.emit(name+' first-update diagnostic saved')
    s.result['status']='diagnostic_failed' if any(not r['first_update_diagnostic']['health']['finite'] for r in rows.values()) else 'diagnostic_passed'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=int,choices=(133,134),required=True)
    p.add_argument('--max-wall-hours',type=float,default=8.5)
    p.add_argument('--save-reserve-minutes',type=float,default=60)
    p.add_argument('--max-new-hf-gb',type=float,default=24)
    p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--state-dir',default='/dev/shm/extent-peer-source')
    p.add_argument('--checkpoint-dir',default='/dev/shm/extent-peer-checkpoints')
    p.add_argument('--dataset-cache-dir',default='/kaggle/working/extent-dataset-cache')
    p.add_argument('--qwen-cache-dir',default='/kaggle/working/qwen3-exp104-weights')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--sync-only',action='store_true')
    p.add_argument('--diagnostic-only',action='store_true',help='Compute and inspect one proposal per arm; never apply it')
    p.add_argument('--run-tag',default=None,help='Separate report/checkpoint namespace; 133 defaults to numerics-v2')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args(argv)
    if args.diagnostic_only and args.experiment!=133:raise ValueError('first-update diagnostic is for EXP133')
    if args.run_tag is None:args.run_tag='first-step-v2' if args.diagnostic_only else 'numerics-v2' if args.experiment==133 else ''
    if args.run_tag and not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}',args.run_tag):raise ValueError('invalid run tag')
    session_budget(args.max_wall_hours,args.save_reserve_minutes)
    if args.save_reserve_minutes<30:raise ValueError('at least30min reserve required')
    if not np.isfinite(args.max_new_hf_gb) or args.max_new_hf_gb<=0:raise ValueError('positive finite upload cap required')
    config,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml');contract=contract_for(config,args.experiment)
    contract['execution']=dict(diagnostic_only=args.diagnostic_only,run_tag=args.run_tag,health_schema='stages-v2')
    if args.plan_only:print(json.dumps(contract,indent=2));return contract
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    s=Session(args,contract,hub);payloads={};layouts={};diags={};kernels={};base=None;teacher_params=None
    if args.sync_only or s.result.get('status') in ('completed','diagnostic_passed','diagnostic_failed'):s.persist(force=True);return s.result
    s.result['status']='preflight';s.emit('started')
    try:
        if not s.persist(force=True):raise RuntimeError('HF report write preflight failed')
        tokens,meta=load_data(Path(args.dataset_cache_dir)/'peer-panels',s.check);s.result['data']=meta
        devices=list(jax.devices())
        if jax.process_count()!=1 or len(devices)!=8 or any(d.platform!='tpu' for d in devices):raise ValueError('single-host TPUv5e8 required')
        import psutil
        if psutil.virtual_memory().available<80*(1<<30):raise MemoryError('source reconstruction requires80GiB host RAM')
        mesh=create_v5e_mesh(devices)
        s.result['roundtrip']=cloud_roundtrip(Path(args.checkpoint_dir),s.prefix,hub,s.deadline)
        model,base,old,oldstate,teacher,tp=load_patient(config,mesh,s)
        tc=teacher.config
        del oldstate,teacher,model;gc.collect()
        columns=2*int(config.hidden_size*config.mamba.expand)+2*config.mamba.mimo_rank*config.mamba.groups*config.mamba.d_state
        # Original111 source replay happens even for133 before loading126 best.
        source_spec=arm_for(130,0)
        source_q=initialize(SimpleNamespace(cfg=config,base=base,current=base,coordinates=old,columns=columns),source_spec)
        source_q=jax.device_put(source_q,coordinate_layout(source_q,mesh))
        source_diag=Diagnostics(config,base,columns,source_spec,mesh,s.check)
        source=source_diag.evaluate(source_q,tokens['A'],0)
        expected=json.loads((ROOT/'results/EXP-115-source-validation.json').read_text())['windows']
        error=float(np.max(np.abs(np.asarray(source['windows'])-expected)))
        if error>.03:raise ValueError('111 sharded source/hook NLL differs')
        s.result['source111_replay']=dict(max_per_window_error=error,tolerance=.03,value=source)
        if args.experiment==133:
            selected,pin=source126(s)
            coordinates={f'layers_{i}/mamba/{projection}/kernel':selected[str(i)]['lora'][projection]
                for i in config.mamba_layer_indices for projection in ('in_proj','out_proj')}
            old=jax.device_put(coordinates,trainable_layout(coordinates,mesh))
            s.result['source126_pin']=dict(revision=pin['revision'],step=4096,manifest_sha256=pin['manifest_sha256'])
        current=apply_corrections(base,old,protected_input_columns=columns)
        teacher_params=jax.device_put(tp,named_sharding_tree(tp,mesh)) if args.experiment==134 else {}
        del tp,source_q,source_diag;gc.collect()
        sources=s.result.setdefault('source_controls',{})
        for spec in contract['arms']:
            name=spec['name'];template,layouts[name],tx=initial_payload(config,base,current,old,columns,spec,mesh)
            saved=s.store.restore(name,dict(contract,arm=name),template,layouts[name])
            row=s.result['branches'].setdefault(name,dict(step=0,history=[],evaluations=[]))
            if saved is None and row.get('durable_step',0):raise FileNotFoundError('claimed durable state unavailable')
            payload=saved[0] if saved else template;cursor=saved[1]['step'] if saved else 0
            if row.get('durable_step',0)>cursor:raise ValueError('durable cursor ahead of restored state')
            payloads[name]=payload;best_step=int(payload['best_step']);reconcile(row,cursor,best_step)
            diag=Diagnostics(config,base,columns,spec,mesh,s.check);diags[name]=diag
            kernels[name]=GuardedJit(make_step(config,tc,columns,spec,tx,detailed=args.diagnostic_only))
            row['coordinates']=sum(x.size for x in jax.tree.leaves(payload['parameters']))
            row['payload_bytes']=sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(payload))
            # Initial controls never use restored/current coordinates.
            if name not in sources:
                q0=template['parameters'];control=dict(panels={k:diag.evaluate(q0,tokens[k],0) for k in ('A','B','C')})
                control['profile']=diag.profile(q0,tokens['A'][0],0);control['context']=diag.context(q0,tokens['B'],0)
                if args.experiment==133:
                    errors=[float(np.max(np.abs(np.asarray(control['panels'][k]['windows'])-np.asarray(pin['reference']['panels'][k]['windows'])))) for k in ('A','B','C')]
                    if max(errors)>.03:raise ValueError('126 T4-to-TPU selected checkpoint replay differs')
                    control['source126_max_per_window_error']=max(errors)
                sources[name]=control;s.persist()
            if args.experiment==133 and best_step<0:
                payload['best']=template['parameters'];payload['best_step']=jnp.asarray(0,jnp.int32)
                best_step=0;row['best_step']=0
            if best_step>=0:row['best_nll']=diag.evaluate(payload['best'],tokens['A'],best_step)['nll']
            else:row['best_nll']=None
            if not row['evaluations'] and cursor==0:
                row['evaluations'].append(dict(step=cursor,A16=diag.evaluate(payload['parameters'],tokens['A'][:16],cursor)))
            del template,saved
        del current;gc.collect()
        rows=s.result['branches'];s.result['status']='running';s.persist();last_sync=time.monotonic();last_notice=time.monotonic()
        specs={v['name']:v for v in contract['arms']}
        remote_interval=7200 if args.experiment==133 else 14400
        try:
            if args.diagnostic_only:
                diagnose_first_updates(s,rows,payloads,kernels,base,teacher_params,tokens,mesh,specs)
            while not args.diagnostic_only and any(row['step']<HORIZON and not row.get('failed') for row in rows.values()):
                eligible=[name for name,row in rows.items() if row['step']<HORIZON and not row.get('failed')]
                name=min(eligible,key=lambda n:rows[n]['step']);row=rows[name];spec=specs[name];diag=diags[name]
                payload=payloads[name];q,state=payload['parameters'],payload['optimizer'];cursor=row['step']
                target=min(HORIZON,((cursor//128)+1)*128)
                try:
                    record_endpoint(s,row,payload,diag,q,tokens,cursor,spec)
                    while cursor<target:
                        s.check();t=training_batch(tokens['FIT'],cursor,args.experiment,spec['train_offset'])
                        begun=time.monotonic();cost=s.result['cost'];cost['attempted_updates']+=1;cost['input_tokens']+=t.size;cost['ce_targets']+=t.shape[1]-1
                        if spec['alpha']:cost['teacher_input_tokens']+=t.size
                        proposal,after,health=kernels[name](q,state,base,teacher_params,jax.device_put(t,batch_sharding(mesh)),jnp.asarray(cursor,jnp.float32))
                        row['training_seconds']=row.get('training_seconds',0)+time.monotonic()-begun
                        if not bool(health['finite']):
                            row['failed_health']=plain(health);s.persist(local_only=True)
                            raise FloatingPointError('nonfinite proposal rejected; see failed_health.stages')
                        q=jax.device_put(proposal,layouts[name]['parameters']);state=jax.device_put(after,layouts[name]['optimizer'])
                        cursor+=1;row['step']=cursor;row.pop('final',None)
                        cost['accepted_updates']=cost.get('accepted_updates',0)+1
                        cost['accepted_input_tokens']=cost.get('accepted_input_tokens',0)+t.size
                        if cursor%32==0:row['history'].append(dict(step=cursor,**plain(health)));s.persist(local_only=True)
                        record_endpoint(s,row,payload,diag,q,tokens,cursor,spec)
                except FloatingPointError as exc:row['failed']=dict(step=cursor,reason=str(exc));s.emit(name+' numerical failure')
                finally:
                    payload.update(parameters=q,optimizer=state);s.checkpoint(name,payload,cursor)
                if time.monotonic()-last_sync>=remote_interval:
                    s.persist(force=True);last_sync=time.monotonic()
                    if s.result.get('storage_error'):raise TimeoutError('upload cap reached')
                if time.monotonic()-last_notice>=1800:s.emit('progress');last_notice=time.monotonic()
            if not args.diagnostic_only:s.result['status']='arm_failed' if any(r.get('failed') for r in rows.values()) else 'completed'
        except TimeoutError:s.result['status']='deadline_partial';s.emit('training deadline')
        # Save FIRST, before spending the remaining reserve on diagnostics.
        s.persist(force=True);s.deadline=s.wall-300
        try:
            if not args.diagnostic_only:final_evaluations(s,rows,payloads,diags,tokens)
        except TimeoutError:s.result['status']='finalization_partial'
        common=min(r['step'] for r in rows.values())
        endpoint_sets=[{v['step'] for v in r['evaluations'] if v.get('selection_eligible')} for r in rows.values()]
        common_evals=sorted(set.intersection(*endpoint_sets)) if endpoint_sets else []
        s.result['comparison']=dict(last_common_registered_endpoint=common_evals[-1] if common_evals else None,
            minimum_latest_cursor=common,latest_steps={k:r['step'] for k,r in rows.items()},
            interpretation='compare common endpoint; arbitrary final steps may differ, B/C never select')
        s.result['accelerator_memory']=[plain(d.memory_stats() or {}) for d in devices]
    except TimeoutError as exc:s.result['status']='deadline_partial';s.result['error']=str(exc)
    except Exception as exc:
        s.result.update(status='failed',error=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc());s.emit('failed');raise
    finally:
        for name,payload in payloads.items():s.checkpoint(name,payload,s.result['branches'][name]['step'])
        s.persist(force=True)
        if s.result.get('storage_error'):s.result['status']='storage_partial';s.persist(local_only=True)
        s.emit('finished and saved' if not s.result['pending_slots'] and not s.result.get('report_sync_error') else 'finished; HF pending')
    return s.result


if __name__=='__main__':main()
