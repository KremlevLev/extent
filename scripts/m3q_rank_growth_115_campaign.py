"""EXP115: single-seed aggressive rank-growth search from durable EXP111."""
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
from dataclasses import replace,asdict
from importlib.metadata import version
import jax
import jax.numpy as jnp
import numpy as np
from extent import HybridForCausalLM
from extent.config import load_config
from extent.rank_growth import ARMS,SEED,PILOT,FINAL,initialize,materialize,optimizers,make_step,select,potential_coordinate_count
from extent.rank_growth_data import load_tokens,PANELS,PG19
from extent.efficiency_recovery import optimizer
from extent.efficiency_sources import ReadOnlySourceStore,tree_digest
from extent.recovery_subspace import initialize_corrections,coordinate_count
from extent.initialization import (abstract_parameter_tree,initialize_sharded_parameters,
    initialize_sharded_optimizer_state)
from extent.sharding import create_v5e_mesh,batch_sharding,named_sharding_tree,replicated_sharding
from extent.recovery_durability import RecoveryCheckpointStore,cloud_roundtrip
from extent.hf_artifact_sync import artifact_config_from_env,restore_artifact,upload_artifacts_together
from extent.campaign_checkpoint import write_json_atomic
from extent.plateau_forensics import GuardedJit,state_digest,plain
from scripts.m3q_efficiency_campaign import contract_for as legacy_contract
from scripts.m3q_batch_recovery_campaign import session_budget
from scripts.m3q_paper_decoder_recovery_campaign import trainable_layout,evaluate
from scripts import m3q_subspace_engine as engine
from scripts.qwen_extended_horizon_campaign import _safe_notify

ROOT=Path(__file__).resolve().parents[1]
PREFIX='experiments/exp115-rank-growth'
STEM='extent-m3q-rank-growth-115'
EVAL_STEPS=(0,512,1024,2048,3072,4096,8192,12288,16384)

def file_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def contract_for(config):
    pin=json.loads((ROOT/'results/EXP-114-source-pin.json').read_text())
    if legacy_contract(config,111)!=pin['source_contract']:raise ValueError('EXP111 source dependencies changed')
    files=('extent/rank_growth.py','extent/rank_growth_data.py','scripts/m3q_rank_growth_115_campaign.py',
        'extent/pinned_pg19_manifest.py','extent/plateau_forensics.py','extent/plateau_data.py',
        'extent/chunked_checkpoint.py','extent/recovery_durability.py','extent/initialization.py',
        'extent/hf_artifact_sync.py','extent/stable_gradient_clip.py')
    return json.loads(json.dumps(dict(experiment=115,seed=SEED,model=asdict(config),
        source_pin_sha=file_sha(ROOT/'results/EXP-114-source-pin.json'),source_revision=pin['revision'],source_runtime=pin['runtime'],
        source_validation_sha=file_sha(ROOT/'results/EXP-115-source-validation.json'),
        legacy_source_contract=pin['source_contract'],data=json.loads((ROOT/'results/EXP-115-data-preflight.json').read_text())['panels'],
        panels=PANELS,pg19=PG19,arms=ARMS,pilot=PILOT,final=FINAL,eval_steps=EVAL_STEPS,
        selection='lowest mean validation3072/4096; top2 noncontrols + R32 continue; locked panels only after durable choices',
        optimizer='old Adam/moments/count retained; independent fresh Adam and128-update warmup per new cohort; shared stable global clip1; CE only',
        interpretation='single-seed exploratory screening, not replicated improvement; all pilot/replay costs charged',
        code={f:file_sha(ROOT/f) for f in files})))

def summarize(result):
    return dict(status=result.get('status'),seed=SEED,selected=result.get('selected'),scores=result.get('scores'),
        progress={a:dict(step=r.get('step',0),durable_step=r.get('durable_step'),failed=r.get('failed'),
            last_validation=(r.get('evaluations') or [{}])[-1].get('validation',{}).get('nll'),
            final=r.get('final')) for a,r in result.get('branches',{}).items()},
        pending=result.get('pending_slots',[]),cost=result.get('cost',{}),error=result.get('error'),
        report_sync_error=result.get('report_sync_error'),findings=result.get('findings'),
        interpretation='Exploratory seed123; validate the selected recipe on other seeds before promoting it')

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
        if self.result['contract']!=contract:raise ValueError('EXP115 resume contract mismatch')
        runtime={n:version(n) for n in ('jax','jaxlib','flax','optax')}
        runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
        if 'source_runtime' in contract:
            source=contract['source_runtime']
            required=dict(source['versions'],matmul_precision=source['matmul_precision'],prng_impl=source['prng_impl'])
            if runtime!=required:raise ValueError('source EXP111 optimizer/math runtime differs')
        if self.result.get('runtime',runtime)!=runtime:raise ValueError('EXP115 optimizer/math runtime changed')
        self.result['runtime']=runtime;self.store=RecoveryCheckpointStore(args.checkpoint_dir,f'{PREFIX}/checkpoints',hub)
        self.next_sync=0.;self.invocation=dict(max_wall_hours=args.max_wall_hours,
            save_reserve_minutes=args.save_reserve_minutes,uploaded_bytes=0)
        self.result['sessions'].append(self.invocation);self.result.pop('error',None)
    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('registered work deadline')
    def emit(self,event):
        self.result.setdefault('notifications',[]).append(dict(event=event,**_safe_notify(self.args.telegram,
            f'Extent EXP115: {event}; status={self.result.get("status")}; selected={self.result.get("selected")}; HF pending={len(self.result["pending_slots"])}')))
        print(f'EXP115 {event}',flush=True);self.persist(local_only=True)
    def checkpoint(self,arm,payload,step):
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
            self.hub,commit_message='EXP115 rank growth metrics')
        except Exception as e:
            status=getattr(getattr(e,'response',None),'status_code',None)
            if status in (401,403):raise
            self.result['report_sync_error']=dict(error_type=type(e).__name__,http_status=status)
            self.next_sync=time.monotonic()+(3600 if status==429 else 120);write();return False
        self.result.pop('report_sync_error',None);self.next_sync=time.monotonic()+600
        self.invocation['uploaded_bytes']+=size;write();return True

def load_patient(config,mesh,s):
    # Reconstruct only seed123. Reuse immutable source readers and Qwen loading primitives.
    seq=engine.configured_contract(config);order=tuple(seq['replacement_order'])
    tc=engine.teacher_config_from_spec(engine.QWEN3_1_7B_BASE,param_dtype='bfloat16',compute_dtype='bfloat16',remat_policy='full')
    teacher=engine.Qwen3ForCausalLM(tc);t=jax.device_put(np.zeros((1,1),np.int32),batch_sharding(mesh))
    initial=initialize_sharded_parameters(teacher,jax.random.key(940),t,mesh)
    tp,_=engine.stream_teacher_qwen_weights(initial.params,engine._ensure_checkpoint(Path(s.args.qwen_cache_dir)),tc)
    del initial
    source_pin=json.loads((ROOT/'results/EXP-111-113-source-pin.json').read_text())
    oldhub=replace(s.hub,repo_id=source_pin['repo_id'],repo_type='dataset',revision=source_pin['revision'])
    stores=(ReadOnlySourceStore(Path(s.args.state_dir)/'source069',f'{engine.EXP069_PREFIX}/checkpoints',oldhub),
        ReadOnlySourceStore(Path(s.args.state_dir)/'source072',f'{engine.EXP072["HF_PREFIX"]}/checkpoints',oldhub))
    endpoints={}
    for depth,layer in enumerate(order,1):
        s.check();prep=stores[0].metadata(f'prep/seed-{SEED}/layer-{layer}',dict(seq['source_contract'],seed=SEED,layer=layer,kind='prepared_mamba'))
        if prep is None:raise FileNotFoundError('missing pinned EXP069 source')
        cc=engine.sequential.endpoint_contract(seq,SEED,'ONPOLICY',layer,depth,prep['checkpoint_sha256'])
        saved=stores[1].restore(f'seed-{SEED}/ONPOLICY/layer-{layer}',cc)
        if saved is None or saved[1]['checkpoint_sha256']!=source_pin['seeds'][str(SEED)][str(layer)]:raise ValueError('EXP072 binary SHA mismatch')
        endpoints[layer]=saved[0]['params']
    model=HybridForCausalLM(config);abstract=abstract_parameter_tree(model)
    base=engine.compose_parameters(tp,endpoints,order,abstract,named_sharding_tree(abstract,mesh))
    pin=json.loads((ROOT/'results/EXP-114-source-pin.json').read_text())
    if tree_digest(jax.device_get(base))!=pin['source_models'][str(SEED)]['sha256']:raise ValueError('ONPOLICY base hash mismatch')
    c=initialize_corrections(base,order,'INOUT-LORA',seed=SEED,rank=32,head_dim=config.mamba.head_dim)
    pl=trainable_layout(c,mesh);c=jax.device_put(c,pl);tx=optimizer(c,peak=3e-4,adapter=True)
    initial=initialize_sharded_optimizer_state(tx,c,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),c),pl,mesh)
    readers=RecoveryCheckpointStore(Path(s.args.state_dir)/'patient',
        'experiments/exp111-recovery-efficiency/checkpoints',replace(s.hub,repo_id=pin['repo_id'],repo_type='dataset',revision=pin['revision']))
    cc=dict(pin['source_contract'],seed=SEED,arm='MAIN');slot=f'{SEED}/MAIN'
    if readers.metadata(slot,cc)!=pin['manifests'][str(SEED)]['metadata']:raise ValueError('EXP111 manifest pin differs')
    restored=readers.restore(slot,cc,dict(parameters=c,optimizer=initial.opt_state),dict(parameters=pl,optimizer=initial.layout))
    if restored is None or restored[1]['step']!=32768:raise FileNotFoundError('EXP111 patient32768 unavailable')
    old,state=restored[0]['parameters'],restored[0]['optimizer']
    if int(state[1].count)!=32768 or int(state[3].count)!=32768:raise ValueError('source Adam cursor mismatch')
    del endpoints,c,initial;gc.collect()
    return model,base,old,state,teacher,jax.device_get(tp)

def init_payload(old,source_state,arm,mesh):
    q=initialize(old,arm);txs=optimizers(arm);states={};pl={};sl={}
    for name in q:
        pl[name]=trainable_layout(q[name],mesh);q[name]=jax.device_put(q[name],pl[name])
        init=initialize_sharded_optimizer_state(txs[name],q[name],
            jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),q[name]),pl[name],mesh)
        states[name]=jax.device_put(source_state,init.layout) if name=='old' else init.opt_state;sl[name]=init.layout
    return dict(parameters=q,optimizer=states),dict(parameters=pl,optimizer=sl)

def reconcile_cursor(row,cursor):
    if row['step']!=cursor:
        row.setdefault('rollbacks',[]).append(dict(reported=row['step'],restored=cursor))
        row['evaluations']=[r for r in row['evaluations'] if r['step']<=cursor]
        row.pop('final',None);row['step']=cursor

def run_branch(s,arm,target,model,base,old,source_state,columns,mesh,tokens,kernels):
    payload,layouts=init_payload(old,source_state,arm,mesh)
    saved=s.store.restore(arm,dict(s.contract,arm=arm),payload,layouts)
    row=s.result['branches'].setdefault(arm,dict(step=0,evaluations=[],attempted_updates=0))
    cursor=0
    if saved:payload,meta=saved;cursor=meta['step']
    elif row.get('durable_step',0):raise FileNotFoundError('durable branch state unavailable')
    reconcile_cursor(row,cursor)
    q,state=payload['parameters'],payload['optimizer'];spec=ARMS[arm]
    expected_old=32768+(0 if spec['freeze_old'] else cursor)
    if int(state['old'][1].count)!=expected_old or int(state['old'][3].count)!=expected_old:raise ValueError('old Adam count mismatch')
    for i,(_,start,_) in enumerate(spec['cohorts']):
        expected=max(0,cursor-start)
        if int(state[f'new{i}'][1].count)!=expected or int(state[f'new{i}'][3].count)!=expected:raise ValueError('new cohort Adam count mismatch')
    row['allocated_coordinates']=coordinate_count(q)
    row['potentially_trainable_coordinates']=potential_coordinate_count(q,columns,spec['freeze_old'])
    row['checkpoint_bytes_estimate']=sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(payload))
    batch=batch_sharding(mesh)
    forward=kernels.setdefault(('forward',arm),GuardedJit(lambda p,f,n:materialize(f,p,columns,spec,n)))
    step=kernels.setdefault(('step',arm),GuardedJit(make_step(model,columns,arm)))
    def evaluation(n):
        s.check();started=time.monotonic();effective=forward(q,base,jnp.asarray(n,jnp.int32))
        value=evaluate(model,effective,tokens['validation'],batch)
        reference=s.result.get('pilot_replay_reference',{}).get(arm,{}).get(str(n))
        if reference is not None:
            error=float(np.max(np.abs(np.asarray(value['windows'])-reference['windows'])))
            if error>1e-4:raise ValueError('selected pilot replay differs; stop scientific comparison')
            row.setdefault('pilot_replay_parity',{})[str(n)]=error
        active_rank=32+sum(rank for rank,start,peak in spec['cohorts'] if n>=start)
        row['evaluations'].append(dict(step=n,input_tokens=n*256,active_rank=active_rank,validation=value))
        row['diagnostic_seconds']=row.get('diagnostic_seconds',0)+time.monotonic()-started
        print(f'EXP115 {arm} step={n} validation={value["nll"]:.6f}',flush=True)
    try:
        if cursor==0 and not row.get('initial_parity'):
            effective=forward(q,base,jnp.asarray(0,jnp.int32))
            canonical=materialize(base,{'old':old},columns,ARMS['R32'],0)
            parity=[]
            for name in old:
                path=name.split('/');a,b=effective,canonical
                for key in path:a,b=a[key],b[key]
                parity.append(bool(jnp.all(a==b)))
            if not all(parity):raise ValueError('rank expansion changed initial BF16 kernels')
            row['initial_parity']=dict(all_corrected_kernels_bitwise_equal=True)
        if cursor in EVAL_STEPS and not any(r['step']==cursor for r in row['evaluations']):evaluation(cursor)
        while cursor<target:
            s.check();w=jax.device_put(tokens['train'][cursor][None],batch);begun=time.monotonic()
            following,next_state,health=step(q,state,base,w,jnp.asarray(cursor,jnp.int32))
            row['attempted_updates']+=1
            s.result['cost']['attempted_updates']+=1;s.result['cost']['input_tokens']+=256;s.result['cost']['ce_targets']+=255
            row['training_seconds']=row.get('training_seconds',0)+time.monotonic()-begun
            row['last_health']=plain(health)
            if not bool(health['finite']):
                row['failed']=dict(step=cursor,reason='nonfinite proposal rejected');s.emit(f'{arm} rejected nonfinite update');break
            q=jax.device_put(following,layouts['parameters']);state=jax.device_put(next_state,layouts['optimizer'])
            cursor+=1;row['step']=cursor
            if cursor%512==0:s.checkpoint(arm,dict(parameters=q,optimizer=state),cursor)
            if cursor in EVAL_STEPS:evaluation(cursor);s.persist()
            elif cursor%256==0:s.persist()
    except MemoryError as e:
        row['failed']=dict(step=cursor,kind='engineering_unavailable',reason=str(e))
        s.emit(f'{arm} unavailable: compiler memory guard')
    finally:
        s.checkpoint(arm,dict(parameters=q,optimizer=state),cursor);s.persist(local_only=True)
        del payload,q,state;gc.collect()

def locked(s,arms,model,base,old,source_state,columns,mesh,tokens,teacher,tp):
    batch=batch_sharding(mesh)
    if not s.result.get('choices_durable'):raise ValueError('cannot view locked panels before durable validation selection')
    for panel in ('locked_wiki','locked_pg19'):
        if panel not in s.result.setdefault('baselines',{}):
            s.check();canonical=materialize(base,{'old':old},columns,ARMS['R32'],0)
            s.result['baselines'][panel]=dict(start=evaluate(model,canonical,tokens[panel],batch),
                qwen=evaluate(teacher,jax.device_put(tp,named_sharding_tree(tp,mesh)),tokens[panel],batch));s.persist()
        for arm in arms:
            row=s.result['branches'][arm]
            if row.get('failed') or row['step']!=FINAL:continue
            if panel in row.get('final',{}):continue
            s.check();template,layouts=init_payload(old,source_state,arm,mesh)
            restored=s.store.restore(arm,dict(s.contract,arm=arm),template,layouts)
            if restored is None or restored[1]['step']!=FINAL:raise FileNotFoundError('final candidate checkpoint unavailable')
            q=restored[0]['parameters'];effective=materialize(base,q,columns,ARMS[arm],FINAL)
            row.setdefault('final',{})[panel]=evaluate(model,effective,tokens[panel],batch)
            s.persist();del restored,template,q,effective;gc.collect()
    control=s.result['branches']['R32'].get('final',{})
    s.result['findings']={arm:{panel:row['final'][panel]['nll']-control[panel]['nll']
        for panel in row.get('final',{}) if panel in control} for arm,row in s.result['branches'].items() if arm!='R32'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='/kaggle/working/output')
    p.add_argument('--state-dir',default='/dev/shm/extent-exp115-source')
    p.add_argument('--checkpoint-dir',default='/dev/shm/extent-exp115-checkpoints')
    p.add_argument('--dataset-cache-dir',default='/kaggle/working/extent-dataset-cache')
    p.add_argument('--qwen-cache-dir',default='/kaggle/working/qwen3-exp104-weights')
    p.add_argument('--max-wall-hours',type=float,default=8.8)
    p.add_argument('--save-reserve-minutes',type=float,default=60)
    p.add_argument('--max-new-hf-gb',type=float,default=6)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--sync-only',action='store_true')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args(sys.argv[1:] if argv is None else argv)
    session_budget(args.max_wall_hours,args.save_reserve_minutes)
    if not np.isfinite(args.max_new_hf_gb) or args.max_new_hf_gb<=0:raise ValueError('positive finite upload cap required')
    config,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml');contract=contract_for(config)
    if args.plan_only:print(json.dumps(contract,indent=2));return contract
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    s=Session(args,contract,hub)
    if args.sync_only or s.result.get('status')=='completed':s.persist(force=True);return s.result
    s.result['status']='preflight';s.emit('started')
    try:
        if not s.persist(force=True):raise RuntimeError('HF report write preflight failed')
        tokens,meta=load_tokens(args.dataset_cache_dir)
        if meta['panels']!=contract['data']:raise ValueError('EXP115 data SHA mismatch')
        s.result['data_provenance']=meta;s.check()
        s.result['roundtrip']=cloud_roundtrip(Path(args.checkpoint_dir),PREFIX,hub,s.deadline)
        devices=list(jax.devices())
        if jax.process_count()!=1 or len(devices)!=8 or any(d.platform!='tpu' for d in devices):raise ValueError('single-host TPU v5e8 required')
        s.result['accelerator']=dict(devices=len(devices),processes=jax.process_count(),
            device_kind=str(devices[0].device_kind))
        import psutil
        if psutil.virtual_memory().available<80*(1<<30):raise MemoryError('source reconstruction needs80GiB host RAM')
        mesh=create_v5e_mesh(devices);model,base,old,state,teacher,tp=load_patient(config,mesh,s)
        before=state_digest(jax.device_get(dict(parameters=old,optimizer=state)))
        s.result['source_initial_hash']=before;s.result['status']='running'
        columns=2*int(config.hidden_size*config.mamba.expand)+2*config.mamba.mimo_rank*config.mamba.groups*config.mamba.d_state
        kernels={}
        # Historical checkpoint validation must replay exactly, before candidate training.
        original=materialize(base,{'old':old},columns,ARMS['R32'],0)
        # Validation is not in reference_final: compare against the recorded111 A score in the pinned contract fixture.
        ref=json.loads((ROOT/'results/EXP-115-source-validation.json').read_text())
        found=evaluate(model,original,tokens['validation'],batch_sharding(mesh))
        error=float(np.max(np.abs(np.asarray(found['windows'])-ref['windows'])))
        if error>1e-4:raise ValueError('EXP111 validation endpoint replay differs')
        s.result['source_validation_parity']=error;del original
        if 'selected' not in s.result:
            for arm in ARMS:
                if s.result['branches'].get(arm,{}).get('failed'):continue
                record=s.result['branches'].get(arm,{})
                if record.get('step',0)<PILOT or not any(r['step']==PILOT for r in record.get('evaluations',[])):
                    run_branch(s,arm,PILOT,model,base,old,state,columns,mesh,tokens,kernels)
                s.emit(f'{arm} pilot finished')
            chosen,scores=select(s.result['branches']);s.result['selected']=chosen;s.result['scores']=scores
            s.result['pilot_replay_reference']={arm:{str(r['step']):r['validation'] for r in s.result['branches'][arm]['evaluations']
                if r['step'] in (3072,4096)} for arm in ['R32']+chosen}
        if not s.result.get('choices_durable'):
            # Persist selection before any fresh locked data evaluation; no model-state copies required here.
            published=s.persist(force=True,checkpoints=False)
            if not published:
                while time.monotonic()<s.deadline:
                    if s.next_sync>=s.deadline:raise TimeoutError('selection report upload cooldown')
                    time.sleep(min(60,max(1,s.next_sync-time.monotonic())))
                    published=s.persist(force=True,checkpoints=False)
                    if published:break
            if not published:raise TimeoutError('selection was not durably published')
            s.result['choices_durable']=True;s.persist(local_only=True)
        finalists=['R32']+s.result['selected']
        # A report may outlive its unsynced model. Never skip training based only
        # on a claimed final cursor after the previous cloud runtime disappeared.
        for arm in finalists:
            metadata=s.store.metadata(arm,dict(s.contract,arm=arm))
            actual=metadata['step'] if metadata is not None else 0
            row=s.result['branches'][arm]
            if metadata is None and row.get('durable_step',0):raise FileNotFoundError('durable finalist manifest missing')
            reconcile_cursor(row,actual)
        for target in (8192,12288,16384):
            for arm in finalists:
                if s.result['branches'][arm].get('failed'):continue
                if s.result['branches'][arm]['step']<target:
                    run_branch(s,arm,target,model,base,old,state,columns,mesh,tokens,kernels)
                actual=s.result['branches'][arm]['step']
                s.emit(f'{arm} step={actual}; target={target}; failed={bool(s.result["branches"][arm].get("failed"))}')
        locked(s,finalists,model,base,old,state,columns,mesh,tokens,teacher,tp)
        after=state_digest(jax.device_get(dict(parameters=old,optimizer=state)))
        if after!=before:raise ValueError('immutable source mutated')
        base_hash=tree_digest(jax.device_get(base))
        expected_base=json.loads((ROOT/'results/EXP-114-source-pin.json').read_text())['source_models'][str(SEED)]['sha256']
        if base_hash!=expected_base:raise ValueError('immutable ONPOLICY base mutated')
        s.result['source_final_base_hash']=base_hash
        s.result['source_final_hash']=after;s.result['status']='completed';s.emit('training and evaluation finished')
    except TimeoutError as e:s.result.update(status='deadline_partial',error=str(e));s.emit('work deadline')
    except Exception as e:
        s.result.update(status='failed',error=str(e),error_type=type(e).__name__,traceback=traceback.format_exc());s.emit('failed');raise
    finally:
        s.persist(force=True)
        while (s.result['pending_slots'] or s.result.get('report_sync_error')) and time.monotonic()<s.wall-60:
            deadlines=[]
            if s.result['pending_slots']:deadlines.append(s.store.next_sync)
            if s.result.get('report_sync_error'):deadlines.append(s.next_sync)
            wait=min(deadlines) if deadlines else s.wall
            if wait>=s.wall-60:break
            time.sleep(min(60,max(1,wait-time.monotonic())));s.persist(force=True)
        s.emit('finished and saved' if not s.result['pending_slots'] and not s.result.get('report_sync_error') else 'finished; HF pending')
    return s.result

if __name__=='__main__':main()
