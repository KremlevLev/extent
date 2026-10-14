"""EXP114: resumable frozen-model plateau examination on Kaggle TPU v5e8."""
import argparse,gc,hashlib,json,sys,time,traceback
from dataclasses import replace
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from extent import HybridForCausalLM
from extent.config import load_config
from extent.campaign_checkpoint import write_json_atomic
from extent.hf_artifact_sync import artifact_config_from_env,restore_artifact,upload_artifacts_together
from extent.recovery_durability import RecoveryCheckpointStore,cloud_roundtrip
from extent.efficiency_sources import load_sources,tree_digest
from extent.efficiency_recovery import optimizer,DATA as OLD_DATA
from extent.calibration_data import load_wikitext2_tokens,load_pg19_tokens
from extent.qwen_source import QWEN3_1_7B_BASE
from extent.efficiency_diagnostics import component_arrays,update_metrics,relative,stats
from extent.decoder_recovery import forward_parameters
from extent.sharding import create_v5e_mesh,named_sharding_tree,batch_sharding,replicated_sharding
from extent.initialization import initialize_sharded_optimizer_state
from extent.plateau_data import load_tokens,PANELS,PG19,hash_array
from extent.plateau_forensics import (GuardedJit,ForensicLM,plain,losses,token_metrics,norm,
    layer_profile,mamba_trace,readout_init,readout_logits,readout_optimizer,readout_step,
    select_layers,paired_summary,state_digest,compress_trace)
from extent.stable_gradient_clip import stable_clip_by_global_norm,scaled_norm_parts
from extent.full_model_distillation import causal_cross_entropy
from scripts.m3q_efficiency_campaign import contract_for as old_contract
from scripts.m3q_paper_decoder_recovery_campaign import evaluate,trainable_layout
from scripts.m3q_batch_recovery_campaign import session_budget
from scripts.qwen_extended_horizon_campaign import _safe_notify

ROOT=Path(__file__).resolve().parents[1]
PREFIX='experiments/exp114-plateau-forensics'
STEM='extent-m3q-plateau-114'
SEEDS=(123,456)
FIT_STEPS=4096
ENDPOINTS=(1024,4096)
DISCOVERY_WINDOWS=16
TOLERANCE=1e-4
BF16_LOWERING_LIMIT=1e-2

def file_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def contract_for(config):
    pin=json.loads((ROOT/'results/EXP-114-source-pin.json').read_text())
    if old_contract(config,111)!=pin['source_contract']:
        raise ValueError('EXP111 dependencies/contract changed; refuse patient reconstruction')
    data=json.loads((ROOT/'results/EXP-114-data-preflight.json').read_text())
    files=('scripts/m3q_plateau_forensics_114_campaign.py','extent/plateau_forensics.py',
        'extent/plateau_data.py','extent/chunked_checkpoint.py','extent/recovery_durability.py',
        'extent/hf_artifact_sync.py','extent/efficiency_sources.py','extent/efficiency_diagnostics.py',
        'scripts/m3q_plateau_114_report.py','extent/plateau_trace_repair.py')
    return dict(experiment=114,source_sha=file_sha(ROOT/'results/EXP-114-source-pin.json'),
        data_sha=file_sha(ROOT/'results/EXP-114-data-preflight.json'),
        source_revision=pin['revision'],source_contract=pin['source_contract'],
        data_manifest=data['panels'],seeds=list(SEEDS),fit_steps=FIT_STEPS,
        readout_endpoints=list(ENDPOINTS),discovery_windows=DISCOVERY_WINDOWS,
        tolerance=TOLERANCE,bf16_lowering_limit=BF16_LOWERING_LIMIT,code={n:file_sha(ROOT/n) for n in files})

def summary(result):
    return dict(experiment=114,status=result.get('status'),gate=None,
        completed_tasks=len(result.get('tasks',{})),modules=result.get('modules',{}),
        selections=result.get('choices',{}),readout_progress=result.get('readout',{}),
        findings=result.get('findings',{}),pending=result.get('pending_slots',[]),
        report_sync_error=result.get('report_sync_error'),
        cost=result.get('cost',{}),error=result.get('error'))

class Session:
    """Atomic per-window result cursor and small probe optimizer checkpoints."""
    def __init__(self,args,contract,hub):
        self.args,self.contract,self.hub=args,contract,hub
        self.begun=time.monotonic();training,reserve=session_budget(args.max_wall_hours,args.save_reserve_minutes)
        self.deadline=self.begun+training;self.wall=self.begun+args.max_wall_hours*3600
        self.output=Path(args.output_dir);self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/f'{STEM}.json';self.small=self.output/f'{STEM}-summary.txt'
        if not self.path.exists():restore_artifact(self.path,f'{PREFIX}/latest.json',hub)
        self.result=json.loads(self.path.read_text()) if self.path.exists() else dict(contract=contract,
            tasks={},modules={},choices={},readout={},sessions=[],pending_slots=[],cost={})
        if self.result['contract']!=contract:
            from extent.plateau_trace_repair import migrate_trace_failure
            migrate_trace_failure(self.result,contract,getattr(args,'repair_trace',False))
        runtime={n:version(n) for n in ('jax','jaxlib','flax','optax')}
        runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
        if self.result.get('runtime',runtime)!=runtime:raise ValueError('EXP114 optimizer/math runtime changed')
        self.result['runtime']=runtime
        self.store=RecoveryCheckpointStore(args.checkpoint_dir,f'{PREFIX}/checkpoints',hub)
        self.kernels={}
        self.next_sync=0.
        self.invocation=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
            max_wall_hours=args.max_wall_hours,save_reserve_minutes=reserve,uploaded_bytes=0)
        self.result['sessions'].append(self.invocation);self.result.pop('error',None)
    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('registered EXP114 work deadline')
    def task(self,key,call):
        if key in self.result['tasks']:return self.result['tasks'][key]['value']
        self.check();begun=time.monotonic()
        try:value=call()
        finally:
            cost=self.result['cost'];cost['diagnostic_seconds']=cost.get('diagnostic_seconds',0)+(time.monotonic()-begun)
            cost['attempted_tasks']=cost.get('attempted_tasks',0)+1
        self.result['tasks'][key]=dict(value=value,seconds=time.monotonic()-begun)
        self.persist();return value
    def emit(self,event):
        self.result.setdefault('notifications',[]).append(dict(event=event,**_safe_notify(self.args.telegram,
            f"Extent EXP-114: {event}; status={self.result.get('status')}; tasks={len(self.result['tasks'])}; HF pending={len(self.result['pending_slots'])}")))
        print(f'EXP114: {event}',flush=True);self.persist(local_only=True)
    def module(self,name,call):
        if self.result['modules'].get(name)=='completed':return
        self.result['modules'][name]='running';self.emit(f'{name} started')
        started=time.monotonic()
        try:
            call();self.result['modules'][name]='completed';self.emit(f'{name} completed')
        finally:
            times=self.result['cost'].setdefault('module_wall_seconds',{})
            times[name]=times.get(name,0)+time.monotonic()-started
    def checkpoint(self,slot,payload,step):
        self.store.save(slot,dict(self.result.get('readout_checkpoint_contract',self.contract),slot=slot),payload,step)
        if slot not in self.result['pending_slots']:self.result['pending_slots'].append(slot)
    def sync_states(self):
        for slot in list(self.result['pending_slots']):
            directory=self.store.directory(slot)
            if not (directory/'checkpoint.json').exists():
                self.result['pending_slots'].remove(slot);continue
            meta=json.loads((directory/'checkpoint.json').read_text())
            receipt=directory/f"{meta['generation']}.uploaded.json"
            before=set(json.loads(receipt.read_text())['files']) if receipt.exists() else set()
            required=sum(c['bytes'] for r in meta['leaves'] for c in r['chunks'] if c['file'] not in before)
            if required+self.invocation['uploaded_bytes']>self.args.max_new_hf_gb*1e9:
                self.result['storage_error']=dict(required_bytes=required,uploaded_bytes=self.invocation['uploaded_bytes']);return False
            success=self.store.sync(slot,deadline=self.wall-60)
            after=set(json.loads(receipt.read_text())['files']) if receipt.exists() else set()
            self.invocation['uploaded_bytes']+=sum(c['bytes'] for r in meta['leaves'] for c in r['chunks'] if c['file'] in after-before)
            if success:
                self.result['pending_slots'].remove(slot)
                seed,kind=slot.split('/')
                row=self.result.get('readout',{}).get(seed)
                if row is not None:
                    if kind=='READOUT':row['durable_step']=meta['step']
                    else:row.setdefault('durable_endpoints',{})[str(meta['step'])]=True
        return not self.result['pending_slots']
    def persist(self,local_only=False,force=False):
        self.invocation['hours']=(time.monotonic()-self.begun)/3600
        write_json_atomic(self.path,self.result);self.small.write_text(json.dumps(summary(self.result),indent=2),encoding='utf-8')
        if local_only or (time.monotonic()<self.next_sync and
            (not force or self.result.get('report_sync_error'))):return False
        self.sync_states()
        # Re-write after the durable checkpoint receipt so metrics never claim an unsynced fit state.
        write_json_atomic(self.path,self.result);self.small.write_text(json.dumps(summary(self.result),indent=2),encoding='utf-8')
        assets=[self.output/name for name in self.result.get('csv_files',[]) if (self.output/name).exists()]
        size=self.path.stat().st_size+self.small.stat().st_size+sum(x.stat().st_size for x in assets)
        if size+self.invocation['uploaded_bytes']>self.args.max_new_hf_gb*1e9:
            self.result['report_sync_error']='invocation upload cap';return False
        try:upload_artifacts_together([(self.path,f'{PREFIX}/latest.json'),(self.small,f'{PREFIX}/latest-summary.txt')]+
            [(x,f'{PREFIX}/reports/{x.name}') for x in assets],self.hub,commit_message='EXP114 diagnostic report')
        except Exception as e:
            code=getattr(getattr(e,'response',None),'status_code',None)
            self.result['report_sync_error']=dict(error_type=type(e).__name__,http_status=code)
            self.next_sync=time.monotonic()+(3600 if code==429 else 120)
            if code in (401,403):raise
            return False
        self.result.pop('report_sync_error',None);self.invocation['uploaded_bytes']+=size
        self.next_sync=time.monotonic()+600;return True
    def lock_choices(self):
        # A-only choices are durably published BEFORE any new C result.
        if not self.persist(force=True):raise RuntimeError('cannot publish choices before locked evaluation')

class Patient:
    def __init__(self,config,model,teacher,teacher_host,tc,base,c,state,mesh,session,seed,tokens):
        self.cfg,self.model,self.teacher,self.tc=config,model,teacher,tc
        self.mesh,self.s,self.seed,self.tokens=mesh,session,seed,tokens
        self.batch=batch_sharding(mesh);self.rep=replicated_sharding(mesh)
        self.base=jax.device_put(base,named_sharding_tree(base,mesh))
        self.c,self.state=c,state
        self.columns=2*int(config.hidden_size*config.mamba.expand)+2*config.mamba.mimo_rank*config.mamba.groups*config.mamba.d_state
        self.current=jax.device_put(forward_parameters('ADAPTER-CE',c,self.base,self.columns,config.mamba.head_dim),named_sharding_tree(self.base,mesh))
        self.q=jax.device_put(teacher_host,named_sharding_tree(teacher_host,mesh))
        self.capture=self.kernel('capture',lambda:GuardedJit(lambda p,t:model.apply({'params':p},t,return_hidden_states=True)))
        self.qcapture=self.kernel('qcapture',lambda:GuardedJit(lambda p,t:teacher.apply({'params':p},t,return_hidden_states=True)))
        self.metrics=self.kernel('metrics',lambda:GuardedJit(token_metrics));self.hooks={};self.teacher_cache={}
        self.features=self.kernel('features',lambda:GuardedJit(lambda p,t:norm(model.apply({'params':p},t,return_hidden_states=True)[1][-1],p['norm'],config)))
        self.vocabulary={n:self.current[n] for n in ('embed_tokens','lm_head') if n in self.current}
        self.head_eval=self.kernel('head-eval',lambda:GuardedJit(lambda h,c,v,t,temp:losses(readout_logits(h,c,v,config)/temp,t)))
    def kernel(self,name,factory):
        if name not in self.s.kernels:self.s.kernels[name]=factory()
        return self.s.kernels[name]
    def t(self,w):return jax.device_put(w[None],self.batch)
    def canonical(self,panel,parameters=None):
        p=self.current if parameters is None else parameters
        key=f'baseline/{self.seed}/{panel}' if parameters is None else f'qwen/{self.seed}/{panel}'
        module=self.model if parameters is None else self.teacher
        return self.evaluate_windows(key,self.tokens[panel],lambda w:evaluate(module,p,w[None],self.batch)['nll'])
    def evaluate_windows(self,key,windows,call):
        return [self.s.task(f'{key}/{i}',lambda w=w:float(call(w))) for i,w in enumerate(windows)]
    def teacher_states(self,w):
        key=hash_array(w)
        if key not in self.teacher_cache:
            _,h=self.qcapture(self.q,self.t(w))
            # Host cache bounded to discovery panel; C is streamed.
            found=tuple(jax.device_get(x) for x in h)
            if len(self.teacher_cache)<DISCOVERY_WINDOWS:self.teacher_cache[key]=found
            return found
        return self.teacher_cache[key]
    def hook_loss(self,mode,indices,w,period=0,tail=None):
        kind=(mode,period,tail)
        if kind not in self.hooks:
            module=ForensicLM(self.cfg,self.tc,mode,period)
            self.hooks[kind]=self.kernel(('hook',kind),lambda:GuardedJit(lambda p,r,m,h,t:losses(module.apply({'params':p},t,r,m,h),t,tail)))
        mask=np.zeros(self.cfg.num_layers,bool);mask[list(indices)]=True
        reference=self.q if mode=='teacher' else self.base
        boundaries=self.teacher_states(w) if mode=='boundary' else tuple(jnp.zeros((1,len(w),self.cfg.hidden_size),jnp.bfloat16) for _ in range(self.cfg.num_layers))
        return float(self.hooks[kind](self.current,reference,jax.device_put(mask,self.rep),boundaries,self.t(w)))
    def intervention(self,panel,mode,indices,period=0,neighbor=False,tail=None):
        windows=self.tokens[panel][:DISCOVERY_WINDOWS] if panel=='A' else self.tokens[panel]
        name=','.join(map(str,indices));key=f'intervention/{self.seed}/{panel}/{mode}/{name}/reset{period}/tail{tail}'
        values=self.evaluate_windows(key,windows,lambda w:self.hook_loss(mode,indices,w,period,tail))
        if tail is None:canonical=self.canonical(panel)[:len(windows)]
        else:
            base=GuardedJit(lambda p,t:losses(self.model.apply({'params':p},t),t,tail))
            canonical=self.evaluate_windows(f'tail-baseline/{self.seed}/{panel}/{tail}',windows,lambda w:base(self.current,self.t(w)))
        # Same executable identity controls isolate interventions from BF16 lowering.
        baseline_indices=indices if mode=='reset' else []
        baseline_period=0 if mode=='reset' else period
        current=self.evaluate_windows(f'hook-baseline/{self.seed}/{panel}/{mode}/{baseline_indices}/reset{baseline_period}/tail{tail}',windows,
            lambda w:self.hook_loss(mode,baseline_indices,w,baseline_period,tail))
        floor=max(abs(a-b) for a,b in zip(current,canonical))
        if floor>BF16_LOWERING_LIMIT:raise ValueError(f'BF16 hook background exceeds registered limit: {floor}')
        report=paired_summary(current,values);report.update(family=mode,layer=indices[0] if len(indices)==1 else None,
            indices=list(indices),period=period,tail=tail,neighbor=neighbor,panel=panel,seed=self.seed,
            canonical_nll=float(np.mean(canonical)),canonical_identity_max_error=floor,
            canonical_identity_window_errors=[a-b for a,b in zip(current,canonical)])
        self.s.result.setdefault('tables',{}).setdefault('interventions',{})[f'{key}/neighbor{neighbor}']=report
        self.s.persist(local_only=True)
        return report
    def profile(self):
        for panel in ('A','B','T_seen','T_new'):
            self.canonical(panel);self.canonical(panel,self.q)
            self.evaluate_windows(f'own-start/{self.seed}/{panel}',self.tokens[panel],lambda w:
                evaluate(self.model,self.base,w[None],self.batch)['nll'])
        calls={}
        frequencies=np.bincount(self.tokens['FIT'].ravel(),minlength=self.cfg.vocab_size)
        for i,w in enumerate(self.tokens['A'][:DISCOVERY_WINDOWS]):
            self.s.check();t=self.t(w);logits,h=self.capture(self.current,t);ql,qh=self.qcapture(self.q,t)
            canonical=self.canonical('A')[i]
            error=abs(float(losses(logits,t))-canonical)
            if error>BF16_LOWERING_LIMIT:raise ValueError(f'capture lowering parity failed: {error}')
            self.s.task(f'profile/{self.seed}/tokens/{i}',lambda:dict(
                student=plain(self.metrics(logits,t)),teacher=plain(self.metrics(ql,t)),
                fit_token_frequency=frequencies[w[1:]].tolist(),canonical_capture_error=error,
                positions={f'{a}-{min(b,len(w)-1)-1}':float(jnp.mean(token_metrics(logits,t)['nll'][:,a:b]))
                    for a,b in ((0,64),(64,128),(128,192),(192,255)) if a<len(w)-1}))
            hs=(self.current['embed_tokens']['embedding'][t],)+h
            qs=(self.q['embed_tokens']['embedding'][t],)+qh
            for j in range(self.cfg.num_layers):
                kind=j in self.cfg.attention_layer_indices
                if kind not in calls:
                    cfg,tc=self.cfg,self.tc
                    calls[kind]=self.kernel(('layer-profile',kind),lambda j=j:GuardedJit(lambda p,q,x,y:layer_profile(p,q,x,y,cfg,j,tc)))
                self.s.task(f'profile/{self.seed}/layer{j}/{i}',lambda j=j:plain(calls[j in self.cfg.attention_layer_indices](
                    self.current[f'layers_{j}'],self.q[f'layers_{j}'],hs[j],qs[j])))
            self.s.task(f'profile/{self.seed}/final_norm/{i}',lambda:plain(dict(
                student=stats(norm(h[-1],self.current['norm'],self.cfg)),
                teacher=stats(norm(qh[-1],self.q['norm'],self.cfg)),
                error=relative(norm(h[-1],self.current['norm'],self.cfg),norm(qh[-1],self.q['norm'],self.cfg)))))
            del logits,ql,h,qh,hs,qs;gc.collect()
    def sweep(self):
        choices=self.s.result['choices']
        if str(self.seed) not in choices:
            rows=[]
            for family,layers in (('rollback',range(self.cfg.num_layers)),('teacher',self.cfg.mamba_layer_indices),('boundary',range(self.cfg.num_layers))):
                identity=self.intervention('A',family,[])
                if abs(identity['delta_nll'])>TOLERANCE or max(abs(x) for x in identity['per_window_delta'])>TOLERANCE:
                    raise ValueError(f'{family} identity lowering exceeds tolerance')
                for j in layers:
                    r=self.intervention('A',family,[j]);rows.append(r)
                    if family=='rollback' and j in self.cfg.attention_layer_indices and max(abs(x) for x in r['per_window_delta'])>BF16_LOWERING_LIMIT:
                        raise ValueError('frozen-block rollback is not null')
            choices[str(self.seed)]=select_layers(rows);self.s.persist(local_only=True)
        selection=choices[str(self.seed)]
        self.intervention('A','teacher',selection['pair'])
        for family,j in selection['single'].items():
            self.intervention('B',family,[j])
            neighbor=j+1 if j+1<self.cfg.num_layers else j-1
            # Teacher replacement at retained attention is an explicit null neighbor.
            self.intervention('B',family,[neighbor],neighbor=True)
        self.intervention('B','teacher',selection['pair'])
        for j in selection['pair']:self.intervention('B','teacher',[j])
        for family in ('rollback','teacher','boundary'):self.intervention('B',family,[])
    def dynamics(self):
        cfg=self.cfg
        trace=self.kernel('mamba-trace',lambda:GuardedJit(lambda p,x:mamba_trace(x,p,cfg)))
        original=self.kernel('mamba-original',lambda:GuardedJit(lambda p,x:Mamba3MIMO(cfg.hidden_size,cfg.mamba,
            jnp.bfloat16,jnp.bfloat16).apply({'params':p},x)))
        # Import here to leave plan/source-preflight model-free.
        for i,w in enumerate(self.tokens['A'][:DISCOVERY_WINDOWS]):
            _,h=self.capture(self.current,self.t(w));inputs=(self.current['embed_tokens']['embedding'][self.t(w)],)+h
            for j in self.cfg.mamba_layer_indices:
                x=norm(inputs[j],self.current[f'layers_{j}']['input_layernorm'],self.cfg)
                def run(j=j,x=x):
                    output,report=trace(self.current[f'layers_{j}']['mamba'],x)
                    control=original(self.current[f'layers_{j}']['mamba'],x)
                    error=float(jnp.max(jnp.abs(output-control)))
                    if error>TOLERANCE:raise ValueError(f'traced scan output parity: {error}')
                    _,start_report=trace(self.base[f'layers_{j}']['mamba'],x)
                    return dict(current=plain(compress_trace(report)),own_start_same_input=plain(compress_trace(start_report)),output_identity_max=error)
                self.s.task(f'dynamics/{self.seed}/layer{j}/{i}',run)
            del h,inputs;gc.collect()
        selected=self.s.result['choices'][str(self.seed)]['pair']
        for panel in ('A','B'):
            for j in selected:
                # reset period0 executes the traced path as an identity control.
                control=self.intervention(panel,'reset',[j],period=0,tail=32)
                if max(abs(v) for v in control['per_window_delta'])>TOLERANCE:raise ValueError('reset normal-carry identity failed')
                for period in (32,128):self.intervention(panel,'reset',[j],period=period,tail=32)
        for i,w in enumerate(self.tokens['A'][:DISCOVERY_WINDOWS]):
            for length in (32,64,128,256):
                self.s.task(f'context/{self.seed}/{i}/{length}',lambda w=w,length=length:dict(
                    student=float(losses(self.model.apply({'params':self.current},self.t(w[-length:])),self.t(w[-length:]),31)),
                    teacher=float(losses(self.teacher.apply({'params':self.q},self.t(w[-length:])),self.t(w[-length:]),31)),
                    input_sha=hash_array(w[-length:]),targets_sha=hash_array(w[-31:])))
    def fit(self):
        s=self.s;seed=str(self.seed);slot=f'{seed}/READOUT'
        c=readout_init(self.cfg.hidden_size,114000+self.seed);pl=trainable_layout(c,self.mesh);c=jax.device_put(c,pl)
        tx=readout_optimizer();template_state=initialize_sharded_optimizer_state(tx,c,
            jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),c),pl,self.mesh)
        payload=dict(parameters=c,optimizer=template_state.opt_state)
        layouts=dict(parameters=pl,optimizer=template_state.layout)
        saved=s.store.restore(slot,dict(s.result.get('readout_checkpoint_contract',s.contract),slot=slot),payload,layouts)
        row=s.result['readout'].setdefault(seed,dict(step=0,attempted_updates=0,curve={}))
        if saved:
            payload,meta=saved;c,state=payload['parameters'],payload['optimizer'];step=meta['step']
        else:
            if row.get('durable_step',0):raise FileNotFoundError('readout durable state missing')
            state=template_state.opt_state;step=0
        if row['step']!=step:
            row.setdefault('rollbacks',[]).append(dict(reported=row['step'],restored=step))
            row['curve']={k:v for k,v in row['curve'].items() if int(k)<=step};row['step']=step
        if int(state[1].count)!=step or int(state[2].count)!=step:raise ValueError('readout Adam/schedule cursor mismatch')
        call=GuardedJit(readout_step(self.cfg));zero=readout_init(self.cfg.hidden_size,114000+self.seed)
        for w in self.tokens['A'][:2]:
            h=self.features(self.current,self.t(w));actual=float(self.head_eval(h,zero,self.vocabulary,self.t(w),jnp.array(1.)))
            canonical=evaluate(self.model,self.current,w[None],self.batch)['nll']
            if abs(actual-canonical)>BF16_LOWERING_LIMIT:raise ValueError('readout identity parity failed')
        def record_endpoint():
            if step in ENDPOINTS and str(step) not in row['curve']:
                endpoint=f'{seed}/READOUT-{step}'
                s.checkpoint(endpoint,dict(parameters=c,optimizer=state),step)
                values=self.head_panel('A',c,f'fit-{step}')
                row['curve'][str(step)]=dict(nll=float(np.mean(values)),windows=values)
                print(f'EXP114 readout {seed} step={step} A NLL={np.mean(values):.5f}',flush=True);s.persist()
        try:
            record_endpoint()
            while step<FIT_STEPS:
                s.check();w=self.tokens['FIT'][step];t=self.t(w)
                feature_begun=time.monotonic();h=self.features(self.current,t);begun=time.monotonic()
                row['feature_seconds']=row.get('feature_seconds',0)+begun-feature_begun
                c2,state2,health=call(c,state,h,self.vocabulary,t)
                row['attempted_updates']+=1;s.result['cost']['fit_input_tokens']=s.result['cost'].get('fit_input_tokens',0)+len(w)
                s.result['cost']['fit_target_tokens']=s.result['cost'].get('fit_target_tokens',0)+len(w)-1
                row['training_seconds']=row.get('training_seconds',0)+time.monotonic()-begun
                if not bool(health['finite']):raise FloatingPointError('readout proposal nonfinite; rejected')
                # Keep the exact same layouts before/after checkpoint restore.
                c=jax.device_put(c2,pl);state=jax.device_put(state2,template_state.layout)
                step+=1;row['step']=step;row['last_health']=plain(health)
                if step%256==0:
                    s.checkpoint(slot,dict(parameters=c,optimizer=state),step);s.persist()
                record_endpoint()
        finally:
            s.checkpoint(slot,dict(parameters=c,optimizer=state),step);s.persist(local_only=True)
    def head_panel(self,panel,c,label,temperature=1.):
        return self.evaluate_windows(f'readout/{self.seed}/{panel}/{label}',self.tokens[panel],lambda w:
            self.head_eval(self.features(self.current,self.t(w)),c,self.vocabulary,self.t(w),jnp.array(temperature)))
    def endpoint(self,step):
        c=readout_init(self.cfg.hidden_size,114000+self.seed);pl=trainable_layout(c,self.mesh);c=jax.device_put(c,pl)
        tx=readout_optimizer();initial=initialize_sharded_optimizer_state(tx,c,
            jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),c),pl,self.mesh)
        slot=f'{self.seed}/READOUT-{step}'
        restored=self.s.store.restore(slot,dict(self.s.result.get('readout_checkpoint_contract',self.s.contract),slot=slot),dict(parameters=c,optimizer=initial.opt_state),
            dict(parameters=pl,optimizer=initial.layout))
        if restored is None or restored[1]['step']!=step:raise FileNotFoundError('selected readout endpoint unavailable')
        return restored[0]['parameters']
    def temperatures(self):
        zero=readout_init(self.cfg.hidden_size,114000+self.seed)
        for temperature in (.5,1.,2.,4.,8.):self.head_panel('A',zero,f'T-{temperature}',temperature)
    def optimizer_probe(self):
        s=self.s
        def objective(c,base,t):
            p=forward_parameters('ADAPTER-CE',c,base,self.columns,self.cfg.mamba.head_dim)
            return causal_cross_entropy(self.model.apply({'params':p},t),t)
        call=GuardedJit(jax.value_and_grad(objective,argnums=0))
        gradients={};individual={}
        for panel in ('T_new','A'):
            total=None;rows=[]
            for w in self.tokens[panel][:8]:
                s.check();loss,g=call(self.c,self.base,self.t(w))
                s.result['cost']['coordinate_probe_target_tokens']=s.result['cost'].get('coordinate_probe_target_tokens',0)+len(w)-1
                rows.append(dict(loss=float(loss),groups=plain(jax.tree.map(stats,g))))
                total=g if total is None else jax.tree.map(lambda a,b:a+b,total,g)
            gradients[panel]=jax.tree.map(lambda x:x/8,total);individual[panel]=rows
        g=gradients['T_new'];tx=optimizer(self.c,peak=3e-4,adapter=True)
        update,_=tx.update(g,self.state,self.c)
        clipped,_=stable_clip_by_global_norm(1).update(g,optax.EmptyState())
        _,_,lu=scaled_norm_parts(update);_,_,lg=scaled_norm_parts(clipped)
        gradient_update=jax.tree.map(lambda x:-x*jnp.power(10.,lu-lg),clipped)
        def tree_cosine(a,b):
            ma,_,_=scaled_norm_parts(a);mb,_,_=scaled_norm_parts(b)
            aa=jax.tree.map(lambda x:x/jnp.maximum(ma,1e-30),a);bb=jax.tree.map(lambda x:x/jnp.maximum(mb,1e-30),b)
            return sum(jnp.sum(x*y) for x,y in zip(jax.tree.leaves(aa),jax.tree.leaves(bb)))/jnp.maximum(
                jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(aa))*sum(jnp.sum(x*x) for x in jax.tree.leaves(bb))),1e-30)
        s.task(f'optimizer/{self.seed}/gradients',lambda:dict(individual=individual,
            mean_coordinate_train=plain(jax.tree.map(stats,g)),mean_coordinate_validation=plain(jax.tree.map(stats,gradients['A'])),
            train_validation_cosine=float(tree_cosine(g,gradients['A'])),group_cosine=plain(jax.tree.map(
                lambda a,b:relative(a,b)['cosine'],g,gradients['A'])),adam_moments=plain(jax.tree.map(stats,{'mu':self.state[1].mu,'nu':self.state[1].nu})),
            clip_retained_fraction=float(jnp.minimum(1.,jnp.power(10.,-scaled_norm_parts(g)[2])))))
        effective=GuardedJit(jax.value_and_grad(lambda p,t:causal_cross_entropy(self.model.apply(
            {'params':jax.tree.map(lambda x:x.astype(jnp.bfloat16),p)},t),t)))
        cfg,model,columns=self.cfg,self.model,self.columns
        def projection(c,base,tangent):
            _,pullback=jax.vjp(lambda q:forward_parameters('ADAPTER-CE',q,base,columns,cfg.mamba.head_dim),c)
            return pullback(jax.tree.map(lambda x:x.astype(jnp.bfloat16),tangent))[0]
        project=GuardedJit(projection)
        fp=jax.tree.map(lambda x:x.astype(jnp.float32),self.current)
        for panel in ('T_new','A'):
            total=None;individual_effective=[]
            for w in self.tokens[panel][:8]:
                s.check();loss,eg=effective(fp,self.t(w));arrays=component_arrays(eg,self.cfg)
                s.result['cost']['effective_probe_target_tokens']=s.result['cost'].get('effective_probe_target_tokens',0)+len(w)-1
                individual_effective.append(dict(loss=float(loss),components=plain({k:stats(v) for k,v in arrays.items()})))
                total=eg if total is None else jax.tree.map(lambda a,b:a+b,total,eg)
            mean=jax.tree.map(lambda x:x/8,total)
            projected=project(self.c,self.base,mean)
            s.task(f'optimizer/{self.seed}/effective/{panel}',lambda mean=mean,individual_effective=individual_effective:dict(
                mean_components=plain({k:stats(v) for k,v in component_arrays(mean,self.cfg).items()}),individual=individual_effective,
                projection_to_coordinates=plain(jax.tree.map(lambda a,b:relative(a,b),projected,gradients[panel])),
                note='effective derivatives include frozen groups; not available optimizer directions'))
            del total,mean,eg,projected;gc.collect()
        del fp;gc.collect()
        for name,direction in (('adam',update),('gradient',gradient_update)):
            for alpha in (0.,.1,1.):
                proposed=jax.tree.map(lambda p,u:p+alpha*u,self.c,direction)
                materialized=forward_parameters('ADAPTER-CE',proposed,self.base,self.columns,self.cfg.mamba.head_dim)
                s.task(f'optimizer/{self.seed}/{name}/{alpha}/motion',lambda:dict(
                    actual_update=plain(update_metrics(self.c,proposed,self.current,materialized,self.cfg)),
                    prediction_train=float(sum(jnp.sum(a*b*alpha) for a,b in zip(jax.tree.leaves(g),jax.tree.leaves(direction)))),
                    prediction_validation=float(sum(jnp.sum(a*b*alpha) for a,b in zip(jax.tree.leaves(gradients['A']),jax.tree.leaves(direction))))))
                for panel in ('T_new','A','B'):
                    windows=self.tokens[panel][:8] if panel=='T_new' else self.tokens[panel]
                    self.evaluate_windows(f'shadow/{self.seed}/{panel}/{name}/{alpha}',windows,lambda w:
                        evaluate(self.model,materialized,w[None],self.batch)['nll'])
                del materialized,proposed;gc.collect()
    def locked(self,modules):
        choices=self.s.result['choices'];selection=choices[str(self.seed)]
        for panel in ('C_wiki','C_pg19'):
            self.canonical(panel);self.canonical(panel,self.q)
            if modules.get('sweep')=='completed':
                for family,j in selection['single'].items():
                    self.intervention(panel,family,[j]);self.intervention(panel,family,[])
                self.intervention(panel,'teacher',selection['pair'])
                for j in selection['pair']:self.intervention(panel,'teacher',[j])
            if modules.get('readout')=='completed':
                zero=readout_init(self.cfg.hidden_size,114000+self.seed)
                self.head_panel(panel,zero,'identity')
                temperature=choices['temperature'];self.head_panel(panel,zero,f'T-{temperature}',temperature)
                c=self.endpoint(choices['readout_endpoint']);self.head_panel(panel,c,'selected')
            if modules.get('dynamics')=='completed':
                for j in selection['pair']:
                    for period in (0,32,128):self.intervention(panel,'reset',[j],period=period,tail=32)
            if modules.get('optimizer')=='completed':
                # Recompute deterministic proposals from frozen source Adam; do not search on C.
                self.locked_shadow(panel)
    def locked_shadow(self,panel):
        call=GuardedJit(jax.grad(lambda c,base,t:causal_cross_entropy(self.model.apply({'params':
            forward_parameters('ADAPTER-CE',c,base,self.columns,self.cfg.mamba.head_dim)},t),t),argnums=0))
        total=None
        for w in self.tokens['T_new'][:8]:
            self.s.check();g=call(self.c,self.base,self.t(w));total=g if total is None else jax.tree.map(lambda a,b:a+b,total,g)
            self.s.result['cost']['coordinate_probe_target_tokens']=self.s.result['cost'].get('coordinate_probe_target_tokens',0)+len(w)-1
        g=jax.tree.map(lambda x:x/8,total);u,_=optimizer(self.c,peak=3e-4,adapter=True).update(g,self.state,self.c)
        clip,_=stable_clip_by_global_norm(1).update(g,optax.EmptyState())
        factor=jnp.power(10.,scaled_norm_parts(u)[2]-scaled_norm_parts(clip)[2])
        for name,d in (('adam',u),('gradient',jax.tree.map(lambda x:-x*factor,clip))):
            for alpha in (0.,.1,1.):
                p=forward_parameters('ADAPTER-CE',jax.tree.map(lambda a,b:a+alpha*b,self.c,d),self.base,self.columns,self.cfg.mamba.head_dim)
                self.evaluate_windows(f'shadow/{self.seed}/{panel}/{name}/{alpha}',self.tokens[panel],lambda w:evaluate(self.model,p,w[None],self.batch)['nll'])
    def precision(self):
        # Memory failure is unavailable; other exceptions remain real failures.
        cfg=replace(self.cfg,param_dtype='float32',compute_dtype='float32');model=HybridForCausalLM(cfg)
        values=None
        try:
            values=jax.tree.map(lambda x:x.astype(jnp.float32),self.current)
            call=GuardedJit(lambda p,t:losses(model.apply({'params':p},t),t))
            with jax.default_matmul_precision('highest'):
                rows=[self.s.task(f'precision/{self.seed}/{i}',lambda w=w:float(call(values,self.t(w)))) for i,w in enumerate(self.tokens['A'][:2])]
            self.s.result.setdefault('precision',{})[str(self.seed)]=dict(fp32_windows=rows,
                canonical_bf16=self.canonical('A')[:2],scope='two discovery windows only')
        except Exception as e:
            if isinstance(e,MemoryError) or 'RESOURCE_EXHAUSTED' in str(e) or 'out of memory' in str(e).lower():
                self.s.result.setdefault('precision',{})[str(self.seed)]=dict(unavailable='memory',error_type=type(e).__name__)
            else:raise
        finally:del values;gc.collect()

def choose_readout(result):
    scores={step:float(np.mean([result['readout'][str(seed)]['curve'][str(step)]['nll'] for seed in SEEDS])) for step in ENDPOINTS}
    endpoint=min(scores,key=lambda step:(scores[step],step))
    temps={}
    for temperature in (.5,1.,2.,4.,8.):
        values=[]
        for seed in SEEDS:
            prefix=f'readout/{seed}/A/T-{temperature}/'
            found=[row['value'] for key,row in result['tasks'].items() if key.startswith(prefix)]
            if len(found)!=result['contract']['data_manifest']['A']['shape'][0]:
                raise ValueError('incomplete temperature evaluation at a seed')
            values.extend(found)
        if not values or not np.all(np.isfinite(values)):raise ValueError('incomplete temperature evaluation')
        temps[temperature]=float(np.mean(values))
    result['choices']['readout_endpoint']=endpoint
    result['choices']['temperature']=min(temps,key=lambda x:(temps[x],x))
    result['choices']['readout_A_scores']=scores;result['choices']['temperature_A_scores']=temps

def analyze_report(result):
    """Paired tables and conditional signals; missing endpoints never become zeros."""
    tasks=result.get('tasks',{});choices=result.get('choices',{});findings={}
    def series(prefix):
        rows={int(k.removeprefix(prefix)):r['value'] for k,r in tasks.items() if k.startswith(prefix)}
        return [rows[i] for i in range(len(rows))] if set(rows)==set(range(len(rows))) else []
    readout={}
    for seed in SEEDS:
        readout[str(seed)]={}
        for panel in ('B','C_wiki','C_pg19'):
            canonical=series(f'baseline/{seed}/{panel}/');found=series(f'readout/{seed}/{panel}/selected/')
            baseline=series(f'readout/{seed}/{panel}/identity/')
            expected=result['contract']['data_manifest'][panel]['shape'][0]
            if len(baseline)==len(found)==expected:
                readout[str(seed)][panel]=paired_summary(baseline,found)
                floor=max(abs(a-b) for a,b in zip(canonical,baseline)) if len(canonical)==expected else None
                readout[str(seed)][panel]['canonical_identity_max_error']=floor
                if floor is None or floor>BF16_LOWERING_LIMIT:
                    raise ValueError('readout held-out identity parity failed')
    findings['readout_alignment']=readout
    signals={}
    table=list(result.get('tables',{}).get('interventions',{}).values())
    for family in ('rollback','teacher','boundary','readout'):
        deltas={};valid=True
        for seed in SEEDS:
            deltas[str(seed)]={}
            for panel in ('B','C_wiki','C_pg19'):
                if family=='readout':row=readout[str(seed)].get(panel)
                else:
                    layer=choices.get(str(seed),{}).get('single',{}).get(family)
                    row=next((r for r in table if r['seed']==seed and r['panel']==panel and r['family']==family
                        and r['indices']==[layer] and not r['neighbor'] and r['tail'] is None),None)
                if row is None:valid=False
                else:deltas[str(seed)][panel]=row['delta_nll']
        if not valid:label='INCOMPLETE'
        elif all(v<=-.1 for r in deltas.values() for v in r.values()):label='PRIORITY_SIGNAL'
        elif any(v<=-.1 for r in deltas.values() for v in r.values()):label='CONDITIONAL_SIGNAL'
        else:label='NO_REGISTERED_SIGNAL'
        signals[family]=dict(status=label,deltas=deltas)
    findings['intervention_signals']=signals
    interactions=[]
    for seed in SEEDS:
        pair=choices.get(str(seed),{}).get('pair',[])
        if len(pair)!=2:continue
        for panel in ('A','B','C_wiki','C_pg19'):
            def lookup(indices):
                return next((r for r in table if r['seed']==seed and r['panel']==panel and r['family']=='teacher'
                    and r['indices']==indices and r['tail'] is None),None)
            both=lookup(pair);first=lookup([pair[0]]);second=lookup([pair[1]])
            if all(r is not None for r in (both,first,second)):
                interactions.append(dict(seed=seed,panel=panel,layers=pair,
                    interaction=both['delta_nll']-first['delta_nll']-second['delta_nll']))
    findings['teacher_pair_interactions']=interactions
    profile={}
    for seed in SEEDS:
        rows=[r['value'] for k,r in tasks.items() if k.startswith(f'profile/{seed}/tokens/')]
        if not rows:continue
        student=np.concatenate([np.asarray(r['student']['nll']).ravel() for r in rows])
        teacher=np.concatenate([np.asarray(r['teacher']['nll']).ravel() for r in rows])
        excess=student-teacher;positive=np.maximum(excess,0);count=max(1,int(np.ceil(.1*len(excess))))
        frequency=np.concatenate([np.asarray(r['fit_token_frequency']) for r in rows])
        def sliced(mask):return dict(count=int(mask.sum()),student_nll=float(student[mask].mean()),
            teacher_nll=float(teacher[mask].mean())) if mask.any() else dict(count=0)
        profile[str(seed)]=dict(student_nll=float(student.mean()),teacher_nll=float(teacher.mean()),
            signed_excess=float(excess.sum()),positive_excess=float(positive.sum()),
            worst10_positive_excess_share=float(np.sort(positive)[-count:].sum()/max(positive.sum(),1e-30)),
            per_window_nll_quantiles=np.quantile([np.mean(r['student']['nll']) for r in rows],[.5,.9,1]).tolist(),
            teacher_difficulty={'below2':sliced(teacher<2),'2to5':sliced((teacher>=2)&(teacher<5)),'above5':sliced(teacher>=5)},
            fit_frequency={'0or1':sliced(frequency<=1),'2to100':sliced((frequency>1)&(frequency<=100)),'above100':sliced(frequency>100)})
    findings['error_profile']=profile
    result['findings']=findings;return findings

def legacy_panel_tokens(cache):
    kw=dict(tokenizer_repo=QWEN3_1_7B_BASE.repo_id,tokenizer_revision=QWEN3_1_7B_BASE.revision)
    rows={name:load_wikitext2_tokens(n*256,cache,token_offset=offset,dataset_split=split,
        dataset_config='wikitext-103-raw-v1',**kw).reshape(n,256)
        for name,split,offset,n in OLD_DATA if name in ('reproduction','locked_test')}
    rows['pg19_test']=load_pg19_tokens(128*256,cache,token_offset=393216,dataset_split='test',**kw).reshape(128,256)
    return rows

def patient_states(config,model,sources,mesh,s,pin,teacher,teacher_host,tc,tokens):
    readers=RecoveryCheckpointStore(Path(s.args.state_dir)/'patients',
        'experiments/exp111-recovery-efficiency/checkpoints',replace(s.hub,
        repo_id=pin['repo_id'],revision=pin['revision'],repo_type='dataset'))
    rows={}
    for seed in SEEDS:
        s.check();base,c=sources[seed];pl=trainable_layout(c,mesh);c=jax.device_put(c,pl)
        tx=optimizer(c,peak=3e-4,adapter=True)
        init=initialize_sharded_optimizer_state(tx,c,jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype),c),pl,mesh)
        cc=dict(pin['source_contract'],seed=seed,arm='MAIN');slot=f'{seed}/MAIN'
        metadata=readers.metadata(slot,cc)
        if metadata!=pin['manifests'][str(seed)]['metadata']:raise ValueError('patient manifest differs from immutable pin')
        restored=readers.restore(slot,cc,dict(parameters=c,optimizer=init.opt_state),dict(parameters=pl,optimizer=init.layout))
        if restored is None or restored[1]['step']!=32768:raise FileNotFoundError('durable patient32768 unavailable')
        payload,meta=restored
        if int(payload['optimizer'][1].count)!=32768 or int(payload['optimizer'][3].count)!=32768:
            raise ValueError('patient Adam/schedule cursor mismatch')
        rows[seed]=(payload['parameters'],payload['optimizer'])
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='/kaggle/working/output')
    p.add_argument('--state-dir',default='/dev/shm/extent-exp114-source')
    p.add_argument('--checkpoint-dir',default='/dev/shm/extent-exp114-checkpoints')
    p.add_argument('--dataset-cache-dir',default='/kaggle/working/extent-dataset-cache')
    p.add_argument('--qwen-cache-dir',default='/kaggle/working/qwen3-exp104-weights')
    p.add_argument('--max-wall-hours',type=float,default=8.8)
    p.add_argument('--save-reserve-minutes',type=float,default=60)
    p.add_argument('--max-new-hf-gb',type=float,default=1)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--repair-trace',action='store_true',help='Explicit audited migration of the original pre-dynamics trace failure; retain completed A/B and readout states')
    p.add_argument('--sync-only',action='store_true')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args(sys.argv[1:] if argv is None else argv)
    if not np.isfinite(args.max_new_hf_gb) or args.max_new_hf_gb<=0:raise ValueError('positive finite upload cap required')
    session_budget(args.max_wall_hours,args.save_reserve_minutes)
    config,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml');contract=contract_for(config)
    if args.plan_only:
        plan=dict(contract=contract,max_wall_hours=args.max_wall_hours,save_reserve_minutes=args.save_reserve_minutes,
            max_new_hf_gb=args.max_new_hf_gb,patient_weights_unchanged=True)
        print(json.dumps(plan,indent=2));return plan
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF checkpoint secrets required')
    s=Session(args,contract,hub)
    if args.sync_only or s.result.get('status')=='completed':
        s.persist(force=True);s.emit('sync complete' if not s.result['pending_slots'] and not s.result.get('report_sync_error') else 'HF pending');return s.result
    s.result['status']='preflight';s.emit('started')
    patients={}
    try:
        if not s.persist(force=True):raise RuntimeError('HF report write preflight failed before models')
        tokens,manifest=load_tokens(args.dataset_cache_dir)
        if manifest['panels']!=contract['data_manifest']:raise ValueError('EXP114 data SHA/capacity mismatch')
        s.result['data_provenance']=manifest
        s.result['roundtrip']=cloud_roundtrip(Path(args.checkpoint_dir),PREFIX,hub,s.deadline)
        devices=list(jax.devices())
        if jax.process_count()!=1 or len(devices)!=8 or any(d.platform!='tpu' for d in devices):raise ValueError('single-host TPU v5e8 required')
        mesh=create_v5e_mesh(devices)
        import psutil
        if psutil.virtual_memory().available<80*(1<<30):raise MemoryError('EXP114 source host caches require >=80GiB available RAM')
        pin=json.loads((ROOT/'results/EXP-114-source-pin.json').read_text())
        # Immutable legacy sources, without rerunning111 or changing source contracts.
        legacy={"original_qwen":True}
        model,order,sources,(teacher,teacher_host,tc)=load_sources(config,mesh,args,hub,{},legacy,
            lambda:None,s.deadline,experiment=111)
        if legacy['source_models']!=pin['source_models']:raise ValueError('reconstructed ONPOLICY source tree hashes differ')
        states=patient_states(config,model,sources,mesh,s,pin,teacher,teacher_host,tc,tokens)
        before={str(seed):state_digest(jax.device_get({'parameters':c,'optimizer':state})) for seed,(c,state) in states.items()}
        s.result['patient_initial_hashes']=before;s.result['status']='running'
        def with_seed(seed,call):
            c,state=states[seed]
            patient=Patient(config,model,teacher,teacher_host,tc,sources[seed][0],c,state,mesh,s,seed,tokens)
            try:return call(patient)
            finally:del patient;gc.collect()
        # Canonical replay BEFORE diagnostics and new-data selection.
        previous=legacy_panel_tokens(args.dataset_cache_dir)
        for seed in SEEDS:
            def parity(patient,seed=seed):
                for panel in ('reproduction','locked_test','pg19_test'):
                    found=evaluate(model,patient.current,previous[panel],patient.batch)
                    expected=pin['reference_final'][str(seed)][panel]
                    if abs(found['nll']-expected['nll'])>TOLERANCE or np.max(np.abs(np.asarray(found['windows'])-expected['windows']))>TOLERANCE:
                        raise ValueError('patient canonical endpoint replay differs')
                    s.result.setdefault('patient_parity',{}).setdefault(str(seed),{})[panel]=dict(mean_error=abs(found['nll']-expected['nll']),
                        max_window_error=float(np.max(np.abs(np.asarray(found['windows'])-expected['windows']))))
            with_seed(seed,parity)
        del previous;gc.collect()
        for name,method in (('profile','profile'),('sweep','sweep'),('fit','fit'),('temperature','temperatures')):
            s.module(name,lambda method=method:[with_seed(seed,lambda patient:getattr(patient,method)()) for seed in SEEDS])
        if not s.result['choices'].get('readout_endpoint'):choose_readout(s.result)
        def readout_verification(patient):
            zero=readout_init(config.hidden_size,114000+patient.seed)
            patient.head_panel('B',zero,'identity')
            temp=s.result['choices']['temperature'];patient.head_panel('B',zero,f'T-{temp}',temp)
            patient.head_panel('B',patient.endpoint(s.result['choices']['readout_endpoint']),'selected')
        s.module('readout',lambda:[with_seed(seed,readout_verification) for seed in SEEDS])
        for name,method in (('optimizer','optimizer_probe'),('dynamics','dynamics'),('precision','precision')):
            s.module(name,lambda method=method:[with_seed(seed,lambda patient:getattr(patient,method)()) for seed in SEEDS])
        s.lock_choices()
        s.module('locked',lambda:[with_seed(seed,lambda patient:patient.locked(s.result['modules'])) for seed in SEEDS])
        after={str(seed):state_digest(jax.device_get({'parameters':c,'optimizer':state})) for seed,(c,state) in states.items()}
        if after!=before:raise ValueError('patient parameters/Adam mutated by diagnostics')
        s.result['patient_final_hashes']=after;s.result['patient_unchanged']=True
        analyze_report(s.result)
        s.result['status']='completed'
    except TimeoutError as e:s.result.update(status='deadline_partial',error=str(e))
    except Exception as e:
        s.result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        s.emit('failed');s.persist(force=True);raise
    finally:
        from scripts.m3q_plateau_114_report import export_tables
        s.result['csv_files']=[x.name for x in export_tables(s.result,s.output)]
        s.persist(force=True)
        while (s.result['pending_slots'] or s.result.get('report_sync_error')) and time.monotonic()<s.wall-60:
            delays=[]
            if s.result['pending_slots']:delays.append(s.store.next_sync)
            if s.result.get('report_sync_error'):delays.append(s.next_sync)
            delay=min(delays) if delays else s.wall
            if delay>=s.wall-60:break
            time.sleep(min(60,max(1,delay-time.monotonic())))
            s.persist(force=True)
        s.emit('finished and saved' if not s.result['pending_slots'] and not s.result.get('report_sync_error') else 'finished; HF pending')
        s.persist(local_only=True)
    return s.result

# Local tests explicitly exercise the same math on tiny models; main never accepts CPU training.
from extent.layers.mamba3 import Mamba3MIMO
if __name__=='__main__':main()
