"""EXP163: expand trainable scope from the SAME EXP161 C4 CURRENT."""
import json
from pathlib import Path
import time
from importlib.metadata import version
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze
from extent.batch_recovery_161 import ROOT,sha,parent_replay,source_pin,restore_source,base_coordinates,adam_counts,Inspector,replay_report,teacher_validation
from extent.decoder_recovery_157 import prepare as old_prepare,dense_coordinates,block_forward as source_block
from extent.corpus_recovery_160 import recipes as old_recipes
from extent.campaign_checkpoint import write_json_atomic
from extent.recovery_durability import RecoveryCheckpointStore
from extent.stable_gradient_clip import stable_clip_by_global_norm

PREFIX='experiments/exp163-method-sweep-v1'
GPU_EXPERIMENTS=(163,)
TPU_EXPERIMENTS=()
LIMITS={163:8760.}
SOURCE_STEP=154852
SOURCE_ARM='KD-C4-FRESH'


def recipes(experiment):
    if experiment!=163:raise ValueError('EXP163 only')
    s=old_recipes(160)[1]
    return [dict(s,name=name,data_stream='edu',scope=scope,decoder_dense=False,
            mode='fresh-adam',mamba_lr=1e-5 if scope=='full' else 3e-4,
            mlp_lr=1e-5,norm_lr=1e-6 if scope=='full' else 1e-4,lr_warmup=512)
            for name,scope in [('R256-CONTROL','rank'),('UPPER8-MLP','upper'),('FULL-KD','full')]]


def freeze_parent(work,hub=None,deadline=None):
    from extent.corpus_switch_162 import freeze_parent as original
    from extent.hf_artifact_sync import upload_artifact
    from extent.hf_download_retry import DownloadPolicy
    root=Path(work);path=root/'methods/source.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():value=json.loads(path.read_text())
    else:
        _,old=original(work,hub,deadline)
        source=old['branches']['C4-CONTINUE']
        value=dict(old,format='methods-parent-v1',branches={s['name']:dict(source,parent_windows=0) for s in recipes(163)})
        write_json_atomic(path,value)
    if value.get('format')!='methods-parent-v1' or any(s['step']!=SOURCE_STEP for s in value['branches'].values()):
        raise ValueError('wrong method source plan')
    if hub is not None:
        until=min(time.time()+600,deadline or time.time()+600)
        DownloadPolicy(root/'methods/transport',deadline=until).call(upload_artifact,path,PREFIX+'/source.json',hub,
            commit_message='EXP163 immutable source descriptor')
    return path,value


def contract(experiment,worker,max_steps,corpus_dir,execution,source_plan,edu_dir):
    from scripts.m3q_corpus_recovery_160 import contract_for
    from extent.broad_corpus import PackedCorpus
    from scripts.prepare_edu_corpus import REVISION
    if experiment!=163 or worker is not None or execution!='server-balanced':raise ValueError('single GPU EXP163 only')
    parent=json.loads(Path(source_plan).read_text());edu=PackedCorpus(edu_dir)
    c=contract_for(160,None,max_steps,corpus_dir,'server-balanced');ec=edu.manifest['contract']
    if (parent.get('format')!='methods-parent-v1' or parent['corpus_identity']!=c['corpus_identity']
        or ec.get('dataset')!='HuggingFaceFW/fineweb-edu' or ec.get('revision')!=REVISION
        or ec.get('excluded_c4_identity')!=c['corpus_identity'] or ec.get('sequence_length')!=256
        or ec.get('tokenizer_revision')!=c['corpus_manifest']['contract']['tokenizer_revision']):
        raise ValueError('source/corpus identity differs')
    horizon=min(int(max_steps),len(edu.arrays['train'])//1536)
    if horizon<1:raise ValueError('Edu stream empty')
    c.update(experiment=163,protocol='exp163-method-sweep-v1',execution_namespace=PREFIX,batch=6,batch_size=6,
        parent=parent,parent_plan_sha=sha(source_plan),edu_identity=edu.identity,edu_contract=ec,recipes=recipes(163),
        horizon=horizon,first_goal_new_steps=min(8192,horizon),global_eval_steps=[],
        eval_steps=sorted(v for v in set([0,128,512,1024,2048]+list(range(4096,horizon+1,4096))+[horizon]) if v<=horizon),
        preparation='same verified C4 CURRENT154852, fresh Adam ALL arms; initial dense replay mandatory',
        optimizer='fresh Adam(.9,.999), warm512, stable globalclip1; scope-specific explicit LR; no inherited moments',
        clock='global154852+accepted updates; fresh Adam count=accepted updates; each Edu data cursor=6*accepted updates',
        selection='shared C4 D diagnostic; never donor-equivalent quality claim; CURRENT always retained',
        comparison='practical recipes: rank256 control vs rank256+upper8 fullMLP vs full decoder KD; LR and parameterization differ',
        continuation='all arms same packed Edu prefix and .5CE+.5KL; no rank growth, new reset, hidden loss or data cycling',
        evaluation='same A/B/C/D/E, EN/RU16 progress and128 final; all inspected panels diagnostic validation',
        gradient='each actual compiled batch6 proposal checked vs6 fixed-q microgradients and ONE Adam step',
        gpu='one active optimizer state on GPU, inactive states offloaded losslessly to host RAM; same batch6 via audited fallback',
        notifications='launcher start/finish only; console heartbeat5min',
        checkpoint='local pruned generation per arm every30min plus exit; HF metrics during work, full states at exit only;32GB cap',
        budget_policy='16 accepted pilot updates per arm per invocation; steady medians exclude first audit; common adaptive target uses remaining work time; target extends if time remains')
    for name in ('extent/method_sweep_163.py','extent/method_execution_163.py','scripts/m3q_method_sweep_163.py','engine_methods.py'):
        c['code'][name]=sha(ROOT/name)
    return c


def coordinates(base,current,cfg,columns,spec):
    if spec['scope']=='full':return dense_coordinates(base,current,cfg,columns,spec)
    q=jax.tree.map(lambda x:x,current)
    if spec['scope']=='upper':
        for i in range(max(0,cfg.num_layers-8),cfg.num_layers):
            q[str(i)]['dense_mlp']=jax.tree.map(lambda v:jnp.asarray(v,jnp.float32),base[f'layers_{i}']['mlp'])
    return q


def prepare(patient,spec,payload,pin,ancestor=None,*,parent,parent_root):
    # Immutable155 only reconstructs frozen base and the exact source tree shapes.
    original=old_recipes(160)[1]
    base,q,_=old_prepare(patient,dict(original,decoder_dense=False,mode='exact'),payload,pin,ancestor)
    from extent.competitor_recovery import optimizer as old_optimizer
    old_state=old_optimizer(q,original).init(q)
    template=dict(parameters=q,optimizer=old_state,best=q,best_step=jnp.asarray(118304,jnp.int32))
    source=parent['branches'][spec['name']];pointer=Path(parent_root)/SOURCE_ARM/'checkpoint.json'
    if sha(pointer)!=source['manifest_sha']:raise ValueError('C4 CURRENT pointer changed')
    saved=RecoveryCheckpointStore(parent_root,'read-only',None).restore(SOURCE_ARM,source['metadata']['contract'],template)
    if saved is None or saved[1]['step']!=SOURCE_STEP:raise ValueError('C4 CURRENT not available')
    if adam_counts(saved[0]['optimizer'])!=[source['adam_count']]*6:raise ValueError('parent Adam descriptor differs')
    # Parent moments deliberately NOT transplanted into new coordinates.
    current=jax.tree.map(jnp.asarray,saved[0]['parameters'])
    return base,coordinates(base,current,patient.cfg,patient.columns,spec),SOURCE_STEP


def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    if 'dense_mlp' in q:
        p=unfreeze(p)
        p['mlp']=jax.tree.map(lambda v,old:v.astype(old.dtype),q['dense_mlp'],p['mlp'])
        q={k:v for k,v in q.items() if k!='dense_mlp'}
    return source_block(cfg,index,p,q,h,columns,spec,strength,details=details)


def optimizer(q,spec):
    def label(path,value):
        names=[getattr(k,'key',None) for k in path]
        if any(k in names for k in ('pregate_weight','input_layernorm','post_attention_layernorm')):return 'norm'
        return 'mlp' if any(k in names for k in ('dense_mlp','mlp','mlp_lora')) else 'mamba'
    labels=jax.tree.map_with_path(label,q)
    def tx(rate):
        return optax.chain(optax.scale_by_adam(b1=.9,b2=spec['beta2'],eps=1e-8),
            optax.scale_by_schedule(lambda n:-rate*jnp.minimum((n+1)/spec['lr_warmup'],1.)))
    return optax.chain(stable_clip_by_global_norm(1.),optax.multi_transform(
        {k:tx(spec[k+'_lr']) for k in ('mamba','norm','mlp')},labels))


from extent.hidden_distillation_159 import DistillationStep
from extent.batch_execution_161 import BatchStep
from extent.reset_streamed_gpu import MemoryCheckedJit


class MethodReference(DistillationStep):
    def kernels(self,index):
        key=(index in self.cfg.attention_layer_indices,index>=max(0,self.cfg.num_layers-8))
        if key not in self.forward:
            def fn(p,q,h,phase):return block_forward(self.cfg,index,p,q,h,self.columns,self.spec,phase)[0]
            self.forward[key]=MemoryCheckedJit(fn)
            def backward(p,q,h,dy,phase):
                _,pull=jax.vjp(lambda q,h:fn(p,q,h,phase),q,h);return pull(dy)
            self.backward[key]=MemoryCheckedJit(backward)
        return self.forward[key],self.backward[key]


class MethodStep(BatchStep):
    def __init__(self,cfg,tc,columns,spec,tx):
        from extent.method_execution_163 import compiled_step
        self.reference=MethodReference(cfg,tc,columns,spec,tx,return_gradients=True)
        self.compiled=MemoryCheckedJit(compiled_step(cfg,tc,columns,spec,tx))
        def finite(g,loss,ce,kl):
            gradient=jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(g)]))
            objective=jnp.all(jnp.isfinite(jnp.stack([loss,ce,kl])))
            return gradient,objective
        self.gradient_finite=MemoryCheckedJit(finite)
        self.check=lambda:None;self.selected=None

    def accumulated(self,q,state,base,tp,tokens,cursor):
        # Dense Adam/proposal trees are huge. Do NOT compute six discarded Adam
        # proposals just to obtain six gradients. Update once after averaging.
        real_update=self.reference.update
        def gradient_only(q,state,g,loss,ce,kl):
            gf,of=self.gradient_finite(g,loss,ce,kl)
            return q,state,dict(finite=gf&of,loss=loss,ce=ce,kl=kl,
                stages=dict(gradients=dict(finite=gf),objective=dict(finite=of)))
        self.reference.update=gradient_only
        gradient=None;loss=ce=kl=0.;n=tokens.shape[0]
        try:
            for i in range(n):
                self.check()
                value=self.reference(q,state,base,tp,tokens[i:i+1],cursor)
                if not bool(value[2]['finite']):return value
                g=jax.tree.map(lambda x:x.astype(jnp.float32)/n,value[3])
                gradient=g if gradient is None else jax.tree.map(lambda a,b:a+b,gradient,g)
                loss+=float(value[2]['loss'])/n;ce+=float(value[2]['ce'])/n;kl+=float(value[2]['kl'])/n
                del value,g
        finally:self.reference.update=real_update
        result=real_update(q,state,gradient,jnp.asarray(loss,jnp.float32),jnp.asarray(ce,jnp.float32),jnp.asarray(kl,jnp.float32))
        return result+(gradient,)


def make_kernel(cfg,tc,columns,spec,tx,*,tpu=False):
    if tpu:raise ValueError('Blackwell single GPU only')
    return MethodStep(cfg,tc,columns,spec,tx)


def windows_seen(c,name,step):
    delta=int(step)-c['parent']['branches'][name]['step']
    if delta<0:raise ValueError('cursor before source')
    return delta*6


def training_windows(wiki,corpora,c,name,step):
    first=windows_seen(c,name,step)
    return np.concatenate([corpora['edu'].batch(first+i) for i in range(6)],axis=0)


def matched_target(rows,horizon,seconds_available):
    """Largest common endpoint fitting measured steady training cost; no rollback."""
    deltas={n:r['step']-r['source_step'] for n,r in rows.items()}
    low=max(deltas.values());high=horizon
    rates={n:float(np.median(r['timing_samples'][-6:])) for n,r in rows.items()}
    if any(not np.isfinite(v) or v<=0 for v in rates.values()):raise ValueError('invalid measured timing')
    budget=max(0,seconds_available)*.85
    while low<high:
        middle=(low+high+1)//2
        cost=sum(max(0,middle-deltas[n])*rates[n] for n in rows)
        if cost<=budget:low=middle
        else:high=middle-1
    return low


def to_host(payload):return jax.tree.map(lambda v:np.asarray(jax.device_get(v)),payload)


class ForkStore(RecoveryCheckpointStore):
    def __init__(self,*args,on_bootstrap=lambda name:None,**kwargs):super().__init__(*args,**kwargs);self.on_bootstrap=on_bootstrap
    def restore(self,name,c,template,layouts=None):
        own=super().restore(name,c,template,layouts)
        if own is not None:
            marker=self.directory(name)/'synced.json'
            if not marker.exists() or json.loads(marker.read_text()).get('generation')!=own[1]['generation']:self.on_bootstrap(name)
            return own
        if adam_counts(template['optimizer'])!=[0]*6:raise ValueError('all EXP163 arms require fresh Adam')
        meta=self.save(name,c,template,SOURCE_STEP);self.on_bootstrap(name)
        return template,meta


def make_session(args,c,hub):
    import jax
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from extent.hf_artifact_sync import restore_artifact
    from extent.server_control import install_session_control
    from scripts.m3q_corpus_recovery_160 import session_budget
    s=Transport.__new__(Transport);s.args=args;s.contract=c;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    s.prefix=PREFIX;s.stem='runtime-methods';s.output=Path(args.output_dir);s.output.mkdir(parents=True,exist_ok=True)
    s.path=s.output/'runtime-methods.json';s.small=s.output/'runtime-methods-summary.txt'
    work,_=session_budget(160,args.max_wall_hours,args.save_reserve_minutes,'server-balanced')
    s.deadline=s.begun+work;s.wall=s.begun+args.max_wall_hours*3600
    if not s.path.exists():restore_artifact(s.path,PREFIX+'/latest.json',hub)
    rows={n:dict(step=v['step'],source_step=v['step'],history=[],evaluations=[],
          training_windows_seen=v['parent_windows'],best_panel='D') for n,v in c['parent']['branches'].items()}
    s.result=json.loads(s.path.read_text()) if s.path.exists() else dict(contract=c,branches=rows,sessions=[],pending_slots=[],
        cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0))
    if s.result['contract']!=c:raise ValueError('resume contract changed')
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    if runtime!=c['parent']['runtime'] or s.result.get('runtime',runtime)!=runtime:raise ValueError('math runtime changed')
    s.result['runtime']=runtime;s.result.pop('error',None);s.next_sync=0.
    s.invocation=dict(max_wall_hours=args.max_wall_hours,save_reserve_minutes=args.save_reserve_minutes,uploaded_bytes=0)
    s.result['sessions'].append(s.invocation)
    def pending(name):
        if name not in s.result['pending_slots']:s.result['pending_slots'].append(name)
    s.store=ForkStore(Path(args.checkpoint_dir)/'runtime-methods',PREFIX+'/checkpoints',hub,
        on_bootstrap=pending)
    def emit(event):
        print('Engine methods:',event,'status=',s.result.get('status'),
              'steps=',{n:r['step'] for n,r in s.result['branches'].items()},flush=True)
        s.persist(local_only=True)
    s.emit=emit;install_session_control(s,args);return s
