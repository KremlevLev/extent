"""EXP164: continue EXP163 FULL-KD, adaptive physical batch, explicit data clock."""
import json,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import jax
import jax.numpy as jnp
from extent.method_sweep_163 import (ROOT,sha,source_pin,restore_source,base_coordinates,adam_counts,
    Inspector,teacher_validation,block_forward,optimizer,to_host)
from extent.method_sweep_163 import prepare as prepare163, recipes as recipes163
from extent.recovery_durability import RecoveryCheckpointStore
from extent.campaign_checkpoint import write_json_atomic
PREFIX='experiments/exp164-full-adaptive-v1'
GPU_EXPERIMENTS=(164,)
TPU_EXPERIMENTS=()
SOURCE_ARM='FULL-KD'
SOURCE_STEP=157684
SOURCE_WINDOWS=16992
SOURCE_ADAM=2832


def recipes(experiment):
    if experiment!=164:raise ValueError('EXP164 only')
    return [dict(recipes163(163)[2],mode='carry-full-adam')]


def freeze_parent(work,hub=None,deadline=None):
    work=Path(work);path=work/'full-adaptive/source.json'
    if path.exists():value=json.loads(path.read_text())
    else:
        report_path=work/'results/runtime-methods.json';report=json.loads(report_path.read_text())
        row=report['branches'][SOURCE_ARM]
        if (report['contract']['protocol']!='exp163-method-sweep-v1' or report.get('pending_slots')
            or row.get('failed') or row['step']!=SOURCE_STEP or row.get('durable_step')!=SOURCE_STEP):
            raise ValueError('require durable EXP163 FULL-KD CURRENT157684, no pending uploads')
        for file,digest in report['contract']['code'].items():
            if sha(ROOT/file)!=digest:raise ValueError('parent source code changed: '+file)
        pointer=work/'states/runtime-methods'/SOURCE_ARM/'checkpoint.json'
        meta=json.loads(pointer.read_text())
        if meta['step']!=SOURCE_STEP or meta['contract']!=dict(report['contract'],arm=SOURCE_ARM):
            raise ValueError('EXP163 CURRENT pointer/report mismatch')
        if row['training_windows_seen']!=SOURCE_WINDOWS:raise ValueError('unexpected parent Edu cursor')
        value=dict(report['contract']['parent'])
        value['branches']={SOURCE_ARM:value['branches'][SOURCE_ARM]}
        value['full_source']=dict(metadata=meta,manifest_sha=sha(pointer),report_sha=sha(report_path),
            step=SOURCE_STEP,windows=SOURCE_WINDOWS,adam_count=SOURCE_ADAM,
            best_step=row['best_step'],best_nll=row['best_nll'],runtime=report['runtime'])
        write_json_atomic(path,value)
    if value['full_source']['step']!=SOURCE_STEP:raise ValueError('wrong full source plan')
    if hub is not None:
        from extent.hf_artifact_sync import upload_artifact
        from extent.hf_download_retry import DownloadPolicy
        DownloadPolicy(work/'full-adaptive/transport',deadline=min(time.time()+600,deadline or time.time()+600)).call(
            upload_artifact,path,PREFIX+'/source.json',hub,commit_message='EXP164 full CURRENT immutable source')
    return path,value


def contract(experiment,worker,max_steps,corpus_dir,execution,source_plan,edu_dir):
    from extent.method_sweep_163 import contract as old_contract
    c=old_contract(163,worker,max_steps,corpus_dir,execution,source_plan,edu_dir)
    source=c['parent']['full_source']
    if source['step']!=SOURCE_STEP or source['adam_count']!=SOURCE_ADAM or source['windows']!=SOURCE_WINDOWS:
        raise ValueError('unexpected full source clocks')
    if source['runtime']!=c['parent']['runtime']:raise ValueError('full parent runtime differs')
    c.update(experiment=164,protocol='exp164-full-adaptive-v1',execution_namespace=PREFIX,
        recipes=recipes(experiment),full_source=source,batch=None,batch_size=None,
        horizon=int(max_steps),first_goal_new_steps=int(max_steps),
        eval_steps=[0,128,512,1024,2048]+list(range(4096,min(int(max_steps),400000)+1,4096)),
        preparation='EXP163 FULL-KD CURRENT157684; preserve q/Adam2832 and Edu window16992; measured CURRENT baseline',
        optimizer='carry full Adam and warm512 schedule; no reset; same .5CE+.5KL',
        clock='global157684+updates; Adam2832+updates; checkpointed data cursor advances by actual selected batch',
        comparison='single full decoder continuation; adaptive batch is an explicit optimization intervention',
        gradient='mandatory actual-state batch1..8 scan each invocation; finite/timing tests; winner gradient audited against fixed-q streamed reference',
        gpu='one full decoder; no silent accumulated fallback; if compiled batches fail, abort before training',
        budget_policy='all8 batch sizes tested before training, fastest measured input tokens/sec wins; candidate probes never accepted',
        continuation='Edu at saved sequential window; no cycling or dropped windows; changing batch on resume explicitly allowed by this contract')
    for name in ('extent/full_adaptive_164.py','extent/batch_planner.py','scripts/m3q_full_adaptive_164.py','engine_full.py'):
        c['code'][name]=sha(ROOT/name)
    return c


def prepare(patient,spec,payload,pin,ancestor=None,*,parent,parent_root):
    base,q,_=prepare163(patient,spec,payload,pin,ancestor,parent=parent,parent_root=parent_root)
    return base,q,SOURCE_STEP


class ForkStore(RecoveryCheckpointStore):
    def __init__(self,*args,on_bootstrap=lambda name:None,source,parent_root,**kwargs):
        super().__init__(*args,**kwargs);self.on_bootstrap=on_bootstrap;self.source=source;self.parent_root=parent_root
    def restore(self,name,c,template,layouts=None):
        own=super().restore(name,c,template,layouts)
        if own is not None:return own
        pointer=Path(self.parent_root)/name/'checkpoint.json'
        if sha(pointer)!=self.source['manifest_sha']:raise ValueError('EXP163 full pointer changed')
        old_template={k:v for k,v in template.items() if k!='data_windows'}
        old_layout={k:v for k,v in layouts.items() if k!='data_windows'} if layouts else None
        saved=RecoveryCheckpointStore(self.parent_root,'experiments/exp163-method-sweep-v1/checkpoints',self.hub).restore(
            name,self.source['metadata']['contract'],old_template,old_layout)
        if saved is None or saved[1]['step']!=SOURCE_STEP or adam_counts(saved[0]['optimizer'])!=[SOURCE_ADAM]*6:
            raise ValueError('FULL-KD CURRENT/Adam differs from descriptor')
        payload=dict(saved[0],data_windows=jax.device_put(jnp.asarray(SOURCE_WINDOWS,jnp.int32),layouts['data_windows']) if layouts else np.asarray(SOURCE_WINDOWS,np.int32))
        meta=self.save(name,c,payload,SOURCE_STEP);self.on_bootstrap(name)
        return payload,meta


def make_session(args,c,hub):
    import jax
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from extent.hf_artifact_sync import restore_artifact
    from extent.server_control import install_session_control
    from scripts.m3q_corpus_recovery_160 import session_budget
    s=Transport.__new__(Transport);s.args=args;s.contract=c;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    s.prefix=PREFIX;s.stem='runtime-full';s.output=Path(args.output_dir);s.output.mkdir(parents=True,exist_ok=True)
    s.path=s.output/'runtime-full.json';s.small=s.output/'runtime-full-summary.txt'
    work,_=session_budget(160,args.max_wall_hours,args.save_reserve_minutes,'server-balanced')
    s.deadline=s.begun+work;s.wall=s.begun+args.max_wall_hours*3600
    if getattr(args,'deadline_unix',None) is not None:
        s.wall=min(s.wall,s.begun+max(0,args.deadline_unix-time.time()))
        s.deadline=min(s.deadline,s.wall-args.save_reserve_minutes*60)
    if not s.path.exists():restore_artifact(s.path,PREFIX+'/latest.json',hub)
    rows={'FULL-KD':dict(step=SOURCE_STEP,source_step=SOURCE_STEP,history=[],evaluations=[],
          training_windows_seen=SOURCE_WINDOWS,best_panel='D')}
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
    s.store=ForkStore(Path(args.checkpoint_dir)/'runtime-full',PREFIX+'/checkpoints',hub,
        on_bootstrap=pending,source=c['full_source'],parent_root=Path(args.checkpoint_dir)/'runtime-methods')
    def emit(event):
        print('Engine methods:',event,'status=',s.result.get('status'),
              'steps=',{n:r['step'] for n,r in s.result['branches'].items()},flush=True)
        s.persist(local_only=True)
    s.emit=emit;install_session_control(s,args);return s
