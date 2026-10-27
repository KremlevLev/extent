"""EXP166: continue EXP165 FULL-KD, adaptive physical batch, explicit data clock."""
import json,time,os
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
PREFIX='experiments/exp166-full-stream-v1'
GPU_EXPERIMENTS=(166,)
TPU_EXPERIMENTS=()
SOURCE_ARM='FULL-KD'
SOURCE_STEP=157685
SOURCE_WINDOWS=16993
SOURCE_ADAM=2833


def recipes(experiment):
    if experiment!=166:raise ValueError('EXP166 only')
    return [dict(recipes163(163)[2],mode='carry-full-adam')]


def freeze_parent(work,hub=None,deadline=None):
    from extent.hf_artifact_sync import upload_artifacts_together,restore_artifact
    from extent.hf_download_retry import DownloadPolicy
    work=Path(work);directory=work/'full-stream';path=directory/'source.json';manifest=directory/'parent-manifest.json'
    directory.mkdir(parents=True,exist_ok=True)
    if path.exists():
        value=json.loads(path.read_text())
        if not manifest.exists() and hub is not None:restore_artifact(manifest,PREFIX+'/parent-manifest.json',hub)
    else:
        report_path=work/'results/runtime-full-next.json';report=json.loads(report_path.read_text());row=report['branches'][SOURCE_ARM]
        if (report['contract']['protocol']!='exp165-full-steady-v1' or report.get('pending_slots')
            or row['step']!=SOURCE_STEP or row.get('durable_step')!=SOURCE_STEP or row.get('accepted_new_updates',0)!=0):
            raise ValueError('require durable unchanged EXP165 CURRENT157685')
        for file,digest in report['contract']['code'].items():
            if sha(ROOT/file)!=digest:raise ValueError('parent scientific code changed: '+file)
        pointer=work/'states/runtime-full-next'/SOURCE_ARM/'checkpoint.json';meta=json.loads(pointer.read_text())
        if meta['step']!=SOURCE_STEP or meta['contract']!=dict(report['contract'],arm=SOURCE_ARM):raise ValueError('parent checkpoint mismatch')
        if row['training_windows_seen']!=SOURCE_WINDOWS:raise ValueError('parent Edu cursor mismatch')
        manifest.write_bytes(pointer.read_bytes())
        branch=report['contract']['parent']['branches'][SOURCE_ARM]
        base_branch={k:branch[k] for k in ('step','adam_count','manifest_sha')}
        base_branch['parent_windows']=0
        value=dict(format='methods-parent-v1',runtime=report['runtime'],corpus_identity=report['corpus_identity'],
            branches={SOURCE_ARM:base_branch},full_source=dict(step=SOURCE_STEP,windows=SOURCE_WINDOWS,adam_count=SOURCE_ADAM,
            runtime=report['runtime'],manifest_sha=sha(manifest),manifest_file='full-stream/parent-manifest.json',
            source_namespace='runtime-full-next',source_prefix='experiments/exp165-full-steady-v1/checkpoints',
            report_sha=sha(report_path),current_reference=dict(step=SOURCE_STEP,panels=row['source_controls'])))
        write_json_atomic(path,value)
    if sha(manifest)!=value['full_source']['manifest_sha']:raise ValueError('parent manifest SHA mismatch')
    if hub is not None:
        marker=directory/'source-published.json'
        signature=dict(plan=sha(path),manifest=sha(manifest))
        if not marker.exists() or json.loads(marker.read_text())!=signature:
            DownloadPolicy(directory/'transport',deadline=min(time.time()+600,deadline or time.time()+600)).call(
                upload_artifacts_together,[(path,PREFIX+'/source.json'),(manifest,PREFIX+'/parent-manifest.json')],hub,
                commit_message='EXP166 external immutable source manifest')
            write_json_atomic(marker,signature)
    return path,value


def contract(experiment,worker,max_steps,corpus_dir,execution,source_plan,edu_dir):
    from extent.method_sweep_163 import contract as old_contract
    c=old_contract(163,worker,max_steps,corpus_dir,execution,source_plan,edu_dir);source=c['parent']['full_source']
    if experiment!=166 or (source['step'],source['adam_count'],source['windows'])!=(SOURCE_STEP,SOURCE_ADAM,SOURCE_WINDOWS):raise ValueError('unexpected source clocks')
    c.update(experiment=166,protocol='exp166-full-stream-v1',execution_namespace=PREFIX,recipes=recipes(experiment),full_source=source,
        batch=None,batch_size=None,horizon=int(max_steps),first_goal_new_steps=int(max_steps),
        eval_steps=[0,128,512,1024,2048]+list(range(4096,min(int(max_steps),400000)+1,4096)),
        preparation='unchanged EXP165 CURRENT157685; full q/Adam2833/Edu16993; mandatory parent NLL replay',
        optimizer='carry full Adam, schedule, .5CE+.5KL and full-scope LRs; no reset',
        clock='Adam/global+1 per accepted update, checkpointed Edu cursor advances by actual batch; no repeats',
        gradient='BATCHED streamed exact chain-rule through all layers; one Adam update per whole batch; mean-microgradient audit',
        gpu='full decoder; layer activations staged in host RAM; frozen teacher GPU resident; BEST on host; no monolithic whole-model executable',
        budget_policy='measure all batches1..8 in actual batched streamed mode; fastest finite input tokens/sec; detailed allocator diagnostics',
        continuation='single full decoder; execution intervention only; adaptive batch and token accounting explicit',
        checkpoint='no initial full-state copy/upload; immutable parent reference until first accepted step; then pruned CURRENT every30min and exit',
        source_provenance='external immutable source manifest checked by SHA, not recursively embedded into reports')
    for name in ('extent/full_stream_166.py','extent/stream_batch_planner.py','scripts/m3q_full_stream_166.py','engine_full_stream.py','extent/steady_batch_planner.py'):
        c['code'][name]=sha(ROOT/name)
    return c


def prepare(patient,spec,payload,pin,ancestor=None,*,parent,parent_root):
    pointer=Path(parent_root)/'KD-C4-FRESH/checkpoint.json'
    source=dict(parent['branches'][spec['name']])
    if sha(pointer)!=source['manifest_sha']:raise ValueError('base161 manifest changed')
    source['metadata']=json.loads(pointer.read_text())
    expanded=dict(parent,branches={spec['name']:source})
    base,q,_=prepare163(patient,spec,payload,pin,ancestor,parent=expanded,parent_root=parent_root)
    return base,q,SOURCE_STEP


class ForkStore(RecoveryCheckpointStore):
    def __init__(self,*args,on_bootstrap=lambda name:None,source,parent_root,manifest_file,**kwargs):
        super().__init__(*args,**kwargs);self.source=source;self.parent_root=parent_root;self.manifest_file=manifest_file
    def restore(self,name,c,template,layouts=None):
        own=super().restore(name,c,template,layouts)
        if own is not None:return own
        if sha(self.manifest_file)!=self.source['manifest_sha']:raise ValueError('source manifest SHA mismatch')
        meta=json.loads(Path(self.manifest_file).read_text())
        pointer=Path(self.parent_root)/name/'checkpoint.json'
        if not pointer.exists() and self.hub is not None:
            from extent.hf_artifact_sync import restore_artifact
            restore_artifact(pointer,self.source['source_prefix']+'/'+name+'/checkpoint.json',self.hub)
        if sha(pointer)!=self.source['manifest_sha']:raise ValueError('source pointer changed')
        saved=RecoveryCheckpointStore(self.parent_root,self.source['source_prefix'],self.hub).restore(name,meta['contract'],template,layouts)
        if saved is None or saved[1]['step']!=SOURCE_STEP or adam_counts(saved[0]['optimizer'])!=[SOURCE_ADAM]*6:raise ValueError('source CURRENT/Adam mismatch')
        if int(saved[0]['data_windows'])!=SOURCE_WINDOWS:raise ValueError('source data clock mismatch')
        # Source is already durable. No new 22GB generation merely for preflight.
        return saved


def make_session(args,c,hub):
    import jax
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from extent.hf_artifact_sync import restore_artifact
    from extent.server_control import install_session_control
    from scripts.m3q_corpus_recovery_160 import session_budget
    s=Transport.__new__(Transport);s.args=args;s.contract=c;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    s.prefix=PREFIX;s.stem='runtime-full-stream';s.output=Path(args.output_dir);s.output.mkdir(parents=True,exist_ok=True)
    s.path=s.output/'runtime-full-stream.json';s.small=s.output/'runtime-full-stream-summary.txt'
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
    if os.environ.get('EXTENT_GPU_BUDGET_POLICY'):
        s.invocation['gpu_budget_policy']=json.loads(os.environ['EXTENT_GPU_BUDGET_POLICY'])
        s.result['gpu_budget_policy']=s.invocation['gpu_budget_policy']
    s.result['sessions'].append(s.invocation)
    def pending(name):
        if name not in s.result['pending_slots']:s.result['pending_slots'].append(name)
    s.store=ForkStore(Path(args.checkpoint_dir)/'runtime-full-stream',PREFIX+'/checkpoints',hub,
        on_bootstrap=pending,source=c['full_source'],parent_root=Path(args.checkpoint_dir)/'runtime-full-next',manifest_file=Path(args.work_root)/c['full_source']['manifest_file'])
    def emit(event):
        print('Engine stream:',event,'status=',s.result.get('status'),
              'steps=',{n:r['step'] for n,r in s.result['branches'].items()},flush=True)
        s.persist(local_only=True)
    s.emit=emit;install_session_control(s,args)
    bounded=s.checkpoint
    def accepted_checkpoint(name,payload,step):
        if s.result['branches'][name].get('invocation_updates',0)==0:
            s.result['branches'][name]['no_update_checkpoint_reused']=True
            if (s.store.directory(name)/'checkpoint.json').exists():
                s.result['branches'][name]['checkpoint_reused']='own CURRENT'
            else:s.result['branches'][name]['durable_source']=dict(experiment=165,step=SOURCE_STEP,manifest_sha=c['full_source']['manifest_sha'])
            return
        bounded(name,payload,step)
    s.checkpoint=accepted_checkpoint
    return s
