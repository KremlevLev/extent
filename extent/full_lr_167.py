"""EXP167: matched LR intervention from EXP166 CURRENT181145, carrying Adam."""
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
PREFIX='experiments/exp167-lr-sweep-v1'
GPU_EXPERIMENTS=(167,)
TPU_EXPERIMENTS=()
SOURCE_ARM='FULL-KD'
SOURCE_STEP=181145
SOURCE_WINDOWS=204673
SOURCE_ADAM=26293


def recipes(experiment):
    if experiment!=167:raise ValueError('EXP167 only')
    return [dict(recipes163(163)[2],mode='carry-full-adam')]


def freeze_parent(work,hub=None,deadline=None,multiplier=1):
    from extent.hf_artifact_sync import upload_artifacts_together
    from extent.hf_download_retry import DownloadPolicy
    if multiplier not in (1,3,10):raise ValueError('LR multiplier must be1,3,10')
    work=Path(work);directory=work/'lr-sweep';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'source-x{multiplier}.json';manifest=directory/'parent-manifest.json'
    if path.exists():
        value=json.loads(path.read_text())
        if value['lr_multiplier']!=multiplier:raise ValueError('source LR mismatch')
    else:
        report_path=work/'results/runtime-full-stream.json';report=json.loads(report_path.read_text());row=report['branches'][SOURCE_ARM]
        if (report['contract']['protocol']!='exp166-full-stream-v1' or report.get('pending_slots')
            or row['step']!=SOURCE_STEP or row.get('durable_step')!=SOURCE_STEP
            or report.get('finalization_status')!='completed' or row['training_windows_seen']!=SOURCE_WINDOWS):
            raise ValueError('require durable EXP166 CURRENT181145 with completed final panels')
        for file,digest in report['contract']['code'].items():
            if sha(ROOT/file)!=digest:raise ValueError('parent scientific code changed: '+file)
        pointer=work/'states/runtime-full-stream'/SOURCE_ARM/'checkpoint.json'
        if not pointer.exists() and hub is not None:
            from extent.hf_artifact_sync import restore_artifact
            pointer.parent.mkdir(parents=True,exist_ok=True)
            if not restore_artifact(pointer,'experiments/exp166-full-stream-v1/checkpoints/'+SOURCE_ARM+'/checkpoint.json',hub):
                raise FileNotFoundError('EXP166 CURRENT manifest missing locally and on HF')
        meta=json.loads(pointer.read_text())
        if meta['step']!=SOURCE_STEP or meta['contract']!=dict(report['contract'],arm=SOURCE_ARM):raise ValueError('parent checkpoint mismatch')
        if manifest.exists() and sha(manifest)!=sha(pointer):raise ValueError('frozen parent changed')
        manifest.write_bytes(pointer.read_bytes())
        base_branch=dict(report['contract']['parent']['branches'][SOURCE_ARM])
        value=dict(format='methods-parent-v1',runtime=report['runtime'],corpus_identity=report['corpus_identity'],
            lr_multiplier=multiplier,branches={SOURCE_ARM:base_branch},full_source=dict(step=SOURCE_STEP,
            windows=SOURCE_WINDOWS,adam_count=SOURCE_ADAM,runtime=report['runtime'],manifest_sha=sha(manifest),
            manifest_file='lr-sweep/parent-manifest.json',source_namespace='runtime-full-stream',
            source_prefix='experiments/exp166-full-stream-v1/checkpoints',report_sha=sha(report_path),
            current_reference=dict(step=SOURCE_STEP,panels=row['final']['last']['panels']),
            fresh_reference=row['final']['last'].get('fresh_nll',{})))
        write_json_atomic(path,value)
    if sha(manifest)!=value['full_source']['manifest_sha']:raise ValueError('parent manifest SHA mismatch')
    if hub is not None:
        marker=directory/f'published-x{multiplier}.json';signature=dict(plan=sha(path),manifest=sha(manifest))
        if not marker.exists() or json.loads(marker.read_text())!=signature:
            DownloadPolicy(directory/'transport',deadline=min(time.time()+600,deadline or time.time()+600)).call(
                upload_artifacts_together,[(path,PREFIX+f'/x{multiplier}/source.json'),(manifest,PREFIX+'/parent-manifest.json')],hub,
                commit_message='EXP167 immutable shared CURRENT181145 source')
            write_json_atomic(marker,signature)
    return path,value


def contract(experiment,worker,max_steps,corpus_dir,execution,source_plan,edu_dir):
    from extent.method_sweep_163 import contract as old_contract
    c=old_contract(163,worker,max_steps,corpus_dir,execution,source_plan,edu_dir);source=c['parent']['full_source']
    if experiment!=167 or (source['step'],source['adam_count'],source['windows'])!=(SOURCE_STEP,SOURCE_ADAM,SOURCE_WINDOWS):raise ValueError('unexpected source clocks')
    multiplier=c['parent']['lr_multiplier']
    spec=recipes(experiment)[0]
    for key in ('mamba_lr','mlp_lr','norm_lr'):spec[key]*=multiplier
    c.update(experiment=167,protocol='exp167-lr-sweep-v1',execution_namespace=PREFIX+f'/x{multiplier}',recipes=[spec],full_source=source,
        batch=8,batch_size=8,lr_multiplier=multiplier,horizon=int(max_steps),first_goal_new_steps=int(max_steps),
        eval_steps=[0,128,512,1024,2048]+list(range(4096,min(int(max_steps),400000)+1,4096)),
        preparation='EXP166 CURRENT181145; full q/Adam26293/Edu204673; mandatory parent NLL replay',
        optimizer='carry Adam26293 and moments; scale all3 LR groups1/3/10; no reset or new warmup; .5CE+.5KL',
        clock='Adam/global+1 per accepted update, checkpointed Edu cursor advances by actual batch; no repeats',
        gradient='BATCHED streamed exact chain-rule through all layers; one Adam update per whole batch; mean-microgradient audit',
        gpu='full decoder; layer activations staged in host RAM; frozen teacher GPU resident; BEST on host; no monolithic whole-model executable',
        budget_policy='measure batches1..8; require fixed physical batch8 in every LR arm; no batch fallback',
        continuation='LR comparison at fixed batch8 and identical fresh Edu prefix; source optimizer moments carried',
        checkpoint='no initial full-state copy/upload; immutable parent reference until first accepted step; then pruned CURRENT every30min and exit',
        source_provenance='external immutable source manifest checked by SHA, not recursively embedded into reports')
    for name in ('extent/full_lr_167.py','extent/lr_batch_planner_167.py','extent/stream_batch_planner.py','scripts/m3q_full_lr_167.py','engine_full_lr.py','engine_lr.py','extent/steady_batch_planner.py'):
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
        payload=dict(saved[0]);payload['best']=payload['parameters'];payload['best_step']=jnp.asarray(SOURCE_STEP,jnp.int32)
        return payload,saved[1]


def make_session(args,c,hub):
    import jax
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from extent.hf_artifact_sync import restore_artifact
    from extent.server_control import install_session_control
    from scripts.m3q_corpus_recovery_160 import session_budget
    s=Transport.__new__(Transport);s.args=args;s.contract=c;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    s.prefix=c['execution_namespace'];s.stem='runtime-lr-x'+str(c['lr_multiplier']);s.output=Path(args.output_dir);s.output.mkdir(parents=True,exist_ok=True)
    s.path=s.output/(s.stem+'.json');s.small=s.output/(s.stem+'-summary.txt')
    work,_=session_budget(160,args.max_wall_hours,args.save_reserve_minutes,'server-balanced')
    s.deadline=s.begun+work;s.wall=s.begun+args.max_wall_hours*3600
    if getattr(args,'deadline_unix',None) is not None:
        s.wall=min(s.wall,s.begun+max(0,args.deadline_unix-time.time()))
        s.deadline=min(s.deadline,s.wall-args.save_reserve_minutes*60)
    if not s.path.exists():restore_artifact(s.path,s.prefix+'/latest.json',hub)
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
    s.store=ForkStore(Path(args.checkpoint_dir)/s.stem,s.prefix+'/checkpoints',hub,
        on_bootstrap=pending,source=c['full_source'],parent_root=Path(args.checkpoint_dir)/'runtime-full-stream',manifest_file=Path(args.work_root)/c['full_source']['manifest_file'])
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
            else:s.result['branches'][name]['durable_source']=dict(experiment=166,step=SOURCE_STEP,manifest_sha=c['full_source']['manifest_sha'])
            return
        bounded(name,payload,step)
    s.checkpoint=accepted_checkpoint
    return s
