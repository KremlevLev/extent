"""Explicit batch6 fork: immutable parent CURRENT + Adam, separate data clock."""
import copy
import hashlib
import json
from pathlib import Path
import time
from importlib.metadata import version
import numpy as np
from extent.corpus_recovery_160 import (
    source_pin,restore_source,prepare,base_coordinates,adam_counts,Inspector,
    block_forward,replay_report,optimizer,teacher_validation,
)
from extent.corpus_recovery_160 import recipes as parent_recipes
from extent.recovery_durability import RecoveryCheckpointStore
from extent.campaign_checkpoint import write_json_atomic

ROOT=Path(__file__).resolve().parents[1]
GPU_EXPERIMENTS=(161,)
TPU_EXPERIMENTS=()
LIMITS={161:8760.}
BATCH=6
PREFIX='experiments/exp161-batch6-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def recipes(experiment):
    if experiment!=161:raise ValueError('EXP161 only')
    # mode remains a template-construction detail: actual q AND optimizer MUST
    # be restored from own checkpoint or verified parent, never fresh template.
    return parent_recipes(160)


def freeze_parent(work,hub=None,deadline=None):
    """Caller holds the shared server lock; parent files are never changed."""
    from extent.hf_artifact_sync import restore_artifact,upload_artifact
    work=Path(work);plan=work/'batch6/source.json';plan.parent.mkdir(parents=True,exist_ok=True)
    def publish():
        if hub is None:return
        from extent.hf_download_retry import DownloadPolicy
        until=min(time.time()+600,deadline if deadline is not None else time.time()+600)
        policy=DownloadPolicy(work/'batch6/transport',deadline=until)
        policy.call(upload_artifact,plan,PREFIX+'/source.json',hub,
                    commit_message='EXP161 immutable batch6 parent plan')
    if not plan.exists() and hub is not None:restore_artifact(plan,PREFIX+'/source.json',hub)
    if plan.exists():
        value=json.loads(plan.read_text())
        if value['format']!='batch6-parent-v1' or value['batch']!=6:raise ValueError('wrong batch source plan')
        publish()
        return plan,value
    source=work/'results/runtime-report.json';parent=json.loads(source.read_text())
    c=parent['contract']
    if c['protocol']!='exp160-corpus-server-v1':raise ValueError('parent must be saved server160')
    if parent.get('status') not in ('deadline_partial','completed','finalization_partial','storage_partial','data_exhausted'):
        raise ValueError('parent did not finish; wait for save/exit before fork')
    for name,expected in c['code'].items():
        if sha(ROOT/name)!=expected:raise ValueError('parent scientific source differs: '+name)
    value=dict(format='batch6-parent-v1',batch=6,parent_report_sha=sha(source),
        parent_contract=c,runtime=parent['runtime'],corpus_identity=c['corpus_identity'],branches={})
    if 'teacher_c4_reference' in parent:value['teacher_c4_reference']=parent['teacher_c4_reference']
    for spec in recipes(161):
        name=spec['name'];row=parent['branches'][name]
        if row.get('source_step')!=118304:raise ValueError('unsupported parent data/Adam clock')
        if not row.get('source_replay',{}).get('passed'):raise ValueError('parent source replay did not pass')
        path=work/'states/runtime-report'/name/'checkpoint.json'
        meta=json.loads(path.read_text())
        if meta['contract']!=dict(c,arm=name) or meta['step']!=row['step']:
            raise ValueError('parent report/checkpoint disagree; do not silently roll back')
        reference=row.get('final',{}).get('last')
        if not reference or reference.get('step')!=meta['step']:
            reference=next((x for x in row.get('evaluations',[]) if x['step']==meta['step']),None)
        value['branches'][name]=dict(step=meta['step'],adam_count=meta['step']-118304,
            parent_windows=meta['step']-118304,metadata=meta,manifest_sha=sha(path),
            current_reference=reference,parent_source_replay=row.get('source_replay'))
    write_json_atomic(plan,value)
    publish()
    return plan,value


def contract(experiment,worker,max_steps,corpus_dir,execution,source_plan):
    from scripts.m3q_corpus_recovery_160 import contract_for
    if experiment!=161 or worker is not None or execution!='server-balanced':raise ValueError('single-GPU EXP161 only')
    parent=json.loads(Path(source_plan).read_text())
    c=contract_for(160,None,max_steps,corpus_dir,'server-balanced')
    if parent['format']!='batch6-parent-v1' or parent['batch']!=BATCH or parent['corpus_identity']!=c['corpus_identity']:
        raise ValueError('batch6 parent/corpus identity differs')
    c.update(experiment=161,protocol='exp161-batch6-v1',execution_namespace=PREFIX,batch=BATCH,
        parent=parent,parent_plan_sha=sha(source_plan),recipes=recipes(161),
        preparation='restore each parent160 CURRENT q/Adam/best; no fold/reset/rank growth; original immutable155 only builds shape/base',
        optimizer='PRESERVE parent Adam moments/count and original LR schedule; no LR scaling with batch',
        clock='global and Adam increment1 per batch6 update; each branch data window increments6 from its own pinned parent origin',
        continuation='same output KD, batch6 on both Wiki control and fresh C4; comparison within this fork, not causal batch-speed/quality proof')
    for file in ('extent/batch_recovery_161.py','extent/batch_execution_161.py','extent/device_execution.py','scripts/m3q_batch_recovery_161.py','engine_batch.py'):
        c['code'][file]=sha(ROOT/file)
    return c


def windows_seen(c,name,global_step):
    origin=c['parent']['branches'][name]
    delta=int(global_step)-origin['step']
    if delta<0:raise ValueError('global cursor before parent CURRENT')
    return origin['parent_windows']+c['batch']*delta


def training_windows(wiki,corpus,c,name,global_step):
    from extent.competitor_tpu import training_batch
    origin=windows_seen(c,name,global_step)
    stream=next(x['data_stream'] for x in c['recipes'] if x['name']==name)
    if stream=='c4':return np.concatenate([corpus.batch(origin+i) for i in range(c['batch'])],axis=0)
    # Parent Wiki selected windows using GLOBAL, not zero-based C4 NEW.
    return np.concatenate([training_batch(wiki,118304+origin+i,133) for i in range(c['batch'])],axis=0)


def parent_replay(control,source):
    reference=(source.get('current_reference') or {}).get('panels',{})
    result=dict(available=bool(reference),passed=None,scope='parent CURRENT; original source/chunks/clocks separately verified')
    if not reference:return result
    errors={}
    for panel,ref in reference.items():
        if panel not in control or 'windows' not in ref:continue
        a=np.asarray(control[panel]['windows']);b=np.asarray(ref['windows'])
        if a.shape!=b.shape:raise ValueError('parent replay window shapes differ')
        errors[panel]=float(np.max(np.abs(a-b))) if np.isfinite(a).all() and np.isfinite(b).all() else float('inf')
    result.update(available=bool(errors),passed=all(e<=.03 for e in errors.values()) if errors else None,
                  max_errors=errors,tolerance=.03)
    return result


class ForkStore(RecoveryCheckpointStore):
    def __init__(self,*args,parent_root,source_plan,on_bootstrap=lambda name:None,**kwargs):
        super().__init__(*args,**kwargs);self.parent_root=Path(parent_root);self.plan=source_plan;self.on_bootstrap=on_bootstrap
    def restore(self,name,c,template,layouts=None):
        own=super().restore(name,c,template,layouts)
        if own is not None:
            marker=self.directory(name)/'synced.json'
            if not marker.exists() or json.loads(marker.read_text()).get('generation')!=own[1]['generation']:
                self.on_bootstrap(name)
            return own
        source=self.plan['branches'][name]
        path=self.parent_root/name/'checkpoint.json'
        if sha(path)!=source['manifest_sha'] or json.loads(path.read_text())!=source['metadata']:
            raise ValueError('pinned parent checkpoint changed; refusing fresh bootstrap')
        reader=RecoveryCheckpointStore(self.parent_root,'read-only',None)
        saved=reader.restore(name,source['metadata']['contract'],template,layouts)
        if saved is None:raise FileNotFoundError('parent CURRENT state missing')
        if adam_counts(saved[0]['optimizer'])!=[source['adam_count']]*6:raise ValueError('parent Adam count differs')
        # Independent baseline snapshot before any batch6 update; parent chunks
        # are never relabelled/overwritten and can remain independently resumable.
        meta=self.save(name,c,saved[0],source['step']);self.on_bootstrap(name)
        return saved[0],meta


def make_session(args,c,hub):
    import jax
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from extent.hf_artifact_sync import restore_artifact
    from extent.server_control import install_session_control
    from scripts.m3q_corpus_recovery_160 import session_budget
    s=Transport.__new__(Transport);s.args=args;s.contract=c;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    s.prefix=PREFIX;s.stem='runtime-b6';s.output=Path(args.output_dir);s.output.mkdir(parents=True,exist_ok=True)
    s.path=s.output/'runtime-b6.json';s.small=s.output/'runtime-b6-summary.txt'
    work,_=session_budget(160,args.max_wall_hours,args.save_reserve_minutes,'server-balanced')
    s.deadline=s.begun+work;s.wall=s.begun+args.max_wall_hours*3600
    if not s.path.exists():restore_artifact(s.path,PREFIX+'/latest.json',hub)
    rows={name:dict(step=v['step'],source_step=v['step'],parent_global_step=v['step'],
        parent_adam_count=v['adam_count'],history=[],evaluations=[],training_windows_seen=v['parent_windows'])
        for name,v in c['parent']['branches'].items()}
    s.result=json.loads(s.path.read_text()) if s.path.exists() else dict(contract=c,branches=rows,sessions=[],pending_slots=[],
        cost=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0))
    if s.result['contract']!=c:raise ValueError('batch6 resume contract differs')
    runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')}
    runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
    if runtime!=c['parent']['runtime'] or s.result.get('runtime',runtime)!=runtime:raise ValueError('parent/own math runtime changed')
    s.result['runtime']=runtime;s.result.pop('error',None);s.next_sync=0.
    if 'teacher_c4_reference' in c['parent']:s.result.setdefault('teacher_c4_reference',c['parent']['teacher_c4_reference'])
    s.invocation=dict(max_wall_hours=args.max_wall_hours,save_reserve_minutes=args.save_reserve_minutes,uploaded_bytes=0)
    s.result['sessions'].append(s.invocation)
    def pending(name):
        if name not in s.result['pending_slots']:s.result['pending_slots'].append(name)
    s.store=ForkStore(Path(args.checkpoint_dir)/'runtime-b6',PREFIX+'/checkpoints',hub,
        parent_root=Path(args.checkpoint_dir)/'runtime-report',source_plan=c['parent'],on_bootstrap=pending)
    def emit(event):
        print('Engine batch6:',event,'status=',s.result.get('status'),
              'steps=',{n:r['step'] for n,r in s.result['branches'].items()},flush=True)
        s.persist(local_only=True)
    s.emit=emit;install_session_control(s,args);return s


def make_kernel(cfg,tc,columns,spec,tx,*,tpu=False):
    if tpu:raise ValueError('single GPU only')
    from extent.batch_execution_161 import BatchStep
    return BatchStep(cfg,tc,columns,spec,tx)
