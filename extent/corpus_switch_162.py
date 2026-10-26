"""Same CURRENT/Adam, two data streams; EXP161 files remain immutable."""
import copy
import json
from pathlib import Path
import time
from importlib.metadata import version
import numpy as np
from extent.batch_recovery_161 import (
    ROOT,sha,parent_replay,make_kernel,source_pin,restore_source,prepare,
    base_coordinates,adam_counts,Inspector,block_forward,replay_report,optimizer,
    teacher_validation,
)
from extent.corpus_recovery_160 import recipes as original_recipes
from extent.recovery_durability import RecoveryCheckpointStore
from extent.campaign_checkpoint import write_json_atomic

PREFIX='experiments/exp162-corpus-switch-v1'
GPU_EXPERIMENTS=(162,)
TPU_EXPERIMENTS=()
LIMITS={162:8760.}
SOURCE_ARM='KD-C4-FRESH'
SOURCE_STEP=154852
BATCH=6


def recipes(experiment):
    if experiment!=162:raise ValueError('EXP162 only')
    source=original_recipes(160)[1]
    return [dict(source,name=name,data_stream=stream) for name,stream in
            [('C4-CONTINUE','c4'),('EDU-FRESH','edu')]]


def freeze_parent(work,hub=None,deadline=None):
    from extent.hf_artifact_sync import upload_artifact
    from extent.hf_download_retry import DownloadPolicy
    work=Path(work);path=work/'corpus-switch/source.json';path.parent.mkdir(parents=True,exist_ok=True)
    def publish():
        if hub is not None:
            until=min(time.time()+600,deadline or time.time()+600)
            DownloadPolicy(work/'corpus-switch/transport',deadline=until).call(
                upload_artifact,path,PREFIX+'/source.json',hub,commit_message='EXP162 immutable C4 CURRENT parent')
    if path.exists():
        value=json.loads(path.read_text())
        if value.get('format')!='corpus-switch-parent-v1':raise ValueError('wrong source plan')
        publish();return path,value
    report_path=work/'results/runtime-b6.json';report=json.loads(report_path.read_text())
    if report['contract']['protocol']!='exp161-batch6-v1' or report.get('pending_slots'):
        raise ValueError('require saved EXP161, no pending state')
    if report.get('status') not in ('deadline_partial','completed','finalization_partial','storage_partial','data_exhausted'):
        raise ValueError('parent still running or failed')
    row=report['branches'][SOURCE_ARM]
    if row['step']!=SOURCE_STEP or row.get('durable_step')!=SOURCE_STEP:
        raise ValueError('expected durable C4 CURRENT154852; never fall back to Wiki BEST')
    if row.get('failed'):raise ValueError('failed parent branch')
    for name,digest in report['contract']['code'].items():
        if sha(ROOT/name)!=digest:raise ValueError('parent scientific code changed: '+name)
    pointer=work/'states/runtime-b6'/SOURCE_ARM/'checkpoint.json';meta=json.loads(pointer.read_text())
    if meta['step']!=SOURCE_STEP or meta['contract']!=dict(report['contract'],arm=SOURCE_ARM):
        raise ValueError('parent CURRENT pointer/report mismatch')
    ref=row.get('final',{}).get('last')
    if not ref or ref['step']!=SOURCE_STEP:raise ValueError('parent CURRENT final reference required')
    value=dict(format='corpus-switch-parent-v1',batch=6,parent_report_sha=sha(report_path),
        runtime=report['runtime'],parent_contract=report['contract'],
        corpus_identity=report['contract']['corpus_identity'],branches={})
    for spec in recipes(162):
        value['branches'][spec['name']]=dict(step=SOURCE_STEP,adam_count=SOURCE_STEP-118304,
            parent_windows=row['training_windows_seen'] if spec['data_stream']=='c4' else 0,
            source_arm=SOURCE_ARM,metadata=meta,manifest_sha=sha(pointer),current_reference=ref)
    write_json_atomic(path,value)
    publish()
    return path,value


def contract(experiment,worker,max_steps,corpus_dir,execution,source_plan,edu_dir):
    from scripts.m3q_corpus_recovery_160 import contract_for
    from extent.broad_corpus import PackedCorpus
    if experiment!=162 or worker is not None or execution!='server-balanced':raise ValueError('one GPU EXP162 only')
    parent=json.loads(Path(source_plan).read_text())
    c=contract_for(160,None,max_steps,corpus_dir,'server-balanced')
    c4=PackedCorpus(corpus_dir);edu=PackedCorpus(edu_dir)
    if parent.get('format')!='corpus-switch-parent-v1' or parent['corpus_identity']!=c['corpus_identity']:
        raise ValueError('parent/C4 identity mismatch')
    ec=edu.manifest['contract']
    from scripts.prepare_edu_corpus import REVISION
    if (ec.get('dataset')!='HuggingFaceFW/fineweb-edu' or ec.get('revision')!=REVISION
        or ec.get('excluded_c4_identity')!=c4.identity or ec.get('sequence_length')!=256
        or ec.get('tokenizer_revision')!=c4.manifest['contract']['tokenizer_revision']):
        raise ValueError('wrong Edu corpus or missing shared validation exclusions')
    remaining=(len(c4.arrays['train'])//256-parent['branches']['C4-CONTINUE']['parent_windows'])//BATCH
    horizon=min(int(max_steps),remaining,len(edu.arrays['train'])//(256*BATCH))
    if horizon<1:raise ValueError('no unused matched training windows')
    c.update(experiment=162,protocol='exp162-corpus-switch-v1',execution_namespace=PREFIX,batch=BATCH,batch_size=BATCH,
        parent=parent,parent_plan_sha=sha(source_plan),recipes=recipes(162),horizon=horizon,
        first_goal_new_steps=min(8192,horizon),edu_identity=edu.identity,edu_contract=ec,
        preparation='same EXP161 C4 CURRENT154852 q/Adam for both; BEST reset to same CURRENT',
        optimizer='preserve Adam moments/count and schedule; identical batch6 and LR',
        selection='C4 D; shared diagnostic selection, NOT an untouched test; CURRENT always retained',
        clock='global/Adam +1/update; each own data cursor +6; no cycling; matched maximum updates',
        continuation='C4 continuation vs FineWeb-Edu switch; all math/weights equal at fork',
        comparison='different corpus intervention at matched new tokens from identical q/Adam; domain/filtering/novelty all differ, not isolated cleaning effect',
        evaluation='same A/B/C/D/E source/last/best; teacher D/E references; EN/RU16 during progress,128 final; diagnostic validation only',
        gradient='batch6 compiled gradient checked against mean of6 fixed-q streamed gradients; ONE Adam update; finite proposal gate',
        notifications='launcher start and finish only; console heartbeat5min, no automatic Telegram progress')
    c['eval_steps']=sorted(v for v in set([0,128,512,1024,2048]+list(range(4096,horizon+1,4096))+[horizon]) if v<=horizon)
    c['global_eval_steps']=[]
    for file in ('extent/corpus_switch_162.py','scripts/m3q_corpus_switch_162.py',
                 'scripts/prepare_edu_corpus.py','engine_compare.py','extent/batch_execution_161.py'):
        c['code'][file]=sha(ROOT/file)
    return c


def windows_seen(c,name,step):
    source=c['parent']['branches'][name];delta=int(step)-source['step']
    if delta<0:raise ValueError('cursor before source')
    return source['parent_windows']+c['batch']*delta


def training_windows(wiki,corpora,c,name,step):
    stream=next(s['data_stream'] for s in c['recipes'] if s['name']==name)
    origin=windows_seen(c,name,step)
    return np.concatenate([corpora[stream].batch(origin+i) for i in range(c['batch'])],axis=0)


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
        source=self.plan['branches'][name];slot=source['source_arm']
        pointer=self.parent_root/slot/'checkpoint.json'
        if sha(pointer)!=source['manifest_sha'] or json.loads(pointer.read_text())!=source['metadata']:
            raise ValueError('parent changed or disappeared; refusing bootstrap')
        saved=RecoveryCheckpointStore(self.parent_root,'read-only',None).restore(slot,source['metadata']['contract'],template,layouts)
        if saved is None:raise FileNotFoundError('parent CURRENT missing')
        if adam_counts(saved[0]['optimizer'])!=[source['adam_count']]*6:raise ValueError('Adam count mismatch')
        import jax.numpy as jnp
        payload=dict(saved[0],best=saved[0]['parameters'],best_step=jnp.asarray(source['step'],jnp.int32))
        meta=self.save(name,c,payload,source['step']);self.on_bootstrap(name)
        return payload,meta


def make_session(args,c,hub):
    import jax
    from scripts.m3q_peer_tpu_campaign import Session as Transport
    from extent.hf_artifact_sync import restore_artifact
    from extent.server_control import install_session_control
    from scripts.m3q_corpus_recovery_160 import session_budget
    s=Transport.__new__(Transport);s.args=args;s.contract=c;s.hub=hub;s.begun=time.monotonic();s.checkpoint_signatures={}
    s.prefix=PREFIX;s.stem='runtime-compare';s.output=Path(args.output_dir);s.output.mkdir(parents=True,exist_ok=True)
    s.path=s.output/'runtime-compare.json';s.small=s.output/'runtime-compare-summary.txt'
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
    s.store=ForkStore(Path(args.checkpoint_dir)/'runtime-compare',PREFIX+'/checkpoints',hub,
        parent_root=Path(args.checkpoint_dir)/'runtime-b6',source_plan=c['parent'],on_bootstrap=pending)
    def emit(event):
        print('Engine comparison:',event,'status=',s.result.get('status'),
              'steps=',{n:r['step'] for n,r in s.result['branches'].items()},flush=True)
        s.persist(local_only=True)
    s.emit=emit;install_session_control(s,args);return s
