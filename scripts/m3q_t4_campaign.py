"""EXP116--119: independent single-GPU workers, bounded layer-streamed patient.

Importing main in the notebook does not initialize JAX. GPU processes receive
CUDA_VISIBLE_DEVICES before importing it; CUDA state never uses fork.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {116: 'layer-profile', 117: 'context-memory', 118: 'precision', 119: 'local-distillation'}
CODE = ('scripts/m3q_t4_campaign.py','extent/t4_patient.py','extent/t4_data.py',
        'extent/t4_search.py','extent/plateau_forensics.py','extent/layers/mamba3.py',
        'extent/model.py','extent/weight_mapping.py','extent/recovery_subspace.py',
        'extent/qwen3_teacher.py','extent/layers/common.py','requirements-t4.txt')


def contract_for(experiment):
    return dict(experiment=experiment, name=EXPERIMENTS[experiment], seed=123,
        source='EXP111 MAIN32768', source_pin_sha=sha(ROOT/'results/EXP-114-source-pin.json'),
        data='A/B/C reused EXP114; FIT first128 EXP115 training windows',
        validation_windows=8, fit_steps=1024, layers=[0,12,26],
        recipes=['ADD32','ADD64','ADD128','ADD256','FREE256','ALL-MAMBA'],
        streaming_token_nll_tolerance=.01, gpu_tpu_per_window_nll_tolerance=.03,
        reset_control_per_window_nll_tolerance=.01, trace_max_abs_tolerance=1e-4,
        interpretation='Exploratory single seed; cross-backend gate required; no independent test claim',
        code={name:sha(ROOT/name) for name in CODE})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def notify(args, message):
    from scripts.qwen_extended_horizon_campaign import _safe_notify
    return _safe_notify(args.telegram, message)


class Session:
    def __init__(self, args, hub):
        from extent.hf_artifact_sync import restore_artifact
        self.args, self.hub = args, hub
        self.prefix = f'experiments/exp{args.experiment}-t4/worker-{args.worker}'
        self.path = Path(args.output_dir)/f'extent-m3q-t4-{args.experiment}-worker-{args.worker}.json'
        self.small = self.path.with_name(self.path.stem+'-summary.txt')
        self.started = time.monotonic(); self.next_sync = 0.; self.uploaded = 0
        if not self.path.exists():
            restore_artifact(self.path, f'{self.prefix}/latest.json', hub)
        c = dict(contract_for(args.experiment), workers=args.workers, worker=args.worker)
        self.result = json.loads(self.path.read_text()) if self.path.exists() else dict(contract=c,
            tasks={}, fits={}, sessions=[], cost={'attempted_fit_updates':0, 'fit_input_tokens':0})
        if self.result['contract'] != c:
            raise ValueError('GPU resume contract changed')
        import jax
        from importlib.metadata import version
        runtime = dict(versions={n:version(n) for n in ('jax','jaxlib','flax','optax')},
            device=jax.devices('gpu')[0].device_kind, matmul_precision=str(jax.config.jax_default_matmul_precision))
        if self.result.get('runtime',runtime) != runtime:
            raise ValueError('GPU runtime changed during resume')
        self.result['runtime'] = runtime
        self.invocation = dict(started_at=time.time(), max_wall_hours=args.max_wall_hours, uploaded_bytes=0)
        self.result['sessions'].append(self.invocation)
        self.result['status']='preflight'; self.result.pop('error',None)

    def check(self):
        if time.time() >= self.args.deadline:
            raise TimeoutError('T4 registered work deadline')

    def persist(self, force=False):
        from extent.hf_artifact_sync import upload_artifacts_together
        self.invocation.update(seconds=time.monotonic()-self.started, uploaded_bytes=self.uploaded)
        write(self.path, self.result)
        write(self.small, dict(status=self.result['status'], completed_tasks=len(self.result['tasks']),
            fits=self.result['fits'], error=self.result.get('error'), cost=self.result['cost'],
            source_replay=self.result.get('source_replay'), report_sync_error=self.result.get('report_sync_error')))
        if time.monotonic() < self.next_sync and not force:
            return False
        size = self.path.stat().st_size + self.small.stat().st_size
        if self.uploaded+size > self.args.max_new_hf_gb*1e9:
            self.result['report_sync_error']='upload cap'; write(self.path,self.result); return False
        try:
            upload_artifacts_together([(self.path,f'{self.prefix}/latest.json'),
                (self.small,f'{self.prefix}/latest-summary.txt')], self.hub, commit_message=f'T4 EXP{self.args.experiment} worker{self.args.worker}')
            self.uploaded += size; self.next_sync=time.monotonic()+600
            self.result.pop('report_sync_error',None); write(self.path,self.result); return True
        except Exception as exc:
            self.result['report_sync_error']=dict(type=type(exc).__name__, http_status=getattr(getattr(exc,'response',None),'status_code',None))
            self.next_sync=time.monotonic()+600; write(self.path,self.result); return False

    def task(self, key, call):
        if key in self.result['tasks']:
            return self.result['tasks'][key]['value']
        self.check(); start=time.monotonic()
        value=call()
        self.result['tasks'][key]=dict(seconds=time.monotonic()-start,value=value)
        print(f'EXP{self.args.experiment} GPU{self.args.worker} {key}: saved',flush=True)
        self.persist(); return value


def baseline(s, patient, tokens):
    import numpy as np
    expected=json.loads((ROOT/'results/EXP-115-source-validation.json').read_text())['windows']
    # Check every A window, not only an aggregate. GPU results use their own
    # paired baseline; tolerance is fixed before hardware execution.
    value=s.task('source-replay', lambda:patient.evaluate(tokens['A']))
    errors=np.abs(np.asarray(value['windows'])-np.asarray(expected))
    report=dict(max_abs_per_window=float(errors.max()), tolerance=.03,
        passed=bool(np.isfinite(errors).all() and errors.max()<=.03),
        gpu_nll=value['nll'], tpu_nll=float(np.mean(expected)), folded_sha=patient.folded_sha)
    s.result['source_replay']=report; s.persist()
    if not report['passed']:
        raise ValueError('GPU patient replay differs from TPU: stop before interventions')


def profile(s,p,tokens):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from extent.plateau_forensics import layer_profile, plain, norm, mixer, mamba_trace, compress_trace
    from extent.efficiency_diagnostics import relative
    for layer in range(p.cfg.num_layers):
        if layer%s.args.workers != s.args.worker: continue
        def run(layer=layer):
            rows=[]
            current=p.placed(p.current[f'layers_{layer}']);teacher=p.placed(p.teacher[f'layers_{layer}'])
            fn=jax.jit(lambda x,y:layer_profile(current,teacher,x,y,p.cfg,layer,p.tc))
            for t in tokens['A'][:8]:
                s.check();t=t[None]
                x=p.forward_hidden(t,stop=layer);y=p.forward_hidden(t,teacher=True,stop=layer)
                rows.append(plain(jax.block_until_ready(fn(x,y))))
            result=dict(layer=layer,profile_windows=rows)
            if layer in p.cfg.mamba_layer_indices:
                t=tokens['A'][0][None]; x=p.forward_hidden(t,stop=layer)
                n=norm(x,current['input_layernorm'],p.cfg)
                actual=mixer(n,current,p.cfg,layer)
                traced,detail=jax.jit(lambda z:mamba_trace(z,current['mamba'],p.cfg))(n)
                error=float(jnp.max(jnp.abs(actual.astype(jnp.float32)-traced.astype(jnp.float32))))
                result['trace_parity']=dict(max_abs=error,accepted=bool(np.isfinite(error) and error<=1e-4))
                if result['trace_parity']['accepted']:
                    result['dynamics']=plain(compress_trace(detail))
            return result
        s.task(f'layer-{layer}',run)


def context(s,p,tokens):
    import numpy as np
    # Eight contiguous 1024-token windows. Each context condition scores the
    # same final127 next-token targets, not different easy/hard positions.
    long=tokens['B'].reshape(8,1024)
    for i,length in enumerate((128,256,512,1024)):
        if i%s.args.workers != s.args.worker: continue
        def run(length=length):
            windows=long[:,-length:]
            def tail(value):
                a=np.asarray(value['token_nll'])[:,-127:]
                return dict(nll=float(a.mean()),windows=a.mean(1).tolist())
            base=tail(p.evaluate(windows)); teacher=tail(p.evaluate(windows,teacher=True))
            carry=tail(p.evaluate(windows,mode='reset',period=0,affected=p.cfg.mamba_layer_indices))
            error=float(np.max(np.abs(np.asarray(base['windows'])-carry['windows'])))
            result=dict(length=length,target_count=127,baseline=base,teacher=teacher,
                trace_control=carry,control_max_abs=error,accepted=error<=.01)
            if error<=.01:
                result['resets']={}
                for period in (32,64,128):
                    changed=tail(p.evaluate(windows,mode='reset',period=period,affected=p.cfg.mamba_layer_indices))
                    result['resets'][str(period)]=dict(**changed,
                        delta_nll=changed['nll']-carry['nll'], paired_delta=(np.asarray(changed['windows'])-carry['windows']).tolist())
            return result
        s.task(f'context-{length}',run)


def precision(s,p,tokens):
    import numpy as np
    groups={'fp32-all':None,'fp16-all':None,
            'fp32-early':tuple(j for j in p.cfg.mamba_layer_indices if j<8),
            'fp32-middle':tuple(j for j in p.cfg.mamba_layer_indices if 8<=j<20),
            'fp32-late':tuple(j for j in p.cfg.mamba_layer_indices if j>=20)}
    for i,(name,group) in enumerate(groups.items()):
        if i%s.args.workers != s.args.worker:continue
        def run(name=name,group=group):
            report={}
            for panel in ('A','B','C'):
                windows=tokens[panel][:8]
                base=p.evaluate(windows)
                try:
                    if group is None:
                        value=p.evaluate(windows,compute='float16' if name.startswith('fp16') else 'float32')
                    else:
                        value=p.evaluate(windows,precision_map={j:'float32' for j in group})
                    report[panel]=dict(nll=value['nll'],baseline=base['nll'],delta_nll=value['nll']-base['nll'],
                        paired_delta=(np.asarray(value['windows'])-base['windows']).tolist(),finite=True)
                except FloatingPointError:
                    report[panel]=dict(finite=False,reason='nonfinite FP16/FP32 forward; no NLL claim')
            return dict(condition=name,affected_layers='all' if group is None else list(group),panels=report,
                scope='projection/MLP compute dtype; stored BF16 weights, recurrence FP32, tied readout BF16')
        s.task(name,run)


def fit(s,p,tokens):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from extent.t4_search import RECIPES,LAYERS,initialize,materialize,fit_step
    from extent.plateau_forensics import norm,plain
    from extent.qwen3_teacher import Qwen3GQAAttention
    from extent.recovery_durability import RecoveryCheckpointStore
    from extent.chunked_checkpoint import digest
    store=RecoveryCheckpointStore(Path(s.args.cache_dir)/f'fits-{s.args.worker}',f'{s.prefix}/fits',s.hub)
    jobs=[(layer,recipe) for layer in LAYERS for recipe in RECIPES]
    caches={}
    for i,(layer,recipe) in enumerate(jobs):
        if (i+i//len(RECIPES))%s.args.workers!=s.args.worker:continue
        key=f'layer-{layer}/{recipe}'
        if key in s.result['tasks']:
            previous=s.result['fits'].get(key,{})
            if not previous.get('checkpoint_sync_error'):
                continue
            # Retry a locally complete candidate save without repeating fit.
            directory=store.directory(key)
            if (directory/'checkpoint.json').exists():
                meta=json.loads((directory/'checkpoint.json').read_text())
                needed=meta['payload_bytes']+(directory/'checkpoint.json').stat().st_size
                if needed+s.uploaded<=s.args.max_new_hf_gb*1e9 and store.sync(key,force=True,
                        deadline=s.args.deadline+s.args.save_reserve_minutes*60-120):
                    s.uploaded+=needed;previous['durable_step']=meta['step']
                    previous.pop('checkpoint_sync_error',None);s.persist()
            # Retain completed measurements even if the cloud destroyed the
            # local candidate. Report unavailability, never claim it durable.
            if previous.get('checkpoint_sync_error'):
                previous['candidate_available_locally']=(directory/'checkpoint.json').exists()
                s.persist()
            continue
        s.check()
        if layer not in caches:
            xs=[];ys=[];current=p.placed(p.current[f'layers_{layer}'])
            teacher=p.placed(p.teacher[f'layers_{layer}']['self_attn'])
            attention=Qwen3GQAAttention(p.tc)
            target=jax.jit(lambda x:attention.apply({'params':teacher},x,jnp.arange(x.shape[1])[None],None))
            for t in tokens['FIT']:
                s.check();x=p.forward_hidden(t[None],stop=layer)
                n=norm(x,current['input_layernorm'],p.cfg);y=jax.block_until_ready(target(n))
                xs.append(np.asarray(jax.device_get(n))[0]);ys.append(np.asarray(jax.device_get(y))[0])
            # HOST cache: never stack all recurrence activations on the GPU.
            caches[layer]=(np.stack(xs),np.stack(ys))
            # One layer's input/target cache at a time.
            caches={layer:caches[layer]}
        x,y=caches[layer];base=p.placed(p.current[f'layers_{layer}'])
        with jax.default_device(p.device):q=initialize(base,recipe)
        tx,step,loss=fit_step(p.cfg,recipe,p.columns);state=tx.init(q)
        cc=dict(s.result['contract'],layer=layer,recipe=recipe)
        saved=store.restore(key,cc,dict(parameters=q,optimizer=state))
        cursor=0
        if saved:
            payload,meta=saved;q=p.placed(payload['parameters']);state=p.placed(payload['optimizer']);cursor=meta['step']
        row=s.result['fits'].setdefault(key,dict(step=0,durable_step=0,history=[]))
        if row.get('durable_step',0)>cursor:
            raise FileNotFoundError('claimed durable fit checkpoint missing')
        if row['step']!=cursor:
            row.setdefault('rollbacks',[]).append(dict(reported=row['step'],restored=cursor))
            row['history']=[r for r in row['history'] if r['step']<=cursor];row['step']=cursor
        def sync():
            meta=store.save(key,cc,dict(parameters=q,optimizer=state),cursor)
            needed=meta['payload_bytes']+store.directory(key).joinpath('checkpoint.json').stat().st_size
            if needed+s.uploaded>s.args.max_new_hf_gb*1e9:
                row['checkpoint_sync_error']='upload cap';return
            if store.sync(key,force=True,deadline=s.args.deadline+s.args.save_reserve_minutes*60-120):
                s.uploaded+=needed;row['durable_step']=cursor;row.pop('checkpoint_sync_error',None)
            else:row['checkpoint_sync_error']='HF sync pending; local state retained'
            s.persist()
        try:
            if cursor==0:
                initial=materialize(base,q,recipe,p.columns)
                if any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree.leaves(initial),jax.tree.leaves(base))):
                    raise ValueError('local candidate zero-effect initialization differs')
            while cursor<1024:
                s.check();n=cursor%len(x)
                start=time.monotonic()
                proposed,after,health=step(q,state,base,p.placed(x[n:n+1]),p.placed(y[n:n+1]))
                health=plain(jax.block_until_ready(health))
                s.result['cost']['attempted_fit_updates']+=1
                s.result['cost']['fit_input_tokens']+=x.shape[1]
                s.result['cost']['fit_seconds']=s.result['cost'].get('fit_seconds',0)+time.monotonic()-start
                if not health['finite']:raise FloatingPointError('nonfinite fit proposal rejected')
                q,state=proposed,after;cursor+=1;row['step']=cursor
                if cursor%128==0:
                    row['history'].append(dict(step=cursor,**health))
                    store.save(key,cc,dict(parameters=q,optimizer=state),cursor)
                    s.persist()
            sync()
            candidate=jax.device_get(materialize(base,q,recipe,p.columns))
            report=dict(layer=layer,recipe=recipe,steps=cursor,training_windows=len(x),
                local_relative_mse=float(loss(q,base,p.placed(x[:1]),p.placed(y[:1]))),panels={})
            for panel in ('A','B','C'):
                s.check();baseline_value=p.evaluate(tokens[panel][:8])
                value=p.evaluate(tokens[panel][:8],overrides={layer:candidate})
                report['panels'][panel]=dict(nll=value['nll'],baseline=baseline_value['nll'],
                    delta_nll=value['nll']-baseline_value['nll'],
                    paired_delta=(np.asarray(value['windows'])-baseline_value['windows']).tolist())
            report['candidate_sha']=tree_hash(candidate)
            # A completed measurement is usable even if state upload is pending;
            # the explicit durable cursor distinguishes deployable candidates.
            s.task(key,lambda:report)
        except TimeoutError:
            sync();raise
        except FloatingPointError as exc:
            row['failed']=str(exc);sync()
            s.task(key,lambda:dict(layer=layer,recipe=recipe,failed=str(exc),accepted_steps=cursor))


def tree_hash(tree):
    from extent.efficiency_sources import tree_digest
    return tree_digest(tree)


def worker(args):
    import jax
    import numpy as np
    from extent.hf_artifact_sync import artifact_config_from_env
    from extent.t4_patient import Patient
    from extent.t4_data import load_data
    devices=jax.devices('gpu')
    if len(devices)!=1:raise ValueError('one visible GPU per worker required')
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF_TOKEN and EXTENT_HF_CHECKPOINT_REPO secrets required')
    s=Session(args,hub)
    try:
        if not s.persist(force=True):raise RuntimeError('HF metrics write preflight failed')
        tokens,meta=load_data(Path(args.cache_dir)/f'data-{args.worker}',s.check)
        s.result['data']=meta;s.persist()
        p=Patient(Path(args.cache_dir)/f'patient-{args.worker}',hub,devices[0],s.check)
        baseline(s,p,tokens);s.result['status']='running';s.persist()
        functions={116:profile,117:context,118:precision,119:fit}
        functions[args.experiment](s,p,tokens)
        s.result['status']='completed'
    except TimeoutError:
        s.result['status']='deadline_partial'
    except Exception as exc:
        s.result['status']='failed';s.result['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        s.result['pending_fit_states']=[key for key,row in s.result['fits'].items()
            if row.get('checkpoint_sync_error')]
        s.result['device_memory']=devices[0].memory_stats()
        s.persist(force=True)
        notify(args,f'Extent T4 EXP{args.experiment} GPU{args.worker}: {s.result["status"]}; tasks={len(s.result["tasks"])}; pending fits={len(s.result["pending_fit_states"])}; HF error={s.result.get("report_sync_error")}')
    if s.result['status']=='failed':raise RuntimeError('T4 worker failed; inspect saved JSON')
    return s.result


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=int,choices=EXPERIMENTS,required=True)
    p.add_argument('--max-wall-hours',type=float,default=11.5)
    p.add_argument('--save-reserve-minutes',type=float,default=45)
    p.add_argument('--max-new-hf-gb',type=float,default=2)
    p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--cache-dir',default='/kaggle/working/extent-t4-cache')
    p.add_argument('--workers',type=int,default=2,choices=(1,2))
    p.add_argument('--worker',type=int,default=None)
    p.add_argument('--deadline',type=float)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--telegram',action=argparse.BooleanOptionalAction,default=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    import math
    if not all(math.isfinite(v) and v>0 for v in (args.max_wall_hours,args.save_reserve_minutes,args.max_new_hf_gb)):
        raise ValueError('positive finite time/storage limits required')
    if args.max_wall_hours*60<=args.save_reserve_minutes:
        raise ValueError('save reserve consumes entire session')
    if args.plan_only:
        value=contract_for(args.experiment);print(json.dumps(value,indent=2));return value
    if args.worker is not None:
        if args.deadline is None or not 0<=args.worker<args.workers:raise ValueError('invalid worker dispatch')
        return worker(args)
    # Separate exec processes are required: never change visibility after JAX
    # initializes, and never fork a CUDA-initialized notebook kernel.
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    started=time.time();deadline=started+args.max_wall_hours*3600-args.save_reserve_minutes*60
    # Configure before imports used by notifications as well as child dispatch.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    notify(args,f'Extent T4 EXP{args.experiment}: started; workers={args.workers}; wall={args.max_wall_hours}h')
    children=[];handles=[];last_progress=0
    try:
        for index in range(args.workers):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(index)
            env['XLA_PYTHON_CLIENT_PREALLOCATE']='false';env['JAX_PLATFORMS']='cuda,cpu'
            env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
            env.pop('LD_LIBRARY_PATH',None)
            cmd=[sys.executable,'-u','-m','scripts.m3q_t4_campaign','--experiment',str(args.experiment),
                '--worker',str(index),'--workers',str(args.workers),'--deadline',str(deadline),
                '--max-wall-hours',str(args.max_wall_hours),'--save-reserve-minutes',str(args.save_reserve_minutes),
                '--max-new-hf-gb',str(args.max_new_hf_gb/args.workers),
                '--output-dir',str(output),'--cache-dir',args.cache_dir]
            if not args.telegram:cmd.append('--no-telegram')
            handle=(output/f'exp{args.experiment}-gpu{index}.log').open('a',encoding='utf-8')
            handles.append(handle);children.append(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT))
        while any(c.poll() is None for c in children):
            if time.time()-started>=args.max_wall_hours*3600:
                for c in children:
                    if c.poll() is None:c.terminate()
                raise TimeoutError('GPU wall limit; workers terminated, inspect durable cursors')
            if time.time()-last_progress>=60:
                rows=[]
                for index,c in enumerate(children):
                    file=output/f'extent-m3q-t4-{args.experiment}-worker-{index}.json'
                    value=json.loads(file.read_text()) if file.exists() else {}
                    rows.append(dict(gpu=index,status=value.get('status','loading'),tasks=len(value.get('tasks',{})),exit=c.poll()))
                print(f'EXP{args.experiment} elapsed={(time.time()-started)/3600:.2f}h {rows}',flush=True)
                last_progress=time.time()
            time.sleep(5)
        values=[]
        for index,c in enumerate(children):
            file=output/f'extent-m3q-t4-{args.experiment}-worker-{index}.json'
            value=json.loads(file.read_text()) if file.exists() else dict(status='failed',error='worker exited before report')
            values.append(dict(worker=index,exit=c.returncode,report=value))
        status='failed' if any(v['exit']!=0 or v['report']['status']=='failed' for v in values) else (
            'completed' if all(v['report']['status']=='completed' for v in values) else 'deadline_partial')
        result=dict(experiment=args.experiment,status=status,wall_hours=(time.time()-started)/3600,workers=values)
        path=output/f'extent-m3q-t4-{args.experiment}.json';write(path,result)
        write(path.with_name(path.stem+'-summary.txt'),dict(experiment=args.experiment,status=status,
            wall_hours=result['wall_hours'],workers=[dict(worker=v['worker'],status=v['report']['status'],
            tasks=len(v['report'].get('tasks',{})),error=v['report'].get('error')) for v in values]))
        print(json.dumps(dict(experiment=args.experiment,status=status,output=str(path)),indent=2),flush=True)
        pending=sum(len(v['report'].get('pending_fit_states',[])) for v in values)
        sync_errors=sum(bool(v['report'].get('report_sync_error')) for v in values)
        notify(args,f'Extent T4 EXP{args.experiment}: {status}; GPU workers stopped; pending fit states={pending}; report sync errors={sync_errors}; inspect Kaggle Output / HF worker prefixes')
        if status=='failed':raise RuntimeError('T4 campaign failed; inspect worker JSON and logs')
        return result
    except Exception as exc:
        notify(args,f'Extent T4 EXP{args.experiment}: runner stopped; {type(exc).__name__}; inspect GPU logs and durable checkpoints')
        raise
    finally:
        for c in children:
            if c.poll() is None:
                c.terminate()
                try:c.wait(timeout=15)
                except subprocess.TimeoutExpired:c.kill();c.wait()
        for h in handles:h.close()


if __name__=='__main__':main()
