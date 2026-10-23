"""EXP156: two independent inference workers, frozen hybrid vs original Qwen."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from extent.quality_benchmark_156 import (ROOT,TASKS,GENERATION_PROMPTS,contract,digest_json,
    pin,prepare_data,summarize_items,paired_results)
from extent.campaign_checkpoint import write_json_atomic

PREFIX='experiments/exp156-frozen-quality-v1'
STEM='extent-m3q-quality-156'


def empty_failed_preflight(prior,c):
    """Only a failed, completely unscored run may cross a code-only repair."""
    if prior.get('status')!='failed' or not prior.get('workers'):return False
    old={k:v for k,v in prior.get('contract',{}).items() if k!='code'}
    new={k:v for k,v in c.items() if k!='code'}
    if old!=new:return False
    return all(not row.get('items') and not row.get('nll') and not row.get('generations')
               and row.get('status') in ('failed','loading') for row in prior['workers'].values())


def readout(result):
    lines=['# EXP156 frozen capability assessment','',
           '| Task | Common questions / full set | Qwen % | Hybrid % | Hybrid minus Qwen, pp |',
           '|---|---:|---:|---:|---:|']
    for task,r in result.get('paired',{}).items():
        metric=result['contract']['primary_metric'][task];v=r[metric]
        planned=result.get('planned_counts',{}).get(task,'unknown')
        lines.append(f'| {task} ({metric}) | {r["n_common"]} / {planned} | {100*v["teacher"]:.2f} | {100*v["hybrid"]:.2f} | {100*v["hybrid_minus_teacher"]:+.2f} |')
    lines.extend(['','Partial coverage and confidence intervals are in the JSON. Zero-shot evaluation; not official few-shot scores.',
                  'Generated code is not executed; decoded completions are qualitative evidence.'])
    return '\n'.join(lines)


def teacher_reference_check(engine,tokenizer,cache):
    """Validate the formula in FP32; record BF16 backend drift separately.

    A CPU BF16 vs GPU BF16 comparison changes both rounding/fusion and backend.
    It is not a tight implementation-equivalence gate. The actual benchmark
    remains source BF16. No tolerance is relaxed for hybrid checkpoint replay.
    """
    import gc
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from extent.quality_benchmark_156 import InferenceEngine
    torch.set_num_threads(2)
    model=AutoModelForCausalLM.from_pretrained(Path(cache)/'patient/qwen',torch_dtype=torch.float32,
        attn_implementation='eager',trust_remote_code=False).cpu().eval()
    texts=('The capital of France is Paris.','Вода замерзает при низкой температуре.',
           'Question: Which object conducts electricity?\nAnswer: copper',
           '2 + 3 = 5. Multiplying by two gives 10.',
           'def add(a, b):\n    return a + b')
    probes=[(text,tokenizer.encode(text,add_special_tokens=False)[:32]) for text in texts]
    rows=[];bf16=[];bridge=None
    def reference(ids):
        tokens=torch.tensor([ids],dtype=torch.long)
        with torch.inference_mode():
            logits=model(tokens,use_cache=False).logits[:,:-1].float()
            return -torch.log_softmax(logits,-1).gather(-1,tokens[:,1:,None])[...,0].cpu().numpy()
    try:
        # Short probes only; the temporary FP32 parameters are freed before any
        # panel or task evaluation. No full-size FP32 training state exists.
        base=jax.tree.map(lambda value:value.astype(jnp.float32),engine.base)
        bridge=InferenceEngine(replace(engine.cfg,param_dtype='float32',compute_dtype='float32'),
            replace(engine.tc,param_dtype='float32',compute_dtype='float32'),base,{},0,{},'teacher',engine.check)
        del base
        for text,ids in probes:
            engine.check();expected=reference(ids);actual=bridge.token_nll(np.asarray([ids],np.int32))
            error=float(np.max(np.abs(actual-expected)))
            rows.append(dict(text=text,tokens=len(ids),max_abs_token_nll_error=error,
                jax_token_nll=actual[0].tolist(),transformers_token_nll=expected[0].tolist(),
                passed=bool(np.isfinite(error) and error<=.003)))
        del bridge;bridge=None;gc.collect()
        model.to(dtype=torch.bfloat16)
        for text,ids in probes:
            engine.check();expected=reference(ids);actual=engine.token_nll(np.asarray([ids],np.int32))
            error=float(np.max(np.abs(actual-expected)))
            if not np.isfinite(error):raise FloatingPointError('nonfinite BF16 reference probe')
            bf16.append(dict(text=text,tokens=len(ids),max_abs_token_nll_error=error,
                jax_token_nll=actual[0].tolist(),transformers_token_nll=expected[0].tolist()))
    finally:
        del bridge,model;gc.collect()
    return dict(passed=all(r['passed'] for r in rows),tolerance=.003,rows=rows,
                bf16_backend_diagnostic=dict(rows=bf16,used_as_equivalence_gate=False,
                    scope='CPU Transformers BF16 vs GPU JAX BF16; rounding/fusion plus backend differ; not proven to be rounding alone'),
                torch_version=str(torch.__version__),transformers_version=transformers.__version__,
                gate_dtype='float32',benchmark_dtype='bfloat16',attention_implementation='eager',
                scope='FP32 formula/mapping check on actual weights, CPU Transformers vs GPU JAX; BF16 discrepancy retained, not silently accepted as identity')


def worker(args):
    import jax
    from importlib.metadata import version
    from transformers import AutoTokenizer
    from extent.hf_artifact_sync import artifact_config_from_env
    from extent.quality_benchmark_156 import load_engine
    from extent.qwen_source import QWEN3_1_7B_BASE
    from extent.t4_scale_data import load_data
    from extent.rank_growth_153 import replay_report
    role=args.worker;output=Path(args.output_dir);cache=Path(args.cache_dir)
    path=output/f'{STEM}-{role}.json';c=contract()
    r=json.loads(path.read_text()) if path.exists() else dict(contract=c,role=role,items={},nll={},generations={},sessions=[])
    if r['contract']!=c or r['role']!=role:raise ValueError('benchmark resume contract differs')
    journal=output/f'{STEM}-{role}-items.jsonl'
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            try:value=json.loads(line)
            except ValueError:continue  # interrupted final line, never a scored item
            if value['contract_sha256']!=digest_json(c):raise ValueError('item journal contract differs')
            r['items'][value['item']['id']]=value['item']
    if r.get('status')=='completed':return r
    started=time.monotonic();r.update(status='loading');r.pop('error',None)
    invocation=dict(started_unix=time.time(),work_deadline_unix=args.deadline_unix)
    r['sessions'].append(invocation);last_save=0.
    def save(force=False):
        nonlocal last_save
        if not force and time.monotonic()-last_save<60:return
        r['summary']=summarize_items(r['items']);invocation['seconds']=time.monotonic()-started
        write_json_atomic(path,r);last_save=time.monotonic()
    def check():
        if time.time()>=args.deadline_unix:raise TimeoutError('registered benchmark work deadline')
    save(True)
    try:
        devices=jax.devices('gpu')
        if len(devices)!=1 or jax.process_count()!=1:raise ValueError('one visible GPU per benchmark worker required')
        if r.get('device_kind',devices[0].device_kind)!=devices[0].device_kind:raise ValueError('do not mix evaluation hardware within a resumed role')
        r['device']=str(devices[0]);r['device_kind']=devices[0].device_kind;r['runtime']={k:version(k) for k in ('jax','jaxlib','flax','optax')}
        r['runtime'].update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
        if r['runtime']!=pin()['runtime']:raise ValueError('source math/runtime differs')
        hub=artifact_config_from_env()
        if hub is None:raise ValueError('HF secrets required')
        tokenizer=AutoTokenizer.from_pretrained(QWEN3_1_7B_BASE.repo_id,revision=QWEN3_1_7B_BASE.revision,cache_dir=cache/'tokenizer',trust_remote_code=False)
        engine=load_engine(role,cache,hub,check)
        r['loading_seconds']=time.monotonic()-started
        if role=='teacher':
            r['transformers_reference']=teacher_reference_check(engine,tokenizer,cache);save(True)
            if not r['transformers_reference']['passed']:raise ValueError('original-Qwen FP32 formula reference differs; refuse benchmark')
        questions=json.loads((cache/'benchmark-data/questions.json').read_text())
        manifest=json.loads((cache/'benchmark-data/data-manifest.json').read_text())
        if digest_json(questions)!=manifest['questions_sha256']:raise ValueError('question cache SHA differs')
        if r.get('data_manifest',manifest)!=manifest:raise ValueError('resumed benchmark data manifest changed')
        r['data_manifest']=manifest
        panels=dict(np.load(cache/'benchmark-data/fresh-panels.npz',allow_pickle=False))
        for key,value in panels.items():
            if hashlib.sha256(value.tobytes()).hexdigest()!=manifest['provenance'][key]['token_sha256']:raise ValueError('fresh panel SHA differs')
        def evaluate(name,windows):
            state=r['nll'].setdefault(name,dict(windows=[],expected=len(windows),targets_per_window=int(windows.shape[1]-1)))
            for t in windows[len(state['windows']):]:
                check();loss=engine.token_nll(t[None])
                if not np.all(np.isfinite(loss)):raise FloatingPointError('nonfinite panel NLL')
                state['windows'].append(float(loss.mean()));save()
            state['nll']=float(np.mean(state['windows']));save(True);return state
        # Replay the measured CURRENT, not the inherited historical BEST.
        old,meta=load_data(cache/'source-panels',check)
        if json.loads(json.dumps(meta))!=pin()['data']:raise ValueError('source replay panel identity differs')
        # Recompute the gate on each invocation, including an own-results resume.
        for p in 'ABC':r['nll'].pop('source_'+p,None)
        controls={p:evaluate('source_'+p,old[p]) for p in 'ABC'}
        if role=='hybrid':
            r['source_replay']=replay_report(controls,pin());save(True)
            if not r['source_replay']['passed']:raise ValueError('frozen CURRENT NLL replay failed; no benchmarks accepted')
        else:r['source_replay']=dict(passed=True,scope='Original Qwen measured on source panels; not hybrid-reference replay')
        r['status']='evaluating';save(True)
        for name,arr in panels.items():
            evaluate(name,arr[:32768].reshape(128,256))
            context=arr[32768:].reshape(8,1024)
            for length in (128,256,512,1024):
                key=f'{name}_context_{length}';state=r['nll'].setdefault(key,dict(windows=[],expected=8,targets_per_window=127))
                for t in context[len(state['windows']):]:
                    check();v=engine.token_nll(t[None,-length:])[:,-127:]
                    if not np.all(np.isfinite(v)):raise FloatingPointError('nonfinite tail NLL')
                    state['windows'].append(float(v.mean()));save()
                state['nll']=float(np.mean(state['windows']));save(True)
        def score(doc):
            if doc['id'] in r['items']:return
            check();begun=time.monotonic();row={k:doc[k] for k in ('id','task','gold','subject')}
            row['prompt_sha256']=digest_json(doc['pairs'])
            try:
                scores=engine.score_pairs(doc['pairs'],tokenizer,max_context=c['max_context'],batch_size=c['choice_batch'])
                ll=np.array([s['loglikelihood'] for s in scores]);norm=ll/np.asarray(doc['norms'])
                row.update(status='scored',scores=scores,normalization_lengths=doc['norms'],
                    prediction=int(ll.argmax()),prediction_norm=int(norm.argmax()),
                    acc=bool(ll.argmax()==doc['gold']),acc_norm=bool(norm.argmax()==doc['gold']))
            except OverflowError as exc:row.update(status='skipped_context',reason=str(exc))
            row['seconds']=time.monotonic()-begun;r['items'][doc['id']]=row
            # Append-only journal protects work between atomic snapshots.
            with journal.open('a',encoding='utf-8') as f:f.write('\n'+json.dumps(dict(contract_sha256=digest_json(c),item=row),ensure_ascii=False)+'\n')
            save()
        pilot=questions[:len(TASKS)*25]
        begun=time.monotonic();before=len(r['items'])
        for doc in pilot:score(doc)
        if len(r['items'])>before:
            r['pilot']=dict(items=len(r['items'])-before,seconds=time.monotonic()-begun,
                            caveat='new items in pilot this invocation; includes first choice-shape compilation, not a completion guarantee')
        save(True)
        for i,(language,prompt) in enumerate(GENERATION_PROMPTS):
            for mode in ('greedy','sample') if i%2==0 else ('greedy',):
                key=f'{i}:{mode}'
                if key not in r['generations']:
                    check();r['generations'][key]=dict(language=language,**engine.generate(tokenizer,prompt,mode,i))
                    save(True)
        for doc in questions[len(pilot):]:score(doc)
        r['status']='completed' if all(x['status']=='scored' for x in r['items'].values()) else 'completed_with_skips'
    except TimeoutError as exc:r.update(status='deadline_partial',stop_reason=str(exc))
    except Exception as exc:
        r.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
    finally:
        r['memory_fallbacks']=getattr(locals().get('engine'),'fallbacks',[]);save(True)
    if r['status']=='failed':raise RuntimeError(f'{role} benchmark failed; original traceback saved')
    return r


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-wall-hours',type=float,default=11.5)
    p.add_argument('--save-reserve-minutes',type=float,default=30.)
    p.add_argument('--max-new-hf-gb',type=float,default=1.)
    p.add_argument('--worker',choices=('teacher','hybrid'))
    p.add_argument('--deadline-unix',type=float)
    p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--cache-dir',default='/kaggle/working/extent-quality-cache')
    p.add_argument('--no-telegram',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args(argv)
    if args.plan_only:value=contract();print(json.dumps(value,indent=2));return value
    if args.worker:
        if args.deadline_unix is None:raise ValueError('worker requires parent deadline')
        return worker(args)
    if not (np.isfinite(args.max_wall_hours) and 1<=args.max_wall_hours<=12 and
            15<=args.save_reserve_minutes<args.max_wall_hours*60 and np.isfinite(args.max_new_hf_gb) and args.max_new_hf_gb>0):raise ValueError('invalid benchmark budget')
    started=time.time();work_deadline=started+args.max_wall_hours*3600-args.save_reserve_minutes*60
    full_deadline=started+args.max_wall_hours*3600;output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    cache=Path(args.cache_dir);cache.mkdir(parents=True,exist_ok=True);c=contract()
    # Parent may import shared notification helpers; prevent it claiming a GPU.
    os.environ['JAX_PLATFORMS']='cpu'
    from extent.reset_quiet import ParentNotifier
    from extent.hf_artifact_sync import artifact_config_from_env,restore_artifact,upload_artifacts_together
    notice=ParentNotifier(not args.no_telegram,156);hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF secrets required')
    previous=output/f'{STEM}.json'
    if not previous.exists():restore_artifact(previous,PREFIX+'/latest.json',hub)
    prior=json.loads(previous.read_text()) if previous.exists() else {}
    retired_preflight=None
    if prior and prior['contract']!=c:
        if not empty_failed_preflight(prior,c):raise ValueError('benchmark contract differs; do not mix runs')
        # This known startup failure scored zero questions and no NLL panels.
        # Preserve it, then start a clean repaired evaluation of the SAME weights.
        for role in ('teacher','hybrid'):
            local=output/f'{STEM}-{role}.json';journal=output/f'{STEM}-{role}-items.jsonl'
            if journal.exists() and journal.stat().st_size:raise ValueError('nonempty old item journal; explicit separate benchmark namespace required')
            if local.exists():
                row=json.loads(local.read_text())
                if row.get('contract')!=prior['contract'] or any(row.get(k) for k in ('items','nll','generations')):
                    raise ValueError('local scored worker differs; refuse preflight-only recovery')
        archive=output/f'{STEM}-failed-preflight-{digest_json(prior["contract"])[:12]}.json'
        write_json_atomic(archive,prior)
        retired_preflight=dict(path=str(archive),old_contract=prior['contract'],
            errors={role:row.get('error') for role,row in prior['workers'].items()},
            reason='code-only startup repair; zero scored items, panels or generations')
        for role in ('teacher','hybrid'):
            local=output/f'{STEM}-{role}.json'
            if local.exists():local.unlink()
        print('EXP156: archived empty failed preflight; restarting repaired reference check',flush=True)
        prior={}
    for role,r in prior.get('workers',{}).items():write_json_atomic(output/f'{STEM}-{role}.json',r)
    def check():
        if time.time()>=work_deadline:raise TimeoutError('registered benchmark preparation deadline')
    children=[];handles=[];uploaded=0;sync_error=None;last_sync=0.
    result=dict(contract=c,workers={},status='preparing',sessions=prior.get('sessions',[]),notifications=[])
    if retired_preflight:result['retired_failed_preflight']=retired_preflight
    elif prior.get('retired_failed_preflight'):result['retired_failed_preflight']=prior['retired_failed_preflight']
    invocation=dict(started_unix=started,max_wall_hours=args.max_wall_hours,save_reserve_minutes=args.save_reserve_minutes)
    result['sessions'].append(invocation)
    notice.emit('started','frozen CURRENT80260 vs original Qwen; two GPUs; no training')
    def collect(final=False):
        nonlocal result
        rows={}
        for role in ('teacher','hybrid'):
            path=output/f'{STEM}-{role}.json'
            if path.exists():rows[role]=json.loads(path.read_text())
        result['workers']=rows;result['notifications']=notice.events
        invocation['elapsed_hours']=(time.time()-started)/3600
        if final and len(rows)==2:
            result['paired']=paired_results(rows['teacher']['items'],rows['hybrid']['items'])
            nll={};rng=np.random.default_rng(156123)
            for name in sorted(rows['teacher'].get('nll',{}).keys()&rows['hybrid'].get('nll',{}).keys()):
                a=rows['teacher']['nll'][name];b=rows['hybrid']['nll'][name]
                if len(a['windows'])!=a['expected'] or len(b['windows'])!=b['expected']:continue
                if a['expected']!=b['expected'] or a['targets_per_window']!=b['targets_per_window']:raise ValueError('paired NLL windows differ')
                delta=np.asarray(b['windows'])-np.asarray(a['windows'])
                boot=[float(rng.choice(delta,len(delta),replace=True).mean()) for _ in range(2000)]
                nll[name]=dict(windows=len(delta),targets_per_window=a['targets_per_window'],teacher=float(np.mean(a['windows'])),
                    hybrid=float(np.mean(b['windows'])),hybrid_minus_teacher=float(delta.mean()),
                    paired_window_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())
            result['paired_nll']=nll
            result['summary_markdown']=readout(result)
        write_json_atomic(previous,result)
    def sync(force=False):
        nonlocal last_sync,uploaded,sync_error
        if not force and time.time()-last_sync<3600:return
        collect();size=previous.stat().st_size
        if uploaded+size>args.max_new_hf_gb*1e9:
            sync_error='result upload cap reached; full local output retained'
            result['report_sync_error']=sync_error;write_json_atomic(previous,result);return
        try:
            upload_artifacts_together([(previous,PREFIX+'/latest.json')],hub,commit_message='EXP156 frozen quality results')
            uploaded+=size;sync_error=None
        except Exception as exc:
            sync_error=dict(type=type(exc).__name__,message=str(exc)[:300])
        result['report_sync_error']=sync_error;result['uploaded_bytes']=uploaded
        write_json_atomic(previous,result)
        last_sync=time.time()
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
        from extent.qwen_source import QWEN3_1_7B_BASE
        tok=AutoTokenizer.from_pretrained(QWEN3_1_7B_BASE.repo_id,revision=QWEN3_1_7B_BASE.revision,cache_dir=cache/'tokenizer',trust_remote_code=False)
        questions,panels,manifest=prepare_data(cache/'benchmark-data',tok,check)
        result['data_manifest']=manifest;result['planned_counts']=manifest['counts']
        print('EXP156 planned questions per model:',manifest['counts'],flush=True)
        # Fetch the common model once before both readers start, no download races.
        for file in ('config.json','model.safetensors'):
            check();hf_hub_download(QWEN3_1_7B_BASE.repo_id,file,revision=QWEN3_1_7B_BASE.revision,local_dir=cache/'patient/qwen')
        sync(True)
        result['status']='running';collect()
        for gpu,role in enumerate(('teacher','hybrid')):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
            cmd=[sys.executable,'-u','-m','scripts.m3q_quality_benchmark_156','--worker',role,'--deadline-unix',str(work_deadline),
                 '--output-dir',str(output),'--cache-dir',str(cache),'--no-telegram']
            handle=(output/f'{STEM}-{role}.log').open('w',encoding='utf-8');handles.append(handle)
            children.append((role,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)))
        last_collect=0.
        while any(child.poll() is None for role,child in children):
            if time.time()-last_collect>=60:
                collect();sync();last_collect=time.time()
                if any(child.poll() not in (None,0) for role,child in children) or any(v.get('status')=='failed' for v in result['workers'].values()):
                    notice.emit('failure','worker failed; stopping peer; original traceback saved')
                    result['peer_stopped']='fatal worker error; stop paired run instead of consuming remaining quota'
                    for role,child in children:
                        if child.poll() is None:child.terminate()
                    break
                notice.emit('progress',str({k:dict(status=v.get('status'),questions=len(v.get('items',{})),generations=len(v.get('generations',{}))) for k,v in result['workers'].items()}))
            if time.time()>full_deadline-120:
                for role,child in children:
                    if child.poll() is None:child.terminate()
                result['forced_stop']='save deadline; latest atomic worker snapshots retained';break
            time.sleep(5)
        for role,child in children:child.wait()
        collect();statuses=[r.get('status') for r in result['workers'].values()]
        failed=any(child.returncode!=0 for role,child in children)
        result['status']='failed' if failed or 'failed' in statuses else 'completed' if statuses==['completed','completed'] else 'completed_with_skips' if all(s in ('completed','completed_with_skips') for s in statuses) else 'deadline_partial'
        if failed:
            result['worker_log_tails']={role:(output/f'{STEM}-{role}.log').read_text(encoding='utf-8',errors='replace')[-20000:] for role,child in children if child.returncode!=0}
    except TimeoutError as exc:result.update(status='deadline_partial',stop_reason=str(exc))
    except Exception as exc:
        result.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
    finally:
        for role,child in children:
            if child.poll() is None:child.terminate();child.wait()
        for handle in handles:handle.close()
        collect(True);sync(True)
        result['report_sync_error']=sync_error;result['uploaded_bytes']=uploaded
        notice.emit('failure' if result['status']=='failed' else 'finished',result['status']+'; download '+STEM+'.json')
        result['notifications']=notice.events;write_json_atomic(previous,result)
        # Publish final status/notification receipt once more, still within cap.
        sync(True)
        print(result.get('summary_markdown','No paired benchmark items yet.'),flush=True)
        if result['status']=='failed':
            # Kaggle's notebook log must show the root cause, not just a generic
            # parent RuntimeError while the real traceback sits only in JSON.
            for role,row in result.get('workers',{}).items():
                if row.get('error'):print(f'EXP156 {role} original error:\n'+row['error'].get('traceback',str(row['error'])),flush=True)
            for role,tail in result.get('worker_log_tails',{}).items():
                if tail:print(f'EXP156 {role} log tail:\n'+tail,flush=True)
    if result['status']=='failed':raise RuntimeError('EXP156 evaluation failed; aggregate includes original worker traceback')
    return result


if __name__=='__main__':main()
