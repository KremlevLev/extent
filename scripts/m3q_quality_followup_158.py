"""EXP158: paired frozen155 checkpoints80260/118304 on the same fresh data."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from extent.quality_benchmark_156 import ROOT,digest_json,prepare_data,pin as old_pin,summarize_items,paired_results
from extent.campaign_checkpoint import write_json_atomic
PREFIX='experiments/exp158-quality-followup-v1'
STEM='extent-m3q-quality-158'

def new_pin():return json.loads((ROOT/'results/EXP-157-source-pin.json').read_text())

def contract():
    return dict(protocol='exp158-quality-followup-v1',source_steps=[80260,118304],source_pins=[digest_json(old_pin()),digest_json(new_pin())],
        questions_per_task=256,fresh_windows=128,context_lengths=[128,256,512,1024],fewshot=0,
        primary_metric={k:('acc' if k in ('mmlu','winogrande') else 'acc_norm') for k in ('arc_easy','arc_challenge','piqa','winogrande','hellaswag','mmlu')},
        code_sha=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        engine_sha=hashlib.sha256((ROOT/'extent/quality_benchmark_156.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        data_pin_sha=digest_json(json.loads((ROOT/'results/EXP-156-data-pin.json').read_text())),
        scope='paired checkpoint diagnostic; same156 question order and data; limited subset; no official benchmark claim')

def comparisons(workers):
    old=workers.get('old',{});new=workers.get('new',{})
    paired=paired_results(old.get('items',{}),new.get('items',{}))
    for task,row in paired.items():
        for metric in ('acc','acc_norm'):
            v=row[metric];v['old']=v.pop('teacher');v['new']=v.pop('hybrid');v['new_minus_old']=v.pop('hybrid_minus_teacher')
    nll={};rng=np.random.default_rng(158)
    for name,a in old.get('nll',{}).items():
        b=new.get('nll',{}).get(name,{})
        # Only compare complete identical window sets, never partially read prefixes.
        if len(a.get('windows',[]))!=a.get('expected') or len(b.get('windows',[]))!=a.get('expected'):continue
        av=np.asarray(a['windows']);bv=np.asarray(b['windows']);delta=bv-av
        boot=[rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(2000)]
        nll[name]=dict(windows=len(delta),old=float(av.mean()),new=float(bv.mean()),new_minus_old=float(delta.mean()),paired_window_ci95=np.quantile(boot,[.025,.975]).tolist())
    return paired,nll

def worker(a):
    import jax
    from importlib.metadata import version
    from transformers import AutoTokenizer
    from extent.hf_artifact_sync import artifact_config_from_env
    import extent.quality_benchmark_156 as quality
    from extent.qwen_source import QWEN3_1_7B_BASE
    from extent.t4_scale_data import load_data
    from extent.rank_growth_153 import replay_report
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True);cache=Path(a.cache_dir)/a.worker
    path=output/f'{STEM}-{a.worker}.json';c=contract();p=old_pin() if a.worker=='old' else new_pin()
    r=json.loads(path.read_text()) if path.exists() else dict(contract=c,role=a.worker,step=p['source_step'],nll={},items={},generations={})
    if r['contract']!=c:raise ValueError('own resume contract differs')
    def save():write_json_atomic(path,r)
    def check():
        if time.time()>=a.deadline_unix:raise TimeoutError('registered evaluation deadline')
    if r.get('status')=='completed':return r
    try:
        r['status']='loading';save()
        if len(jax.devices('gpu'))!=1:raise ValueError('one visible GPU per process')
        runtime={k:version(k) for k in ('jax','jaxlib','flax','optax')};runtime.update(matmul_precision=jax.config.jax_default_matmul_precision,prng_impl=jax.config.jax_default_prng_impl)
        if runtime!=p['runtime']:raise ValueError('source runtime differs')
        hub=artifact_config_from_env()
        if hub is None:raise ValueError('HF configuration required')
        tok=AutoTokenizer.from_pretrained(QWEN3_1_7B_BASE.repo_id,revision=QWEN3_1_7B_BASE.revision,cache_dir=cache/'tokenizer')
        prepare_data(cache/'benchmark-data',tok,check)
        # The legacy loader is parameterized only within this single-worker process.
        original=quality.pin;quality.pin=lambda:p
        try:engine=quality.load_engine('hybrid',cache,hub,check)
        finally:quality.pin=original
        old,meta=load_data(cache/'source-panels',check)
        if json.loads(json.dumps(meta))!=p['data']:raise ValueError('source data differs')
        def evaluate(name,windows,tail=None):
            row=r['nll'].setdefault(name,dict(windows=[],targets_per_window=255 if tail is None else tail,expected=len(windows)))
            for t in windows[len(row['windows']):]:
                check();loss=engine.token_nll(t[None]);loss=loss if tail is None else loss[:,-tail:]
                if not np.isfinite(loss).all():raise FloatingPointError('nonfinite NLL')
                row['windows'].append(float(loss.mean()));save()
            row['nll']=float(np.mean(row['windows']));save();return row
        for panel in 'ABC':r['nll'].pop('source_'+panel,None)
        controls={k:evaluate('source_'+k,old[k]) for k in 'ABC'}
        r['source_replay']=replay_report(controls,p);save()
        if not r['source_replay']['passed']:raise ValueError('source per-window replay failed')
        data=cache/'benchmark-data';manifest=json.loads((data/'data-manifest.json').read_text());r['data_manifest']=manifest
        panels=dict(np.load(data/'fresh-panels.npz',allow_pickle=False))
        for name,arr in panels.items():
            if hashlib.sha256(arr.tobytes()).hexdigest()!=manifest['provenance'][name]['token_sha256']:raise ValueError('fresh panel SHA differs')
            evaluate(name,arr[:32768].reshape(128,256))
            ctx=arr[32768:].reshape(8,1024)
            for length in c['context_lengths']:evaluate(name+'_context_'+str(length),ctx[:,-length:],127)
        questions=json.loads((data/'questions.json').read_text())
        if digest_json(questions)!=manifest['questions_sha256']:raise ValueError('question SHA differs')
        counts={}
        for doc in questions:
            task=doc['task'];counts[task]=counts.get(task,0)+1
            if counts[task]>c['questions_per_task'] or doc['id'] in r['items']:continue
            check();scores=engine.score_pairs(doc['pairs'],tok,max_context=2048,batch_size=2)
            ll=np.array([s['loglikelihood'] for s in scores]);norm=ll/np.asarray(doc['norms'])
            r['items'][doc['id']]=dict(id=doc['id'],task=task,gold=doc['gold'],subject=doc['subject'],status='scored',
                prompt_sha256=digest_json(doc['pairs']),acc=bool(ll.argmax()==doc['gold']),acc_norm=bool(norm.argmax()==doc['gold']),scores=scores)
            if len(r['items'])%16==0:save()
        for i,(lang,prompt) in enumerate(quality.GENERATION_PROMPTS):
            key=str(i)
            if key not in r['generations']:
                check();r['generations'][key]=dict(language=lang,**engine.generate(tok,prompt,'greedy',i));save()
        r['status']='completed'
    except TimeoutError as exc:r.update(status='deadline_partial',stop_reason=str(exc))
    except Exception as exc:r.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));traceback.print_exc()
    finally:r['summary']=summarize_items(r['items']);save()
    return r

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=['old','new']);p.add_argument('--deadline-unix',type=float)
    p.add_argument('--max-wall-hours',type=float,default=11.5);p.add_argument('--save-reserve-minutes',type=float,default=18)
    p.add_argument('--output-dir',default='/kaggle/working/extent-results');p.add_argument('--cache-dir',default='/kaggle/working/extent-quality-158-cache')
    p.add_argument('--no-telegram',action='store_true');p.add_argument('--plan-only',action='store_true');a=p.parse_args(argv)
    if a.plan_only:return contract()
    if a.worker:
        if a.deadline_unix is None:raise ValueError('worker deadline required')
        return worker(a)
    if not (0<a.max_wall_hours<=12 and 18<=a.save_reserve_minutes<a.max_wall_hours*60):raise ValueError('invalid budget')
    from extent.hf_artifact_sync import artifact_config_from_env,upload_artifact,restore_artifact
    from extent.reset_quiet import ParentNotifier
    hub=artifact_config_from_env()
    if hub is None:raise ValueError('HF configuration required')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);path=out/(STEM+'.json');c=contract()
    if not path.exists():restore_artifact(path,PREFIX+'/latest.json',hub)
    prior=json.loads(path.read_text()) if path.exists() else {'contract':c,'workers':{}}
    if prior['contract']!=c:raise ValueError('parent resume contract differs')
    for role,w in prior['workers'].items():write_json_atomic(out/f'{STEM}-{role}.json',w)
    deadline=time.time()+a.max_wall_hours*3600-a.save_reserve_minutes*60
    notice=ParentNotifier(not a.no_telegram,158);notice.emit('started','frozen80260 vs118304; fresh EN/RU +256 questions/task')
    processes=[];next_sync=0;r=prior
    try:
        for gpu,role in enumerate(('old','new')):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            log=open(out/f'exp158-{role}.log','a',encoding='utf-8')
            cmd=[sys.executable,'-m','scripts.m3q_quality_followup_158','--worker',role,'--deadline-unix',str(deadline),'--output-dir',str(out),'--cache-dir',a.cache_dir,'--no-telegram']
            processes.append((role,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        while True:
            r={'contract':c,'workers':{}}
            for role,proc,log in processes:
                f=out/f'{STEM}-{role}.json'
                if f.exists():r['workers'][role]=json.loads(f.read_text())
            r['paired'],r['paired_nll']=comparisons(r['workers'])
            r['status']='failed' if any(w.get('status')=='failed' for w in r['workers'].values()) else 'completed' if len(r['workers'])==2 and all(w.get('status')=='completed' for w in r['workers'].values()) else 'deadline_partial'
            write_json_atomic(path,r)
            if time.time()>=next_sync or all(proc.poll() is not None for _,proc,_ in processes):
                try:upload_artifact(path,PREFIX+'/latest.json',hub,commit_message='EXP158 paired checkpoint diagnostic');next_sync=time.time()+3600;r['report_sync_error']=None
                except Exception as exc:
                    response=getattr(exc,'response',None);headers=getattr(response,'headers',{})
                    try:delay=max(120.,float(headers.get('Retry-After',120)))
                    except (ValueError,TypeError):delay=120.
                    next_sync=time.time()+delay;r['report_sync_error']=dict(type=type(exc).__name__,retry_after_seconds=delay)
                    print('report upload retry pending',type(exc).__name__,flush=True)
            if r['status']=='failed' or time.time()>deadline+30 or all(proc.poll() is not None for _,proc,_ in processes):break
            if any(proc.poll() not in (None,0) for _,proc,_ in processes):r['status']='failed';write_json_atomic(path,r);break
            time.sleep(10)
    finally:
        for _,proc,log in processes:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
            log.close()
        # Final worker snapshots are read after termination; preserve the latest saved work.
        for role,_,_ in processes:
            f=out/f'{STEM}-{role}.json'
            if f.exists():r['workers'][role]=json.loads(f.read_text())
        r['paired'],r['paired_nll']=comparisons(r['workers'])
        if r['status']=='failed':
            for role,_,_ in processes:
                logpath=out/f'exp158-{role}.log'
                if logpath.exists():print(role+' worker log tail:\n'+logpath.read_text(encoding='utf-8',errors='replace')[-4000:],flush=True)
        write_json_atomic(path,r)
        try:upload_artifact(path,PREFIX+'/latest.json',hub,commit_message='EXP158 final report');r['report_sync_error']=None
        except Exception as exc:r['report_sync_error']=str(exc)
        notice.emit('finished',r['status']+'; download '+path.name);write_json_atomic(path,r)
    if r['status']=='failed':raise RuntimeError('EXP158 failed; aggregate and worker logs saved')
    return r
if __name__=='__main__':main()
