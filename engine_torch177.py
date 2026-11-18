"""EXP177 matched CE versus CE+KL, isolated workers and recoverable checkpoints."""
import argparse,hashlib,json,math,os,shutil,subprocess,sys,time,traceback
from datetime import datetime
from pathlib import Path
import extent
_extra=Path(__file__).resolve().parent/'extent'
if str(_extra) not in extent.__path__:extent.__path__.append(str(_extra))

PREFIX='experiments/exp177-ce-kd-v1'
SOURCE_PREFIX='experiments/exp176-paired-precision-v1/fp32'
SOURCE_SHA='d62339838b407ffbc332e4d96df699aa7810a980110df3b84631359945acc71b'
TEACHER='Qwen/Qwen3-1.7B-Base'
REVISION='ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
EFFECTIVE=16;LENGTH=1024;INPUTS=EFFECTIVE*LENGTH

class FastPause(Exception):
    pass


def code_identity():
    from extent import torch_distill_177,torch_continue_176,torch_clean_170
    from extent.broad_corpus import digest_file
    return {str(Path(x).name):digest_file(Path(x)) for x in (__file__,torch_distill_177.__file__,torch_continue_176.__file__,torch_clean_170.__file__)}


def choose_budget(remaining, ce_seconds, kd_seconds, reserve=1800, safety=.8):
    from extent.torch_distill_177 import matched_token_budget
    return matched_token_budget(max(0,remaining-reserve)*safety,ce_seconds,kd_seconds)


def validate_branch_state(saved, descriptor, actual_sha, arm, target_updates, expected_code):
    """Reject a wrong, corrupted, mismatched, or cursor-inconsistent resume."""
    contract=saved['contract']
    if actual_sha!=descriptor['sha256'] or saved['step']!=descriptor['step'] or saved['cursor']!=descriptor['cursor']:
        raise ValueError('branch descriptor differs')
    if contract.get('exp177_arm')!=arm or contract.get('exp177_target_updates')!=target_updates or contract.get('exp177_code')!=expected_code:
        raise ValueError('branch contract differs')
    origin=contract['exp177_origin_cursor']
    if saved['step']<2502 or saved['step']>2502+target_updates or saved['cursor']!=origin+(saved['step']-2502)*INPUTS:
        raise ValueError('branch cursor or step differs')
    return origin


def verify_source_nll(value, expected=2.537057278677821, tolerance=.02):
    if not math.isfinite(value) or abs(value-expected)>tolerance:
        raise ValueError('source NLL replay failed')


def trim_history(report, saved_step):
    """Exclude journal records beyond the durable checkpoint on restart."""
    result=dict(report)
    result['history']=[row for row in report.get('history',[]) if row.get('step',0)<=saved_step]
    result['evaluations']=[row for row in report.get('evaluations',[]) if row.get('updates',0)<=saved_step-2502]
    return result


def notify(text):
    try:
        from extent.notifications import send_telegram_message
        send_telegram_message('Extent EXP177: '+text)
    except Exception as exc: print('notification_error',type(exc).__name__,flush=True)


def worker(args):
    import torch,numpy as np
    from torch.utils.checkpoint import checkpoint
    from extent.torch_continue_176 import FullStudent,load_config,save_current
    from extent.torch_distill_177 import distillation_microbatch
    from extent.broad_corpus import digest_file,write_json
    from extent.weekend_storage_176 import disk_guard,check_ram_growth,host_available
    from engine_torch170 import sync_remote
    from extent.hf_download_retry import DownloadPolicy
    from transformers import AutoModelForCausalLM
    root=Path(args.root); work=Path(args.work_dir); branch=root/(('smoke-' if args.smoke else '')+args.arm);branch.mkdir(exist_ok=True)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    free,total=torch.cuda.mem_get_info();budget=min(args.memory_gib*2**30,free-5*2**30)
    if budget<40*2**30:raise MemoryError('insufficient GPU memory with 5GiB headroom')
    torch.cuda.set_per_process_memory_fraction(budget/total)
    saved_file=branch/'CURRENT.pt' if (branch/'CURRENT.pt').exists() else root/'source'/'CURRENT.pt'
    saved=torch.load(saved_file,map_location='cpu',mmap=True,weights_only=True)
    if saved_file.parent.name=='source':
        if saved['step']!=2502 or (not args.probe and digest_file(saved_file)!=SOURCE_SHA):raise ValueError('source integrity differs')
    else:
        d=json.loads((branch/'CURRENT.json').read_text())
        validate_branch_state(saved,d,digest_file(saved_file),args.arm,args.target_updates,code_identity())
    start_step=int(saved['step']);cursor=int(saved['cursor']);origin=int(saved['contract'].get('exp177_origin_cursor',cursor))
    if cursor!=origin+(start_step-2502)*INPUTS:raise ValueError('cursor/step mismatch')
    model=FullStudent(load_config(root/'teacher'),backend='chunked');model.load_state_dict(saved['model'],strict=True,assign=True);model.cuda()
    def hidden(self,tokens):
        h=self.qwen.embed_tokens(tokens);pos=torch.arange(tokens.shape[1],device=tokens.device)[None];rope=self.qwen.rotary_emb(h,pos)
        for i,layer in enumerate(self.qwen.layers):
            mixer=None if i in self.retained else self.mixers[str(i)];inputs=(h,layer,mixer,pos,rope)
            h=checkpoint(self.block,*inputs,use_reentrant=False) if self.checkpoint_blocks and torch.is_grad_enabled() else self.block(*inputs)
        return self.qwen.norm(h)
    import types
    model.hidden=types.MethodType(hidden,model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=0.,fused=True)
    for group in saved['optimizer']['param_groups']:group.update(fused=True,foreach=False)
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:group['lr']=1e-6
    teacher=AutoModelForCausalLM.from_pretrained(root/'teacher',local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    for p in teacher.parameters():p.requires_grad_(False)
    corpus=Path(args.corpus);tr=np.memmap(corpus/'train.bin',dtype='<i4',mode='r');va=np.memmap(corpus/'validation.bin',dtype='<i4',mode='r')
    c4=np.memmap(Path(args.c4)/'validation.bin',dtype='<i4',mode='r')
    if len(tr)<origin+args.target_updates*INPUTS or len(va)<128*LENGTH or len(c4)<128*LENGTH:raise ValueError('corpus exhausted')
    report_path=work/('exp177-probe-'+args.arm+'-'+str(args.batch)+'.json' if args.probe else 'exp177-'+('smoke-' if args.smoke else '')+args.arm+'.json')
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(arm=args.arm,history=[],evaluations=[],origin_cursor=origin,source_step=2502)
    report=trim_history(report,start_step)
    report.update(status='running',step=start_step,cursor=cursor,physical_batch=args.batch,effective_batch=16,source_sha=SOURCE_SHA)
    def write():
        write_json(report_path,report)
        if args.probe:return
        rows={}
        for name in ('ce','kd'):
            path=work/('exp177-'+name+'.json')
            if not path.exists():continue
            item=json.loads(path.read_text());evaluations=item.get('evaluations',[])
            row=dict(step=item.get('step'),source_step=2502,accepted_input_tokens=item.get('inputs',0),selected_batch=item.get('physical_batch'),loop_tokens_per_second=item.get('loop_tps'),history=[dict(ce=item.get('last_ce'),lr=1e-6)] if item.get('last_ce') is not None else [])
            row['last_kl']=item.get('last_kl') if name=='kd' else None
            if evaluations:row.update(last_validation_nll=evaluations[-1]['nll'],last_c4_nll=evaluations[-1]['c4_nll'],validation_step=2502+evaluations[-1]['updates'],validation_windows=128)
            elif item.get('source_nll') is not None:row.update(last_validation_nll=item['source_nll'],validation_step=2502,validation_windows=128)
            rows[name.upper()]=row
        write_json(work/'exp177-status.json',dict(status=report['status'],contract=dict(experiment=177,format='torch-full-ce-v1'),branches=rows))
    def batch(array,at,n):return torch.from_numpy(np.array(array[at:at+n*LENGTH],dtype=np.int64).reshape(n,LENGTH)).cuda()
    def validate(array,detail=False):
        model.eval();v=[]
        with torch.no_grad():
            for i in range(128):
                if (work/'FAST_STOP').exists():raise FastPause()
                v.append(float(model(batch(array,i*LENGTH,1))))
        mean=sum(v)/len(v)
        return (mean,v) if detail else mean
    def validate_kl(array):
        model.eval();values=[]
        with torch.no_grad():
            for i in range(8):
                if (work/'FAST_STOP').exists():raise FastPause()
                _,kl=distillation_microbatch(model,teacher,batch(array,i*LENGTH,1),tile=args.tile);values.append(float(kl))
        return sum(values)/len(values)
    if start_step==2502 and not args.probe:
        report['source_nll']=validate(va)
        verify_source_nll(report['source_nll'])
    write()
    def save():
        disk_guard(work,30);used=check_ram_growth(root,120)
        expected_size=(root/'source'/'CURRENT.pt').stat().st_size
        if used+expected_size>120*2**30 or shutil.disk_usage(root).free<expected_size+5*2**30 or host_available()<expected_size+48*2**30:
            raise OSError('insufficient temporary RAM for atomic checkpoint; previous retained')
        contract=dict(saved['contract'],exp177_arm=args.arm,exp177_origin_cursor=origin,exp177_target_updates=args.target_updates,exp177_code=code_identity(),exp177_teacher_revision=REVISION)
        save_current(branch,dict(version=1,contract=contract,step=step,cursor=cursor,model=model.state_dict(),optimizer=optimizer.state_dict()))
        report['saved_step']=step;report['saved_cursor']=cursor;write()
    step=start_step;last_save=time.monotonic();started=time.monotonic();probe_times=[]
    while step<(start_step+5 if args.probe else 2502+args.target_updates) and time.time()<args.train_deadline:
        if (work/'STOP').exists() or (work/'FAST_STOP').exists():break
        update_started=time.monotonic()
        model.train();optimizer.zero_grad(set_to_none=True);ce_sum=kl_sum=0.
        for offset in range(0,EFFECTIVE,args.batch):
            count=min(args.batch,EFFECTIVE-offset);tokens=batch(tr,cursor+offset*LENGTH,count)
            if args.arm=='ce':ce=model(tokens);kl=ce.detach().new_zeros(())
            else:ce,kl=distillation_microbatch(model,teacher,tokens,tile=args.tile)
            objective=ce if args.arm=='ce' else .5*(ce+kl)
            if not bool(torch.isfinite(objective)):raise FloatingPointError('nonfinite objective')
            (objective*count/EFFECTIVE).backward();ce_sum+=float(ce.detach())*count/EFFECTIVE;kl_sum+=float(kl.detach())*count/EFFECTIVE
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=True));optimizer.step();torch.cuda.synchronize()
        checked=list(model.parameters())
        for state in optimizer.state.values():checked.extend((state['exp_avg'],state['exp_avg_sq']))
        if not bool(torch.stack([torch.isfinite(x).all() for x in checked]).all()):raise FloatingPointError('nonfinite state; previous checkpoint retained')
        step+=1;cursor+=INPUTS;u=step-2502
        if not args.probe:report['hf_verified']=False
        if args.probe:
            probe_times.append(time.monotonic()-update_started)
            print('PROBE',args.arm,args.batch,len(probe_times),'seconds',round(probe_times[-1],2),flush=True)
            if len(probe_times)<5:continue
            import statistics
            report.update(status='measured',seconds_per_update=statistics.median(probe_times[2:]),samples_seconds=probe_times,step=step,accepted_updates=0);write();return report
        if u%16==0:
            disk_guard(work,30);check_ram_growth(root,120)
            report.update(step=step,cursor=cursor,inputs=u*INPUTS,last_ce=ce_sum,last_kl=kl_sum,gradient_norm=norm,teacher_training_inputs=u*INPUTS if args.arm=='kd' else 0,loop_tps=(step-start_step)*INPUTS/max(time.monotonic()-started,1e-9));report['history'].append(dict(step=step,input_tokens=u*INPUTS,ce=ce_sum,kl=kl_sum,gradient_norm=norm));write();print('PROGRESS',args.arm,u,report['inputs'],report['loop_tps'],flush=True)
        if u%128==0:
            try:nll,edu_windows=validate(va,detail=True);c4nll,c4_windows=validate(c4,detail=True);kl8=validate_kl(va)
            except FastPause:break
            report['evaluations'].append(dict(updates=u,input_tokens=u*INPUTS,nll=nll,c4_nll=c4nll,edu_per_window=edu_windows,c4_per_window=c4_windows,teacher_kl_8windows=kl8));write();print('VALIDATION',args.arm,u,nll,c4nll,flush=True)
        if step==start_step+1 or time.monotonic()-last_save>=1800:save();last_save=time.monotonic()
    report.update(status='matched_complete' if step==2502+args.target_updates else 'partial',step=step,cursor=cursor,inputs=(step-2502)*INPUTS,teacher_training_inputs=(step-2502)*INPUTS if args.arm=='kd' else 0,last_ce=ce_sum if step>start_step else report.get('last_ce'),last_kl=kl_sum if step>start_step else report.get('last_kl'))
    save();print('LOCAL_SAVED',args.arm,step,flush=True)
    if (work/'FAST_STOP').exists():
        notify(args.arm+' fast pause; state retained in RAM');return report
    if time.time()<args.hard_deadline-120:
        try:
            report['final_nll'],report['final_edu_per_window']=validate(va,detail=True);report['final_c4_nll'],report['final_c4_per_window']=validate(c4,detail=True);report['final_teacher_kl_8windows']=validate_kl(va);write()
        except FastPause:
            notify(args.arm+' fast pause during validation; state retained in RAM');return report
    if args.sync:
        try:
            remote=sync_remote(branch,report_path,DownloadPolicy(root/'transport',deadline=args.hard_deadline-30),prefix=args.remote_prefix+'/'+args.arm,remove_local=False,checkpoint_step=step)
            report.update(hf_verified=True,remote=remote);write();notify(args.arm+' saved; step='+str(step))
        except Exception as e:
            report.update(hf_verified=False,hf_error=type(e).__name__);write();notify(args.arm+' HF failed; RAM checkpoint retained');raise
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--work-dir',required=True);p.add_argument('--until',required=True);p.add_argument('--corpus',required=True);p.add_argument('--c4',required=True);p.add_argument('--save-reserve-minutes',type=float,default=25);p.add_argument('--target-updates',type=int,default=0);p.add_argument('--batch',type=int,default=1);p.add_argument('--tile',type=int,default=128);p.add_argument('--memory-gib',type=float,default=90);p.add_argument('--root');p.add_argument('--arm',choices=('ce','kd'));p.add_argument('--train-deadline',type=float);p.add_argument('--hard-deadline',type=float);p.add_argument('--remote-prefix',default=PREFIX);p.add_argument('--sync',action='store_true');p.add_argument('--probe',action='store_true');p.add_argument('--clear-stop',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--max-batch',type=int,default=8);a=p.parse_args()
    if a.arm:
        try:return worker(a)
        except BaseException as exc:
            from extent.broad_corpus import write_json
            name=('exp177-probe-'+a.arm+'-'+str(a.batch)+'.json' if a.probe else 'exp177-'+a.arm+'-failure.json')
            write_json(Path(a.work_dir)/name,dict(status='failed',error_type=type(exc).__name__,error=str(exc)[:1200],traceback=traceback.format_exc()))
            raise
    from filelock import FileLock
    from extent.weekend_storage_176 import create_ram_workspace,disk_guard
    from extent.hf_download_retry import DownloadPolicy
    from engine_torch170 import restore_remote
    from huggingface_hub import snapshot_download
    deadline=datetime.fromisoformat(a.until)
    if deadline.tzinfo is None:raise ValueError('deadline needs timezone')
    hard=deadline.timestamp();work=Path(a.work_dir).resolve();work.mkdir(parents=True,exist_ok=True)
    remote_prefix=PREFIX+'/'+hashlib.sha256((str(work)+SOURCE_SHA).encode()).hexdigest()[:20]
    if a.save_reserve_minutes<18 or hard-time.time()<3600 or not 1<=a.batch<=16 or not 1<=a.max_batch<=16 or a.memory_gib<=0:raise ValueError('invalid deadline, reserve or batch')
    if not os.environ.get('TELEGRAM_CHAT_ID') and os.environ.get('TELEGRAM_USER_ID'):os.environ['TELEGRAM_CHAT_ID']=os.environ['TELEGRAM_USER_ID']
    missing=[name for name in ('HF_TOKEN','TELEGRAM_BOT_TOKEN','TELEGRAM_CHAT_ID') if not os.environ.get(name)]
    if missing:raise ValueError('Missing required environment variables: '+', '.join(missing))
    with FileLock(str(work/'server.lock'),timeout=0):
        for name in ('STOP','FAST_STOP'):
            flag=work/name
            if a.clear_stop:flag.unlink(missing_ok=True)
            elif flag.exists():raise ValueError(name+' exists; clear explicitly to resume')
        disk_guard(work,30);root=create_ram_workspace(work)
        os.environ.update(HF_HOME=str(root/'hf-cache'),HF_XET_CACHE=str(root/'xet'),TMPDIR=str(root/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'),TRITON_CACHE_DIR=str(root/'triton'),CUDA_CACHE_PATH=str(root/'cuda-cache'),USE_TORCH='1',USE_TF='0',USE_FLAX='0')
        (root/'tmp').mkdir(exist_ok=True);policy=DownloadPolicy(root/'transport',deadline=hard-a.save_reserve_minutes*60)
        notify('started; source step2502')
        from extent.broad_corpus import write_json
        write_json(work/'exp177-status.json',dict(status='preparing',contract=dict(experiment=177),branches={}))
        try:
            from extent.status_autostart import ensure_listener
            ensure_listener(work,'exp177-status.json')
        except Exception as exc:print('status_bot_error',type(exc).__name__,flush=True)
        try:
            source=root/'source';source.mkdir(exist_ok=True)
            if not (source/'CURRENT.pt').exists():restore_remote(source,policy,prefix=SOURCE_PREFIX)
            from extent.broad_corpus import digest_file,write_json
            if digest_file(source/'CURRENT.pt')!=SOURCE_SHA:raise ValueError('pinned source checksum differs')
            from extent.broad_corpus import PackedCorpus
            expected=json.loads((Path(__file__).parent/'results/EXP-176-source-reference.json').read_text())
            corpus=PackedCorpus(a.corpus)
            if corpus.identity!=expected['corpus_identity']:raise ValueError('training corpus identity mismatch')
            available_updates=max(0,(len(corpus.arrays['train'])-124878848)//INPUTS)
            PackedCorpus(a.c4)
            policy.call(snapshot_download,repo_id=TEACHER,revision=REVISION,local_dir=str(root/'teacher'),allow_patterns=['*.json','*.safetensors'],token=os.environ.get('HF_TOKEN'))
            existing_plan=work/'exp177-plan.json'
            if a.target_updates<=0 and existing_plan.exists() and not a.preflight_only:
                prior=json.loads(existing_plan.read_text());a.target_updates=prior['target_updates'];a.batch=prior['batch'];a.tile=prior['tile'];a.memory_gib=prior['memory_gib']
            if a.target_updates<=0:
                timings={}
                for arm in ('ce','kd'):
                    best=None
                    for batch in range(1,a.max_batch+1):
                        if (work/'STOP').exists() or (work/'FAST_STOP').exists():
                            write_json(work/'exp177-status.json',dict(status='paused_before_training',contract=dict(experiment=177),branches={}))
                            notify('paused during preflight; source retained in RAM');return
                        if time.time()>hard-a.save_reserve_minutes*60-3600:break
                        path=work/('exp177-probe-'+arm+'-'+str(batch)+'.json')
                        path.unlink(missing_ok=True)
                        cmd=[sys.executable,'-u','-m','engine_torch177','--arm',arm,'--probe','--work-dir',str(work),'--root',str(root),'--until',a.until,'--corpus',a.corpus,'--c4',a.c4,'--target-updates','1','--batch',str(batch),'--tile',str(a.tile),'--memory-gib',str(a.memory_gib),'--train-deadline',str(hard-a.save_reserve_minutes*60),'--hard-deadline',str(hard)]
                        try:result=subprocess.run(cmd,timeout=min(600,max(1,hard-time.time()-a.save_reserve_minutes*60)),check=False)
                        except subprocess.TimeoutExpired:break
                        if not path.exists():raise RuntimeError('probe produced no report; inspect output')
                        row=json.loads(path.read_text())
                        if result.returncode:
                            message=(row.get('error_type','')+' '+row.get('error','')).lower()
                            if 'outofmemory' in message or 'out of memory' in message or row.get('error_type')=='MemoryError':break
                            raise RuntimeError('non-OOM probe failure: '+message[:300])
                        if row.get('status')!='measured':break
                        if best is None or row['seconds_per_update']<best['seconds_per_update']:best=dict(batch=batch,seconds_per_update=row['seconds_per_update'])
                    if best is None:raise RuntimeError('no verified '+arm+' batch; inspect worker output')
                    timings[arm]=best
                # Use one physical batch for both arms to hold optimization geometry fixed.
                a.batch=min(timings['ce']['batch'],timings['kd']['batch'])
                # The best batches may differ. Measure the shared physical batch itself.
                shared={}
                for arm in ('ce','kd'):
                    path=work/('exp177-probe-'+arm+'-'+str(a.batch)+'.json')
                    row=json.loads(path.read_text())
                    shared[arm]=row['seconds_per_update']
                a.target_updates,_=choose_budget(hard-time.time(),shared['ce']*1.1,shared['kd']*1.1,reserve=a.save_reserve_minutes*60+1200,safety=.9)
                a.target_updates=min(a.target_updates,available_updates)
                if a.target_updates<1:raise RuntimeError('no time or fresh corpus left after preflight')
                write_json(work/'exp177-preflight.json',dict(timings=timings,shared_batch_seconds=shared,chosen_batch=a.batch,target_updates=a.target_updates,limitation='median of three full updates after two warmups; 10 percent speed and 10 percent wall-time margins; matched endpoints may be partial'))
            if a.preflight_only:
                notify('preflight finished; no accepted training updates');return
            if a.target_updates>available_updates:raise ValueError('matched target exceeds remaining corpus')
            contract=work/'exp177-plan.json';existing=json.loads(contract.read_text()) if contract.exists() else None
            plan=dict(target_updates=a.target_updates,batch=a.batch,tile=a.tile,memory_gib=a.memory_gib,source_sha=SOURCE_SHA,teacher_revision=REVISION,code=code_identity())
            if existing and existing!=plan:raise ValueError('existing plan differs; use a new work dir')
            write_json(contract,plan)
            def invoke(arm,limit):
                branch=root/arm;report_path=work/('exp177-'+arm+'.json')
                if not (branch/'CURRENT.pt').exists():
                    previous=json.loads(report_path.read_text()) if report_path.exists() else {}
                    if previous.get('saved_step') and not previous.get('hf_verified'):raise RuntimeError('local state missing and no verified remote; refuse restart')
                    branch.mkdir(exist_ok=True);restore_remote(branch,policy,prefix=remote_prefix+'/'+arm)
                    if (branch/'CURRENT.pt').exists():
                        import torch
                        restored=torch.load(branch/'CURRENT.pt',map_location='cpu',mmap=True,weights_only=True)
                        write_json(branch/'CURRENT.json',dict(step=restored['step'],cursor=restored['cursor'],contract=restored['contract'],sha256=digest_file(branch/'CURRENT.pt'),bytes=(branch/'CURRENT.pt').stat().st_size))
                cmd=[sys.executable,'-u','-m','engine_torch177','--arm',arm,'--work-dir',str(work),'--root',str(root),'--until',a.until,'--corpus',a.corpus,'--c4',a.c4,'--target-updates',str(a.target_updates),'--batch',str(a.batch),'--tile',str(a.tile),'--memory-gib',str(a.memory_gib),'--train-deadline',str(limit),'--hard-deadline',str(hard),'--remote-prefix',remote_prefix,'--sync']
                try:return subprocess.run(cmd,check=False,timeout=max(1,hard-time.time()-30)).returncode
                except subprocess.TimeoutExpired:raise RuntimeError(arm+' exceeded hard deadline; last local state retained')
            work_end=hard-a.save_reserve_minutes*60
            half=time.time()+(work_end-time.time())/2
            for arm,limit in (('ce',half),('kd',work_end)):
                if time.time()>=limit:break
                code=invoke(arm,limit)
                if code:raise RuntimeError(arm+' worker failed; inspect logs and retained checkpoint')
                if (work/'STOP').exists() or (work/'FAST_STOP').exists():break
            reports=[json.loads((work/('exp177-'+x+'.json')).read_text()) for x in ('ce','kd') if (work/('exp177-'+x+'.json')).exists()]
            matched=len(reports)==2 and all(x['status']=='matched_complete' for x in reports)
            write_json(work/'exp177-summary.json',dict(status='matched_complete' if matched else 'partial',arms=reports,source_sha=SOURCE_SHA))
            status_file=work/'exp177-status.json'
            if status_file.exists():
                status=json.loads(status_file.read_text());status['status']='matched_complete' if matched else 'partial';write_json(status_file,status)
            notify(('matched complete' if matched else 'partial')+'; inspect reports')
            if matched and all(x.get('hf_verified') for x in reports):
                try:
                    from extent.weekend_storage_176 import verify_owned
                    verified=verify_owned(root)
                    if any(p.is_symlink() for p in verified.rglob('*')):raise ValueError('symlink prevents cleanup')
                    for arm,report in zip(('ce','kd'),reports):
                        state=verified/arm/'CURRENT.pt';descriptor=json.loads((state.parent/'CURRENT.json').read_text())
                        if digest_file(state)!=report['remote']['sha256'] or descriptor['sha256']!=report['remote']['sha256']:raise ValueError('cleanup SHA differs')
                    if digest_file(verified/'source'/'CURRENT.pt')!=SOURCE_SHA:raise ValueError('source changed; retain RAM')
                    shutil.rmtree(verified);(work/'ram-workspace.json').unlink(missing_ok=True)
                    write_json(work/'exp177-cleanup.json',dict(status='verified_and_removed'))
                    print('VERIFIED RAM CLEANUP COMPLETE',flush=True)
                except Exception as exc:
                    write_json(work/'exp177-cleanup.json',dict(status='retained',error=type(exc).__name__))
                    notify('training completed; automatic RAM cleanup refused; files retained')
        except BaseException as e:
            status_file=work/'exp177-status.json'
            if status_file.exists():
                status=json.loads(status_file.read_text());status['status']='failed';write_json(status_file,status)
            notify('failed; local states retained; '+type(e).__name__);raise
if __name__=='__main__':main()
