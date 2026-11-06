"""EXP173: frozen171 -> composed gqa7 CE+KD, isolated phase-aware TPU workers."""
import argparse,faulthandler,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from extent.broad_corpus import write_json,digest_file
from extent.isolated_tpu_planner import terminate_group

PREFIX='experiments/exp173-gqa7-end-to-end-v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default='/dev/shm/extent173')
    p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--max-wall-hours',type=float,default=8.8)
    p.add_argument('--stage',choices=['parent','prepare','probe','train'],default='parent')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--context',type=int,default=1024,choices=[256,1024,4096])
    p.add_argument('--effective-batch',type=int,default=16)
    p.add_argument('--total-updates',type=int,default=12208)
    p.add_argument('--planner-minutes',type=float,default=60.)
    p.add_argument('--probe-minutes',type=float,default=20.)
    p.add_argument('--max-batch',type=int,default=16)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--no-hf',action='store_true')
    p.add_argument('--no-telegram',action='store_true')
    return p


def phase(root,name,**extra):
    write_json(root/'phase.json',dict(phase=name,unix=time.time(),pid=os.getpid(),**extra))
    print('EXP173 phase='+name,extra,flush=True)


def environment(root,platform):
    env=dict(os.environ,JAX_PLATFORMS=platform,USE_TORCH='0',USE_TF='0',USE_FLAX='0',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HOME=str(root/'hf'),HF_HUB_CACHE=str(root/'hf/hub'),
        TMPDIR=str(root/'tmp'),EXTENT_HF_RETRY_STATE_DIR=str(root/'transport'))
    for key in ('PJRT_DEVICE','TPU_PROCESS_ADDRESSES','CUDA_VISIBLE_DEVICES'):env.pop(key,None)
    (root/'tmp').mkdir(exist_ok=True)
    return env


def prepare(root):
    from huggingface_hub import hf_hub_download
    from extent.hf_download_retry import install_download_retries
    install_download_retries()
    from extent.qwen_source import QWEN3_1_7B_BASE as source
    phase(root,'download_original_Qwen')
    for name in ('config.json','model.safetensors'):
        hf_hub_download(source.repo_id,name,revision=source.revision,local_dir=root/'qwen')
    phase(root,'restore_verified_Edu')
    prefix='experiments/exp169-data-c4-edu600m-v1'
    file=hf_hub_download(os.environ.get('EXTENT_HF_CHECKPOINT_REPO','levkremlev888/extent'),
        prefix+'/inventory.json',repo_type='dataset',local_dir=root/'bundle')
    inventory=json.loads(Path(file).read_text())
    for leaf in ('manifest.json','train.bin','validation.bin'):
        name='data-pack-edu-600m/'+leaf
        meta=inventory['files'][name]
        target=root/'edu'/leaf;target.parent.mkdir(exist_ok=True)
        if target.is_file() and target.stat().st_size==meta['bytes'] and digest_file(target)==meta['sha256']:
            continue
        downloaded=Path(hf_hub_download(os.environ.get('EXTENT_HF_CHECKPOINT_REPO','levkremlev888/extent'),
            prefix+'/'+name,repo_type='dataset',local_dir=root/'bundle'))
        if downloaded.stat().st_size!=meta['bytes'] or digest_file(downloaded)!=meta['sha256']:
            raise ValueError('published Edu SHA mismatch')
        if downloaded.resolve()!=target.resolve():downloaded.replace(target)
    from extent.broad_corpus import PackedCorpus
    corpus=PackedCorpus(root/'edu')
    if corpus.identity!=inventory['identities']['data-pack-edu-600m']:raise ValueError('Edu identity mismatch')
    if corpus.manifest['contract'].get('tokenizer')!=source.repo_id or corpus.manifest['contract'].get('tokenizer_revision')!=source.revision:
        raise ValueError('Edu tokenizer differs from pinned original Qwen')
    from extent.quality_172 import fetch_bank
    phase(root,'restore_frozen171_CURRENT5120_parameters_only')
    fetch_bank(root/'bank')
    phase(root,'data_ready')


def wait_child(command,root,log_path,timeout,*,stop=None):
    """No accelerator in parent. SIGUSR1 writes Python stacks before timeout."""
    begun=time.monotonic();last=-1
    with Path(log_path).open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
        try:
            while child.poll() is None:
                elapsed=time.monotonic()-begun
                if int(elapsed//60)!=last:
                    last=int(elapsed//60)
                    stamp=json.loads((root/'phase.json').read_text()) if (root/'phase.json').exists() else {}
                    print('EXP173 worker elapsed_minutes=',round(elapsed/60,1),'stage=',stamp.get('phase'),flush=True)
                if elapsed>timeout or (stop and stop()):
                    if os.name=='posix':
                        try:os.kill(child.pid,signal.SIGUSR1)
                        except ProcessLookupError:pass
                    time.sleep(1)
                    terminate_group(child)
                    return dict(status='phase_timeout' if elapsed>timeout else 'requested_stop',returncode=child.returncode,
                        last_phase=json.loads((root/'phase.json').read_text()) if (root/'phase.json').exists() else {},seconds=elapsed)
                time.sleep(.5)
        finally:
            if child.poll() is None:terminate_group(child)
    return dict(status='exited',returncode=child.returncode,seconds=time.monotonic()-begun)


def worker(a,root):
    faulthandler.enable()
    if hasattr(signal,'SIGUSR1'):faulthandler.register(signal.SIGUSR1,all_threads=True)
    faulthandler.dump_traceback_later(180,repeat=True)
    phase(root,'import_JAX')
    import jax,jax.numpy as jnp,numpy as np,optax
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from extent.clean_tpu_171 import initialize_host,layout,optimizer,teacher_block,student_block,rms
    from extent.end_to_end_173 import make_grad,learning_rate as lr
    from extent.broad_corpus import PackedCorpus
    phase(root,'load_original_host_weights')
    c,tc,q,teacher,last,embedding,finalnorm=initialize_host(root/'qwen',lambda name:phase(root,name))
    from extent.quality_172 import validate_manifest,row_array
    source=json.loads((root/'bank/checkpoint.json').read_text())
    for row in validate_manifest(source):
        parent=q
        for name in row['path'][1:-1]:parent=parent[name]
        value=row_array(row,root/'bank')
        if value.shape!=parent[row['path'][-1]].shape:raise ValueError('source shape mismatch')
        parent[row['path'][-1]]=value
    phase(root,'TPU_device_discovery')
    devices=jax.devices('tpu')
    if len(devices)!=8 or jax.process_count()!=1:raise ValueError('requires8 local TPU chips, one process')
    mesh=Mesh(np.array(devices),('model',));rep=NamedSharding(mesh,P())
    phase(root,'eight_chip_collective_smoke')
    smoke=jax.device_put(np.arange(1024,dtype=np.float32).reshape(8,128),NamedSharding(mesh,P('model',None)))
    smoke_value=jax.jit(lambda x:jnp.sum(x*x))(smoke)
    if not math.isfinite(float(smoke_value)):raise FloatingPointError('TPU collective smoke failed')
    jax.config.update('jax_compilation_cache_dir',str(root/'compile-cache'))
    phase(root,'place_sharded_original_and_clean_bank')
    qlayout=layout(q,mesh);tlayout=layout(teacher,mesh)
    q=jax.device_put(q,qlayout);teacher=jax.device_put(teacher,tlayout)
    last=jax.device_put(last,layout(last,mesh,stacked=False))
    embedding=jax.device_put(embedding,layout(embedding,mesh,stacked=False));finalnorm=jax.device_put(finalnorm,layout(finalnorm,mesh,stacked=False))
    tx=optimizer();state=tx.init(q)
    state=jax.device_put(state,layout(state,mesh,stacked=False))
    corpus=PackedCorpus(root/'edu');length=a.context;effective=a.effective_batch
    contract=dict(experiment=173,protocol='gqa7-end-to-end-ce-kd-v1',source='Qwen/Qwen3-1.7B-Base',
        revision='ea980cb0a6c2ae4b936e82123acc929f1cec04c1',corpus=corpus.identity,context=length,
        effective_batch=effective,total_updates=a.total_updates,seed=123,geometry='H16/G8/P128/N128/R4',
        loss='.5 CE + .5 full-vocabulary forward KL, T1; composed gqa7', source_step=5120,source_revision='d341d2b8f87592f617c44c92c5d7532669c75755',source_cursor=83886080,optimizer='Adam(.9,.95),eps1e-8,clip1,warm128,cosine.1,peak1e-4',
        code=digest_file(Path(__file__).parent/'extent/end_to_end_173.py'),core_code=digest_file(Path(__file__).parent/'extent/clean_tpu_171.py'),Z='random; QKVO and QK norms inherited')
    from extent.recovery_durability import RecoveryCheckpointStore
    from extent.hf_artifact_sync import artifact_config_from_env
    os.environ.setdefault('EXTENT_HF_CHECKPOINT_REPO','levkremlev888/extent')
    hub=None if a.no_hf else artifact_config_from_env(include_kaggle_secrets=False)
    store=RecoveryCheckpointStore(root/'states',PREFIX,hub)
    step=0;cursor=83886080
    if a.stage in ('train','probe'):
        phase(root,'restore_OWN_CURRENT_only')
        template=dict(parameters=q,optimizer=state,cursor=np.asarray(83886080,np.int64))
        saved=store.restore('ALIGN',contract,template)
        if saved:
            payload,meta=saved;q=jax.device_put(payload['parameters'],qlayout);state=jax.device_put(payload['optimizer'],layout(payload['optimizer'],mesh,stacked=False))
            step=meta['step'];cursor=int(payload['cursor'])
            if cursor!=83886080+step*effective*length:raise ValueError('CURRENT optimizer/data clock mismatch')
    result=dict(status='preflight',contract=contract,branches={'CE-KD-GQA7':dict(step=step,source_step=5120,
        accepted_input_tokens=cursor-83886080,history=[],evaluations=[],selected_batch=a.batch)},pending_slots=[])
    original_count=sum(x.size for x in jax.tree.leaves((teacher,last,embedding,finalnorm)))
    mixer_count=sum(x.size for x in jax.tree.leaves(q))
    attention_per_layer=sum(x.size for x in jax.tree.leaves(teacher['self_attn']))//27
    result['parameters']=dict(trainable_mixers=mixer_count//27*21,original=original_count,
        hybrid_gqa7=original_count+21*(mixer_count//27-attention_per_layer))
    report=root/('probe-result.json' if a.stage=='probe' else 'runtime-tpu173.json')
    write_json(report,result)
    phase(root,'compile_teacher_and_27layer_gradient',batch=a.batch)
    gradient=jax.jit(make_grad(c,tc),out_shardings=(rep,qlayout,rep))
    def batch(at,count,split='train'):
        values=corpus.arrays[split];end=at+count*length
        if end>len(values):raise EOFError('fresh corpus exhausted; no cycling')
        return jax.device_put(np.array(values[at:end]).reshape(count,length),rep)
    # Compile explicitly and separately from measured execution; stage record
    # survives TPU fatal abort or Python import stalls.
    tokens=batch(cursor,a.batch)
    compile_start=time.monotonic()
    executable=gradient.lower(q,teacher,embedding,last,finalnorm,tokens).compile()
    result['compile_seconds']=time.monotonic()-compile_start
    phase(root,'first_end_to_end_gradient_execution')
    loss,g,perlayer=executable(q,teacher,embedding,last,finalnorm,tokens);jax.block_until_ready((loss,g))
    if not bool(jnp.isfinite(loss)) or not all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g)):
        raise FloatingPointError('first clean gradient nonfinite')
    # BF16 check against independent recurrent Mamba on a real donor layer.
    phase(root,'recurrent_gradient_gate')
    from extent.clean_transplant_170 import forward as scan_forward
    from extent.clean_tpu_171 import mamba_forward
    p0=jax.tree.map(lambda x:x[0],q);h=embedding[tokens[:1,:64]].astype(jnp.bfloat16)
    def objective(p,fast):
        out=mamba_forward(h,p,c) if fast else scan_forward(h,p,c)
        return jnp.mean(out.astype(jnp.float32)**2),out
    gates=[]
    for fast in (True,False):gates.append(jax.jit(jax.value_and_grad(lambda p:objective(p,fast),has_aux=True))(p0))
    (v1,y1),g1=gates[0];(v2,y2),g2=gates[1]
    flat1=jnp.concatenate([x.reshape(-1) for x in jax.tree.leaves(g1)]);flat2=jnp.concatenate([x.reshape(-1) for x in jax.tree.leaves(g2)])
    rel=float(jnp.linalg.norm(flat1-flat2)/jnp.maximum(jnp.linalg.norm(flat2),1e-12))
    cos=float(jnp.vdot(flat1,flat2)/(jnp.linalg.norm(flat1)*jnp.linalg.norm(flat2)+1e-30))
    output=float(jnp.linalg.norm(y1.astype(jnp.float32)-y2.astype(jnp.float32))/jnp.maximum(jnp.linalg.norm(y2.astype(jnp.float32)),1e-12))
    result['gradient_gate']=dict(relative_l2=rel,cosine=cos,output_relative_l2=output)
    if rel>.15 or cos<.98 or output>.03:raise ValueError('clean gradient gate failed; thresholds unchanged')
    phase(root,'compile_global_Adam_update')
    def apply(q,state,grad,rate):
        updates,newstate=tx.update(grad,state,q)
        return optax.apply_updates(q,jax.tree.map(lambda v:-rate*v,updates)),newstate
    update=jax.jit(apply,out_shardings=(qlayout,layout(state,mesh,stacked=False)))
    update=update.lower(q,state,g,jnp.asarray(1e-4)).compile()
    phase(root,'measure_complete_effective_update')
    zero=jax.jit(lambda p:jax.tree.map(jnp.zeros_like,p),out_shardings=qlayout)
    accumulate=jax.jit(lambda acc,g,w:jax.tree.map(lambda x,y:x+y*w,acc,g),out_shardings=qlayout)
    isfinite=jax.jit(lambda tree:jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)])),out_shardings=rep)
    def complete(q,state,at,rate,check_stop=False):
        total=zero(q);loss_sum=0.;layers=0.
        for offset in range(0,effective,a.batch):
            count=min(a.batch,effective-offset)
            value,grad,layer_losses=gradient(q,teacher,embedding,last,finalnorm,batch(at+offset*length,count))
            total=accumulate(total,grad,jnp.asarray(count/effective,jnp.float32))
            loss_sum+=value*(count/effective);layers=layers+layer_losses*(count/effective)
        # Finite reduction precedes accepting an optimizer update.
        finite=jnp.isfinite(loss_sum)&isfinite(total)
        if not bool(finite):raise FloatingPointError('nonfinite gradient; no accepted update')
        newq,newstate=update(q,state,total,jnp.asarray(rate,jnp.float32));jax.block_until_ready((newq,newstate))
        if not bool(isfinite((newq,newstate))):raise FloatingPointError('nonfinite optimizer state; previous CURRENT preserved')
        return newq,newstate,float(loss_sum),np.asarray(layers).tolist()
    if a.stage=='probe':
        durations=[];pq,ps=q,state
        for i in range(3):
            begin=time.monotonic();pq,ps,loss_value,_=complete(pq,ps,cursor,1e-4);duration=time.monotonic()-begin
            if i:durations.append(duration)
        result.update(status='measured',finite=True,accepted_updates=0,
            input_tokens_per_second=effective*length/float(np.median(durations)),samples_seconds=durations,
            engineering_input_presentations=3*effective*length,
            per_chip_memory=[d.memory_stats() for d in devices])
        write_json(report,result);phase(root,'probe_complete');return
    branch=result['branches']['CE-KD-GQA7'];result['status']='running'
    deadline=min(time.time()+max(.01,a.max_wall_hours)*3600,float(os.environ.get('EXTENT173_WALL_UNIX','inf')))-18*60
    def snapshot():
        phase(root,'save_OWN_CURRENT',step=step)
        store.save('ALIGN',contract,dict(parameters=q,optimizer=state,cursor=np.asarray(cursor,np.int64)),step)
    # Layer-scanned full model evaluates original teacher and both retention masks.
    masks={'teacher':set(range(28)),'gqa7':{3,7,11,15,19,23,27}}
    def eval_hidden(q,tokens,mask):
        hidden=embedding[tokens].astype(jnp.bfloat16)
        def body(h,inp):
            index,p,m=inp
            h=jax.lax.cond(mask[index],lambda _:teacher_block(h,p,tc),lambda _:student_block(h,p,m,c),None)
            return h,None
        hidden,_=jax.lax.scan(body,hidden,(jnp.arange(27),teacher,q))
        return rms(teacher_block(hidden,last,tc),finalnorm,c.rms_norm_eps)
    evaluate_hidden=jax.jit(eval_hidden)
    def logits_loss(h,labels):
        logits=h.astype(jnp.bfloat16)@embedding.T
        return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32),labels))
    logits_loss=jax.jit(logits_loss)
    def evaluate():
        phase(root,'full_hybrid_NLL',step=step)
        values={}
        for name,retained in masks.items():
            mask=jax.device_put(np.array([i in retained for i in range(27)]),rep)
            total=0.;targets=0
            for window in range(4):
                if (root/'FAST_STOP').exists():return
                token=batch(window*length,1,'validation');hidden=evaluate_hidden(q,token,mask)
                for start in range(0,length-1,128):
                    end=min(length-1,start+128)
                    total+=float(logits_loss(hidden[:,start:end],token[:,start+1:end+1]));targets+=end-start
            values[name+'_edu_validation_nll']=total/targets
        branch['evaluations'].append(dict(step=step,**values))
    evaluate()
    if step==0:
        baseline=branch['evaluations'][-1]
        if abs(baseline['gqa7_edu_validation_nll']-2.5832235722)>.08:raise ValueError('frozen171 source replay mismatch')
    phase(root,'verify_own_local_checkpoint_roundtrip_before_training')
    snapshot()
    recovered=store.restore('ALIGN',contract,dict(parameters=q,optimizer=state,cursor=np.asarray(cursor,np.int64)))
    if recovered is None or recovered[1]['step']!=step or int(recovered[0]['cursor'])!=cursor:
        raise ValueError('own checkpoint roundtrip failed before training')
    del recovered
    last_save=time.monotonic();begin_run=time.monotonic();origin=cursor
    try:
        while step<a.total_updates and time.time()<deadline:
            if any((root/name).exists() for name in ('STOP','FAST_STOP')):break
            if cursor+effective*length>len(corpus.arrays['train']):result['status']='corpus_exhausted';break
            phase(root,'training_update',step=step)
            begin=time.monotonic();q,state,value,layer_losses=complete(q,state,cursor,lr(step,a.total_updates))
            elapsed=time.monotonic()-begin;step+=1;cursor+=effective*length
            branch.update(step=step,accepted_input_tokens=cursor-83886080)
            branch['history'].append(dict(step=step,loss=value,lr=float(lr(step-1,a.total_updates)),seconds=elapsed,
                input_tokens_per_second=effective*length/elapsed,ce=layer_losses[0],kl=layer_losses[1]))
            result['whole_training_loop_tokens_per_second']=(cursor-origin)/max(1.,time.monotonic()-begin_run)
            write_json(report,result)
            if step%16==0:print('EXP173 TRAIN',dict(step=step,physical_batch=a.batch,effective_batch=effective,input_tokens=cursor-83886080,ce=layer_losses[0],kl=layer_losses[1],tokens_per_second=effective*length/elapsed),flush=True)
            if step%128==0 and deadline-time.time()>900:evaluate()
            if time.monotonic()-last_save>900:snapshot();last_save=time.monotonic()
        result['status']='completed' if step==a.total_updates else ('stopped' if (root/'STOP').exists() else 'deadline_partial')
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        snapshot()
        if (root/'FAST_STOP').exists():result['status']='fast_paused'
        elif not a.no_hf:
            phase(root,'sync_verified_CURRENT_to_HF',step=step)
            if not store.sync_until('ALIGN',time.monotonic()+16*60):result['pending_slots']=['ALIGN']
        result['local_checkpoint_step']=step
        result['per_chip_memory']=[d.memory_stats() for d in devices]
        write_json(report,result)
        faulthandler.cancel_dump_traceback_later()


def main(argv=None):
    a=parser().parse_args(argv);root=Path(a.work_dir);root.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':
        faulthandler.enable()
        if hasattr(signal,'SIGUSR1'):faulthandler.register(signal.SIGUSR1,all_threads=True)
        faulthandler.dump_traceback_later(180,repeat=True)
        try:prepare(root)
        finally:faulthandler.cancel_dump_traceback_later()
        return
    if a.stage in ('probe','train'):worker(a,root);return
    from engine_tpu169 import ram_root
    from filelock import FileLock
    if not 1<=a.max_wall_hours<=9:raise ValueError('1..9 total hours required')
    if min(a.context,a.effective_batch,a.total_updates,a.max_batch)<=0:raise ValueError('positive counts required')
    if not a.no_hf and (a.context,a.effective_batch,a.total_updates)!=(1024,16,12208):raise ValueError('noncanonical smoke requires --no-hf')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    result=dict(experiment=173,status='preparing',trials=[],resources=ram_root(root))
    path=out/'extent-m3q-tpu173.json';write_json(path,result)
    deadline=time.time()+a.max_wall_hours*3600
    os.environ['EXTENT173_WALL_UNIX']=str(deadline)
    def notify(message):
        if a.no_telegram:return
        try:
            from extent.notifications import send_telegram_message
            send_telegram_message('Extent EXP173: '+message)
        except Exception:pass
    def stop(signum,frame):(root/'STOP').touch()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    base=[sys.executable,'-u','-m','engine_tpu173','--work-dir',str(root),'--context',str(a.context),
        '--effective-batch',str(a.effective_batch),'--total-updates',str(a.total_updates)]
    if a.no_hf:base.append('--no-hf')
    with FileLock(str(root/'server.lock'),timeout=0):
        try:
            if any((root/k).exists() for k in ('STOP','FAST_STOP')):raise ValueError('STOP exists; remove explicitly to resume')
            os.environ.update(environment(root,'cpu'))
            os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(deadline-18*60)
            os.environ['EXTENT_HF_SAVE_DEADLINE_UNIX']=str(deadline-60)
            notify('started; frozen171 -> composed gqa7 CE+KD; isolated TPU planner')
            prep=wait_child(base+['--stage','prepare'],root,out/'prepare173.log',min(2700,deadline-time.time()-3600))
            if prep['returncode']:raise RuntimeError('original source/data prepare failed')
            os.environ.update(environment(root,'tpu,cpu'))
            planner_end=min(deadline-3600,time.time()+a.planner_minutes*60)
            fits=[]
            for batch in range(1,min(a.max_batch,a.effective_batch)+1):
                if time.time()>=planner_end:break
                trial=wait_child(base+['--stage','probe','--batch',str(batch)],root,out/f'probe173-b{batch}.log',
                    min(a.probe_minutes*60,planner_end-time.time()),stop=lambda:(root/'STOP').exists())
                source=root/'probe-result.json'
                if source.exists():
                    trial['report']=json.loads(source.read_text());source.unlink()
                trial['batch']=batch;result['trials'].append(trial);write_json(path,result)
                print('EXP173 PROBE',dict(batch=batch,returncode=trial['returncode'],
                    tokens_per_second=trial.get('report',{}).get('input_tokens_per_second'),
                    phase=trial.get('last_phase',{}).get('phase')),flush=True)
                if trial['returncode']!=0 or trial.get('report',{}).get('status')!='measured':break
                fits.append(trial)
            if not fits:raise RuntimeError('no verified clean TPU fit; inspect phase and Python stacks in probe173 logs')
            best=max(fits,key=lambda row:row['report']['input_tokens_per_second'])
            result.update(selected_batch=best['batch'],measured_input_tokens_per_second=best['report']['input_tokens_per_second'])
            print('EXP173 SELECTED',dict(physical_batch=best['batch'],effective_batch=a.effective_batch,
                input_tokens_per_update=a.context*a.effective_batch,
                measured_tokens_per_second=result['measured_input_tokens_per_second']),flush=True)
            if a.preflight_only:result['status']='preflight_completed';return result
            hours=(deadline-time.time())/3600
            trial=wait_child(base+['--stage','train','--batch',str(best['batch']),'--max-wall-hours',str(hours)],
                root,out/'train173.log',max(1,deadline-time.time()))
            source=root/'runtime-tpu173.json'
            if source.exists():
                value=json.loads(source.read_text());write_json(out/source.name,value);result['training']=value
            if trial['returncode']:raise RuntimeError('clean training worker failed; see train173.log')
            result['status']=result.get('training',{}).get('status','unknown')
            if result.get('training',{}).get('pending_slots'):result['status']='storage_partial'
        except Exception as exc:result.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
        finally:
            if (root/'phase.json').exists():result['last_phase']=json.loads((root/'phase.json').read_text())
            write_json(path,result);notify(result['status']+'; download extent-m3q-tpu173.json')
    return result


if __name__=='__main__':main()
