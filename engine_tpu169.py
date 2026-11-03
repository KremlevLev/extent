"""EXP169 CPU supervisor. TPU is initialized only by disposable workers."""
import argparse,json,os,shutil,subprocess,sys,time,math,signal
from pathlib import Path


def ram_root(path):
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True)
    mounts=[]
    for row in Path('/proc/self/mountinfo').read_text().splitlines():
        fields=row.split();mount=Path(fields[4]);kind=row.split(' - ',1)[1].split()[0]
        if path==mount or mount in path.parents:mounts.append((len(str(mount)),kind))
    if not mounts or max(mounts)[1]!='tmpfs':raise ValueError('Large work directory MUST be tmpfs, not the20GB SSD')
    free=shutil.disk_usage(path).free
    available=int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:')))*1024
    limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and limit.read_text().strip()!='max':available=min(available,int(limit.read_text())-int(current.read_text()))
    legacy_limit=Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    if legacy_limit.exists():
        available=min(available,int(legacy_limit.read_text())-int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text()))
    if free<128*2**30 or available<256*2**30:raise MemoryError('Need >=128GiB free tmpfs and >=256GiB effective host RAM before source loading')
    return dict(tmpfs_free_bytes=free,effective_host_available=available)


def environment(root,platform):
    env=dict(os.environ,JAX_PLATFORMS=platform,HF_HOME=str(root/'hf'),HF_HUB_CACHE=str(root/'hf/hub'),
        HUGGINGFACE_HUB_CACHE=str(root/'hf/hub'),TMPDIR=str(root/'tmp'),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        USE_TORCH='0',USE_TF='0',USE_FLAX='0')
    (root/'tmp').mkdir(exist_ok=True)
    env.pop('CUDA_VISIBLE_DEVICES',None)
    return env


def compare_pilots(endpoints,target):
    names=('control','ce-only')
    if any(n not in endpoints for n in names):return dict(comparable=False,reason='missing pilot')
    rows=[endpoints[n] for n in names]
    if any(r.get('accepted_input_tokens')!=target['input_tokens'] or r.get('status')!='pilot_completed' for r in rows):
        return dict(comparable=False,reason='incomplete or unequal accepted input budgets')
    if rows[0]['data_windows']!=rows[1]['data_windows']:return dict(comparable=False,reason='data cursor differs')
    panels=[(r.get('final') or {}).get('last',{}).get('panels',{}) for r in rows]
    if any(set(p)!=set('ABCDE') for p in panels):return dict(comparable=False,reason='partial final evaluation')
    return dict(comparable=True,ce_only_minus_mixed_nll={p:panels[1][p]['nll']-panels[0][p]['nll'] for p in 'ABCDE'},
        scope='short matched fresh-prefix pilot; not long-run superiority')


def prepare(root,hours):
    deadline=time.time()+hours*3600
    from extent.hf_download_retry import install_download_retries
    install_download_retries()
    from extent.hf_artifact_sync import artifact_config_from_env,restore_artifact
    root=Path(root);(root/'results').mkdir(exist_ok=True)
    hub=artifact_config_from_env(include_kaggle_secrets=False)
    if hub is None:raise ValueError('HF_TOKEN required')
    report=root/'results/source168.json'
    if not report.exists() and not restore_artifact(report,'experiments/exp168-full-long-v1/latest.json',hub):raise FileNotFoundError('EXP168 report unavailable')
    from extent.tpu_diagnostic_169 import freeze_parent
    freeze_parent(root,hub,deadline=deadline)
    from extent.tpu_data_169 import restore
    if restore(root,hub,json.loads(report.read_text())):
        print('EXP169 verified corpus packs restored; no retokenization',flush=True)
        return
    from scripts.prepare_broad_corpus import main as c4
    from scripts.prepare_edu_corpus import main as edu
    from scripts.prepare_extended_edu import main as extended
    def remaining():
        value=(deadline-time.time())/3600
        if value<=0:raise TimeoutError('CPU preparation budget exhausted; journals retained')
        return str(value)
    c4(['--output',str(root/'data-pack'),'--train-tokens','100000000','--validation-windows','512','--max-hours',remaining(),'--max-local-gb','8'])
    edu(['--output',str(root/'data-pack-edu'),'--c4-dir',str(root/'data-pack'),'--max-hours',remaining()])
    extended(['--output',str(root/'data-pack-edu-600m'),'--old-dir',str(root/'data-pack-edu'),'--c4-dir',str(root/'data-pack'),'--max-hours',remaining()])


def worker(root,mode,batch,engine,hours,probe,stop_new_steps=None):
    import jax
    source_runtime=json.loads((root/'tpu169/source.json').read_text())['runtime']
    jax.config.update('jax_default_matmul_precision',source_runtime['matmul_precision'])
    jax.config.update('jax_default_prng_impl',source_runtime['prng_impl'])
    jax.config.update('jax_compilation_cache_dir',str(root/'compile-cache'))
    os.environ.update(EXTENT169_BATCH=str(batch),EXTENT169_ENGINE=engine,EXTENT169_PROBE='1' if probe else '0',EXTENT169_MODE=mode)
    from scripts.m3q_tpu_diagnostic_169 import parser,run
    from extent.tpu_diagnostic_169 import make_session
    from extent.hf_artifact_sync import artifact_config_from_env
    from extent.hf_download_retry import install_download_retries
    install_download_retries()
    args=parser().parse_args(['--source-plan',str(root/'tpu169/source.json'),'--mode',mode,
        '--corpus-dir',str(root/'data-pack'),'--edu-dir',str(root/'data-pack-edu-600m'),
        '--max-wall-hours',str(hours),'--save-reserve-minutes','18','--max-steps','1000000000',
        '--max-new-hf-gb','48','--output-dir',str(root/'results'),'--cache-dir',str(root/'cache'),
        '--checkpoint-dir',str(root/'states'),'--work-root',str(root),'--max-local-gb','220',
        '--min-free-gb','16','--stop-file',str(root/'STOP'),'--no-telegram']+(['--probe-only'] if probe else [])+(['--stop-new-steps',str(stop_new_steps)] if stop_new_steps else []))
    result=run(args)
    if probe:
        target=root/'probe-result.json';target.write_text(json.dumps(result['probe']))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default='/dev/shm/extent169');p.add_argument('--output-dir',default='/kaggle/working/extent-results')
    p.add_argument('--max-wall-hours',type=float,default=9.);p.add_argument('--stage',choices=('supervise','prepare','worker'),default='supervise')
    p.add_argument('--batch',type=int,default=1);p.add_argument('--engine',choices=('compiled','streamed'),default='compiled')
    p.add_argument('--mode',default='control');p.add_argument('--probe',action='store_true')
    p.add_argument('--stop-new-steps',type=int)
    a=p.parse_args(argv);root=Path(a.work_dir);root.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':return prepare(root,a.max_wall_hours)
    if a.stage=='worker':return worker(root,a.mode,a.batch,a.engine,a.max_wall_hours,a.probe,a.stop_new_steps)
    from extent.isolated_tpu_planner import grow,run_probe,terminate_group
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);path=out/'extent-m3q-tpu169.json'
    report=dict(experiment=169,status='preflight',trials={},warnings=[],target_input_tokens_per_second=6400)
    def persist():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
    def notify(text):
        try:
            from extent.notifications import send_telegram_message
            send_telegram_message('Extent EXP169: '+text)
        except Exception as exc:report['warnings'].append('notification: '+type(exc).__name__)
    started=time.time();wall=started+a.max_wall_hours*3600
    def stop(signum,frame):
        (root/'STOP').write_text('requested by signal '+str(signum))
        print('EXP169 soft stop requested; waiting for worker save',flush=True)
    old_signals={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    os.environ.setdefault('EXTENT_HF_CHECKPOINT_REPO','levkremlev888/extent');os.environ.setdefault('EXTENT_HF_REPO_TYPE','dataset')
    os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(wall-18*60);os.environ['EXTENT_HF_SAVE_DEADLINE_UNIX']=str(wall-60)
    try:
        if not 1<=a.max_wall_hours<=9:raise ValueError('total budget must be between1 and9 hours')
        report['resources']=ram_root(root);persist();notify('started; shared9h maximum; isolated integer batch probes')
        command=[sys.executable,'-u','-m','engine_tpu169','--work-dir',str(root)]
        env=environment(root,'cpu')
        prep=subprocess.Popen(command+['--stage','prepare','--max-wall-hours','1.25'],env=env,start_new_session=(os.name=='posix'))
        try:code=prep.wait(timeout=min(4500,max(1,wall-time.time()-1800)))
        except subprocess.TimeoutExpired:terminate_group(prep);raise TimeoutError('data/source preparation timed out')
        if code:raise RuntimeError('CPU source/data preparation failed; see notebook log')
        env=environment(root,'tpu,cpu');os.environ.update(env)
        deadline=min(time.monotonic()+3000,time.monotonic()+max(0,wall-time.time()-3600))
        fits=[]
        for engine in ('compiled','streamed'):
            engine_deadline=min(deadline,time.monotonic()+1500)
            def probe(batch):
                name=f'probe-{engine}-b{batch}'
                return run_probe(command+['--stage','worker','--probe','--engine',engine,'--batch',str(batch),'--mode',name,'--max-wall-hours','1'],
                    root/'probe-result.json',root/(name+'.log'),min(600,max(1,engine_deadline-time.monotonic())),stop=lambda:(root/'STOP').exists())
            result=grow(probe,max_batch=128,deadline=engine_deadline,record=lambda value:(report['trials'].update({engine:value}),persist()))
            for row in result['trials']:
                if row['status']=='measured':fits.append((row['report']['input_tokens_per_second'],row['batch'],engine))
            if time.monotonic()>=deadline:break
        if not fits:raise RuntimeError('No verified TPU batch fit; worker logs retained')
        if (root/'STOP').exists():raise RuntimeError('requested stop during planning; no training launched')
        speed,batch,engine=max(fits);report.update(status='training',selected_batch=batch,execution=engine,measured_probe_tokens_per_second=speed);persist()
        # Same source, Adam, physical batch and fresh1M-token prefix. These are
        # short pilots, not long-run superiority claims. Continue control after.
        pilot_steps=math.ceil(1_000_000/(batch*256))
        report['pilot_target']=dict(updates=pilot_steps,input_tokens=pilot_steps*batch*256)
        for mode in ('control','ce-only'):
            if wall-time.time()<3*3600:report['warnings'].append('paired pilots omitted/partial: remaining budget');break
            pilot=subprocess.Popen(command+['--stage','worker','--engine',engine,'--batch',str(batch),'--mode',mode,
                '--stop-new-steps',str(pilot_steps),'--max-wall-hours',str(min(1.25,(wall-time.time()-2*3600)/3600))],env=env,start_new_session=(os.name=='posix'))
            try:code=pilot.wait(timeout=min(4500,max(1,wall-time.time()-2*3600)))
            except subprocess.TimeoutExpired:terminate_group(pilot);code=-1
            report.setdefault('pilot_attempts',[]).append(dict(mode=mode,returncode=code));persist()
            pilot_path=root/'results'/('runtime-tpu169-'+mode+'.json')
            if pilot_path.exists():
                value=json.loads(pilot_path.read_text());row=value['branches']['FULL-KD']
                report.setdefault('pilot_endpoints',{})[mode]=dict(step=row['step'],data_windows=row.get('training_windows_seen'),
                    accepted_input_tokens=row.get('accepted_input_tokens'),final=row.get('final'),status=value.get('status'))
            if (root/'STOP').exists():break
        report['paired_comparison']=compare_pilots(report.get('pilot_endpoints',{}),report['pilot_target']);persist()
        # Retry the SAME contract to restore own CURRENT, never source-reset after updates.
        for attempt in range(2):
            if (root/'STOP').exists():break
            hours=(wall-time.time())/3600
            if hours<.55:break
            train=subprocess.Popen(command+['--stage','worker','--engine',engine,'--batch',str(batch),'--mode','control','--max-wall-hours',str(hours)],env=env,start_new_session=(os.name=='posix'))
            try:code=train.wait(timeout=max(1,wall-time.time()+30))
            except subprocess.TimeoutExpired:terminate_group(train);code=-1
            report.setdefault('training_attempts',[]).append(dict(returncode=code));persist()
            if code==0:break
        report['status']='finished' if report.get('training_attempts',[])[-1:]==[dict(returncode=0)] else 'training_partial'
    except Exception as exc:report.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
    finally:
        report['wall_seconds']=time.time()-started
        for name in ('runtime-tpu169-control.json','runtime-tpu169-ce-only.json'):
            source=root/'results'/name
            if source.exists():
                shutil.copy2(source,out/name)
                if name=='runtime-tpu169-control.json':
                    value=json.loads(source.read_text());row=value['branches']['FULL-KD']
                    report['control_summary']=dict(status=value.get('status'),step=row['step'],
                        accepted_input_tokens=row.get('accepted_input_tokens',0),durable_step=row.get('durable_step'),
                        pending_slots=value.get('pending_slots',[]),finalization_status=value.get('finalization_status'))
                    if value.get('pending_slots') or value.get('storage_error'):report['status']='storage_partial'
        for log in root.glob('probe-*.log'):
            with log.open('rb') as handle:
                handle.seek(max(0,log.stat().st_size-65536));(out/log.name).write_bytes(handle.read())
        persist();notify(report['status']+'; download extent-m3q-tpu169.json and runtime-tpu169-control.json')
        for sig,handler in old_signals.items():signal.signal(sig,handler)
    return report


if __name__=='__main__':main()
