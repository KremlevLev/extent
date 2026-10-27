"""Register reports and ensure one read-only listener per bot on this host."""
import hashlib,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
from filelock import FileLock,Timeout
from extent.campaign_checkpoint import write_json_atomic


def paths(token):
    tag=hashlib.sha256(token.encode()).hexdigest()[:20]
    base=Path(tempfile.gettempdir())
    directory=base/('engine-status-service-'+tag);directory.mkdir(mode=0o700,exist_ok=True)
    return directory,base/('engine-status-'+tag+'.lock')


def register(config,root,report):
    root=str(Path(root).resolve())
    rows=json.loads(config.read_text()).get('sessions',[]) if config.exists() else []
    same=[r for r in rows if r.get('work_dir') and str(Path(r['work_dir']).resolve())==root]
    if same:same[0]['report']=report
    else:
        used={r['id'] for r in rows};name='blackwell';i=2
        while name in used:name='blackwell-'+str(i);i+=1
        rows.append(dict(id=name,work_dir=root,report=report))
    write_json_atomic(config,dict(sessions=rows))
    return rows


def legacy_pids(token):
    # Migrate only an exact engine_status module owned by this bot token.
    # Never inspect/terminate training, generic Python or unrelated services.
    found=[]
    if not sys.platform.startswith('linux'):return found
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:
            cmd=(entry/'cmdline').read_bytes().split(b'\0')
            if not any(cmd[i:i+2]==[b'-m',b'engine_status'] for i in range(len(cmd)-1)):continue
            env=dict(part.split(b'=',1) for part in (entry/'environ').read_bytes().split(b'\0') if b'=' in part)
            if env.get(b'TELEGRAM_BOT_TOKEN')==token.encode():found.append(int(entry.name))
        except (OSError,ValueError):continue
    return found


def lock_free(path):
    lock=FileLock(str(path))
    try:lock.acquire(timeout=0)
    except Timeout:return False
    else:lock.release();return True


def ensure_listener(root,report):
    token=os.environ.get('TELEGRAM_BOT_TOKEN')
    chat=os.environ.get('TELEGRAM_CHAT_ID') or os.environ.get('TELEGRAM_USER_ID')
    if not token or not str(chat or '').isdigit():raise ValueError('Telegram credentials required for automatic /check')
    directory,listener_lock=paths(token);config=directory/'sessions.json';ready=directory/'ready.json'
    with FileLock(str(directory/'startup.lock'),timeout=45):
        # Preserve explicit manually registered sessions on first migration.
        old=Path(root)/'status-sessions.json'
        if not config.exists() and old.exists():
            from engine_status import sessions
            write_json_atomic(config,dict(sessions=sessions(old,root)))
        register(config,root,report)
        service=json.loads(ready.read_text()) if ready.exists() else {}
        if (not lock_free(listener_lock) and service.get('config')==str(config)
            and (not sys.platform.startswith('linux') or service.get('pid') in legacy_pids(token))):
            print('status_bot=registered; existing listener reloads registry on /check',flush=True)
            return dict(status='registered',config=str(config))
        if not lock_free(listener_lock):
            pids=legacy_pids(token)
            if len(pids)!=1:raise RuntimeError('Bot lock held by unknown listener; refusing duplicate getUpdates')
            os.kill(pids[0],signal.SIGTERM)
            until=time.monotonic()+35
            while not lock_free(listener_lock) and time.monotonic()<until:time.sleep(.2)
            if not lock_free(listener_lock):raise RuntimeError('Existing bot listener has not stopped; no duplicate started')
        ready.unlink(missing_ok=True)
        repo=Path(__file__).resolve().parents[1]
        with (directory/'listener.log').open('a') as log:
            kwargs=dict(cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW
            else:kwargs['start_new_session']=True
            child=subprocess.Popen([sys.executable,'-u','-m','engine_status','--work-dir',str(root),
                '--config',str(config),'--ready-file',str(ready)],**kwargs)
        until=time.monotonic()+10
        while time.monotonic()<until:
            if child.poll() is not None:raise RuntimeError('Status listener exited; inspect '+str(directory/'listener.log'))
            if ready.exists():
                data=json.loads(ready.read_text())
                if data.get('pid')==child.pid:
                    print('status_bot=started; /check reads '+report,flush=True)
                    return dict(status='started',pid=child.pid,config=str(config))
            time.sleep(.1)
        raise RuntimeError('Status listener startup not confirmed; inspect '+str(directory/'listener.log'))
