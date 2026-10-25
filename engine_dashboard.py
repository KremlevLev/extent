"""Read-only W&B mirror of the engine report; no imports of training/JAX code."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


PANELS={'A':'wiki_A','B':'wiki_B','C':'wiki_C','D':'c4_D'}


def number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return None


def points(report,arm):
    row=report['branches'][arm];c=report['contract'];origin=int(row['source_step'])
    batch=int(c.get('batch',1));length=int(c.get('model',{}).get('sequence_length',256))
    # Current contracts store model context separately; all exported161 inputs
    # have verified length256. Validation targets_per_window is unrelated to B.
    if c.get('batch')==6:length=256
    values={}
    def at(step):
        step=int(step)
        if step<origin:return None
        return values.setdefault(step,dict({'progress/global_step':step,
            'progress/optimizer_updates':step-origin,
            'progress/new_input_tokens':(step-origin)*batch*length}))
    def evaluation(point):
        data=at(point['step'])
        if data is None:return
        for panel,item in point.get('panels',{}).items():
            if panel in PANELS and number(item.get('nll')):data[f'validation/{PANELS[panel]}_nll']=item['nll']
        for key,item in point.get('fresh_nll',{}).items():
            count=len(item.get('windows',[]))
            label={'english_stories':'en_stories','russian_wiki':'ru_wiki'}.get(key,key)
            if number(item.get('nll')):data[f'validation/{label}_{count or "unknown"}windows_nll']=item['nll']
    if row.get('source_controls'):evaluation(dict(step=origin,panels=row['source_controls']))
    for item in row.get('history',[]):
        data=at(item['step'])
        if data is None:continue
        for key in ('loss','ce','kl','grad_max_abs','log10_grad_norm'):
            if number(item.get(key)):data['train/'+key]=item[key]
        if number(item.get('seconds')) and item['seconds']>0 and 'engine_gate' not in item:
            data['performance/update_seconds']=item['seconds']
            data['performance/input_tokens_per_second']=batch*length/item['seconds']
    for item in row.get('evaluations',[]):evaluation(item)
    # LAST is the current model. Historical BEST must not become a second
    # current trajectory at the same x coordinate.
    final=row.get('final',{}).get('last')
    if final:evaluation(final)
    return [values[k] for k in sorted(values)]


def run_identity(report,arm,project,entity):
    c=report['contract'];recovery=report['branches'][arm].get('rollbacks',[])
    raw=json.dumps([project,entity,c.get('protocol'),c.get('parent_plan_sha'),arm,recovery],sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def gpu_snapshot(gpu):
    try:
        out=subprocess.run(['nvidia-smi','--id',gpu,
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
        cells=out.stdout.strip().splitlines()[0].split(',')
        keys=('utilization_percent','memory_used_mib','memory_total_mib','temperature_c','power_watts')
        result={'gpu/'+k:float(v.strip()) for k,v in zip(keys,cells) if v.strip() not in ('N/A','[N/A]')}
        for field in ('memory_used','memory_total'):
            if 'gpu/'+field+'_mib' in result:result['gpu/'+field+'_gib']=result.pop('gpu/'+field+'_mib')/1024
        return result
    except (OSError,subprocess.SubprocessError,ValueError,IndexError):return {}


def trainer_active(root):
    """Linux read-only lock lookup; never acquire the trainer's exclusive lock."""
    try:
        stat=(root/'server.lock').stat()
        target=(os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino)
        lines=Path('/proc/locks').read_text().splitlines()
    except (OSError,AttributeError):return None
    for line in lines:
        cells=line.split()
        if not any(kind in cells for kind in ('FLOCK','POSIX','OFDLCK')):continue
        for cell in cells:
            parts=cell.split(':')
            if len(parts)!=3:continue
            try:found=(int(parts[0],16),int(parts[1],16),int(parts[2]))
            except ValueError:continue
            if found==target:return True
    return False


class Mirror:
    def __init__(self,sdk,directory,project,entity=None,mode='online'):
        if getattr(sdk,'__name__',None)=='wandb':
            # Pinned0.31 optional SDK usage analytics is separate from Run.log.
            # Avoid unrelated OTLP collectors, including offline analytics calls.
            try:
                from wandb.analytics.opentelemetry.opentelemetry_proxy import disable
                disable()
            except ImportError:pass
        self.sdk=sdk;self.directory=directory;directory.mkdir(parents=True,exist_ok=True)
        self.project=project;self.entity=entity;self.mode=mode;self.runs={}
        self.state_path=directory/'mirror-state.json';self.state=read_json(self.state_path) or {'runs':{}}
        self.started=self.state.setdefault('started_unix',time.time())

    def run(self,report,arm):
        key=run_identity(report,arm,self.project,self.entity)
        if key not in self.runs:
            row=report['branches'][arm];c=report['contract']
            label='C4 fresh' if 'C4' in arm else 'Wiki control'
            segment=len(row.get('rollbacks',[]));label+=f' / b{c.get("batch",1)}'+(f' / recovery{segment}' if segment else '')
            settings=self.sdk.Settings(console='off',disable_git=True,save_code=False,
                x_disable_stats=True,x_disable_meta=True,init_timeout=30,finish_timeout=10,silent=True)
            run=self.sdk.init(project=self.project,entity=self.entity,id=key,resume='allow',reinit='create_new',
                name=label,group=f"experiment-{c.get('experiment','unknown')}",mode=self.mode,
                dir=str(self.directory),settings=settings,
                config=dict(experiment=c.get('experiment'),batch=c.get('batch',1),arm=arm,sequence_length=256,
                    loss='.5CE+.5KL',observer_only=True,token_axis_scope='new inputs in this fork, excludes parent/preparation',
                    gpu_metrics_scope='whole selected GPU including other processes'))
            run.define_metric('progress/new_input_tokens')
            run.define_metric('progress/*',hidden=True)
            for name in ('train/*','performance/*','validation/*'):
                run.define_metric(name,step_metric='progress/new_input_tokens')
            run.define_metric('runtime/elapsed_hours')
            for name in ('gpu/*','runtime/report_age_seconds','storage/*'):
                run.define_metric(name,step_metric='runtime/elapsed_hours')
            run.define_metric('gpu/memory_total_gib',hidden=True)
            run.define_metric('performance/update_seconds',hidden=True)
            self.runs[key]=run
            self.state['runs'].setdefault(key,{'points':{}})
            self.state['runs'][key]['url']=getattr(run,'url',None)
            print(label,':',getattr(run,'url',None) or '(offline)',flush=True)
            write_json(self.directory/'links.json',{k:v.get('url') for k,v in self.state['runs'].items()})
        return key,self.runs[key]

    def update(self,report,age=0.,gpu=None,active=None):
        for arm,row in report.get('branches',{}).items():
            key,run=self.run(report,arm);seen=self.state['runs'][key]['points']
            for point in points(report,arm):
                step=str(point['progress/global_step']);old=seen.setdefault(step,{})
                changes={k:v for k,v in point.items() if not k.startswith('progress/') and old.get(k)!=v}
                if changes:
                    run.log(dict(changes,**{k:v for k,v in point.items() if k.startswith('progress/')}))
                    old.update(changes)
            summary=dict(trainer_status=report.get('status','unknown'),global_step=row.get('step'),
                best_wiki_A_nll=row.get('best_nll'),best_global_step=row.get('best_step'),
                last_remote_step=row.get('durable_step'),hf_pending_slots=len(report.get('pending_slots',[])),
                observer_only=True,trainer_process_active=active)
            if row.get('execution_checks'):summary['execution_engine']=row['execution_checks'][-1].get('selected')
            if row.get('failed'):summary['branch_failed']=True
            if isinstance(report.get('error'),dict):summary['trainer_error_type']=report['error'].get('type')
            if 'stop_reason' in report:summary['trainer_stop_reason']=report['stop_reason']
            run.summary.update({k:v for k,v in summary.items() if v is not None})
            heartbeat={'runtime/elapsed_hours':(time.time()-self.started)/3600,
                'runtime/report_age_seconds':age,'storage/hf_pending_slots':len(report.get('pending_slots',[])),
                'storage/report_sync_error':int(bool(report.get('report_sync_error')))}
            heartbeat.update(gpu or {})
            run.log(heartbeat)
        write_json(self.state_path,self.state)

    def finish(self):
        for run in self.runs.values():
            try:run.finish()
            except Exception as exc:print('Dashboard finish:',type(exc).__name__,flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default=str(Path(__file__).resolve().parent.parent/'engine-core'))
    p.add_argument('--project',default=os.environ.get('WANDB_PROJECT','engine-core'))
    p.add_argument('--entity',default=os.environ.get('WANDB_ENTITY') or None)
    p.add_argument('--mode',choices=('online','offline'),default='online')
    p.add_argument('--interval',type=float,default=15.)
    p.add_argument('--once',action='store_true')
    p.add_argument('--gpu',default='0')
    p.add_argument('--gpu-stats',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args(argv)
    if not number(a.interval) or a.interval<5:raise ValueError('interval must be >=5 seconds')
    # SDK cannot auto-upload the checkout, console or machine metadata. No weights,
    # samples, environment dictionaries or complete report are ever passed to it.
    os.environ['WANDB_DISABLE_CODE']='true'
    os.environ['WANDB_CONSOLE']='off'
    import wandb
    root=Path(a.work_dir).expanduser().resolve();report_path=root/'results/runtime-b6.json'
    dashboard=root/'dashboard';dashboard.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    with FileLock(str(dashboard/'observer.lock'),timeout=0):
        mirror=Mirror(wandb,dashboard,a.project,a.entity,a.mode);stop=[False]
        def request_stop(*args):stop[0]=True
        old={sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            while not stop[0]:
                report=read_json(report_path)
                if report and report.get('contract'):
                    active=trainer_active(root)
                    mirror.update(report,max(0,time.time()-report_path.stat().st_mtime),
                                  gpu_snapshot(a.gpu) if a.gpu_stats else None,active)
                    final=read_json(root/'runtime-b6-status.json')
                    if a.once:break
                    if active is False and final and final.get('sessions')==report.get('sessions') and final.get('status')==report.get('status'):
                        print('Trainer finished; dashboard mirror synced final report',flush=True);break
                elif a.once:raise FileNotFoundError('runtime-b6.json not available yet')
                time.sleep(a.interval)
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
            mirror.finish()


if __name__=='__main__':main()
