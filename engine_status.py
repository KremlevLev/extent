"""One read-only Telegram command listener for multiple registered sessions."""
import argparse
import email.utils
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import tempfile
import threading
import time
from urllib import error, parse, request

from engine_dashboard import number, points, read_json, trainer_active, write_json


class TransportError(Exception):
    def __init__(self, code=0, retry_after=0):
        self.code=code;self.retry_after=retry_after
        super().__init__('HTTP '+str(code) if code else 'network/invalid response')


def api(token, method, payload=None, timeout=40):
    """Never include credential-bearing URL or server text in an exception."""
    req=request.Request('https://api.telegram.org/bot'+token+'/'+method,
        data=parse.urlencode(payload or {}).encode(),method='POST')
    try:
        with request.urlopen(req,timeout=timeout) as response:
            data=json.load(response)
    except error.HTTPError as exc:
        retry=0
        try:retry=json.loads(exc.read()).get('parameters',{}).get('retry_after',0)
        except (ValueError,AttributeError):pass
        raise TransportError(exc.code,retry) from None
    except (OSError,ValueError):raise TransportError() from None
    if not data.get('ok'):
        raise TransportError(data.get('error_code',0),data.get('parameters',{}).get('retry_after',0))
    return data['result']


def sessions(config, root):
    if not config:
        return [dict(id='blackwell',work_dir=str(root),report='results/runtime-b6.json')]
    rows=json.loads(Path(config).read_text(encoding='utf-8')).get('sessions',[])
    if not rows or len(rows)>20:raise ValueError('config requires 1..20 sessions')
    ids=set()
    for row in rows:
        name=row.get('id','')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}',name) or name.lower() in ids:
            raise ValueError('session ids must be unique short aliases')
        ids.add(name.lower())
        if ('work_dir' in row)==('hf_file' in row):raise ValueError('choose work_dir OR hf_file')
        if 'hf_file' in row:
            if row.get('repo_type','dataset') not in ('dataset','model'):raise ValueError('invalid repo_type')
            if not re.fullmatch(r'[\w.-]+/[\w.-]+',row.get('repo_id','')):raise ValueError('repo_id required')
            if '..' in row['hf_file'].split('/') or row['hf_file'].startswith('/'):
                raise ValueError('hf_file must be repository relative')
    return rows


class Reports:
    def __init__(self):self.cache={}

    def get(self, row):
        if 'work_dir' in row:
            root=Path(row['work_dir']).expanduser()
            path=root/row.get('report','results/runtime-b6.json')
            report=read_json(path)
            if not report:raise ValueError('report unavailable')
            return report,path.stat().st_mtime,trainer_active(root),False
        # HF is read only, fetched only on request, at most once a minute.
        key=json.dumps(row,sort_keys=True);old=self.cache.get(key)
        if old and time.time()-old[0]<60:return old[1],old[2],None,False
        prefix='datasets/' if row.get('repo_type','dataset')=='dataset' else ''
        url='https://huggingface.co/'+prefix+row['repo_id']+'/resolve/main/'+parse.quote(row['hf_file'],safe='/')
        headers={'Cache-Control':'no-cache'}
        if os.environ.get('HF_TOKEN'):headers['Authorization']='Bearer '+os.environ['HF_TOKEN']
        try:
            with request.urlopen(request.Request(url,headers=headers),timeout=12) as response:
                raw=response.read(64*1024*1024+1)
                if len(raw)>64*1024*1024:raise ValueError('report exceeds size limit')
                report=json.loads(raw)
                stamp=response.headers.get('Last-Modified')
                modified=email.utils.parsedate_to_datetime(stamp).timestamp() if stamp else None
            self.cache[key]=(time.time(),report,modified)
            return report,modified,None,False
        except (OSError,ValueError):
            if old:return old[1],old[2],None,True
            raise ValueError('remote report unavailable') from None


def short(value, places=3):return f'{value:.{places}f}' if number(value) else '—'


def format_report(alias,report,modified,active=None,cached=False,now=None):
    now=time.time() if now is None else now
    contract=report.get('contract',{});exp=contract.get('experiment','?')
    lines=[f'{alias} · EXP-{exp}',f'Статус в отчёте: {report.get("status","unknown")}']
    if active is not None:lines.append('Процесс: '+('держит lock' if active else 'не держит lock'))
    else:lines.append('Удалённый снимок; работает ли процесс сейчас — неизвестно')
    if number(modified):
        age=max(0,now-modified)
        lines.append(f'Возраст файла: {age/60:.1f} мин'+(' ⚠ старше 15 мин' if age>900 else ''))
    else:lines.append('Дата изменения удалённого файла недоступна')
    if cached:lines.append('⚠ HF недоступен: показан предыдущий снимок')
    branches=report.get('branches',{})
    if not branches:lines.append('Ветки ещё не появились: подготовка или другой формат отчёта')
    for arm,row in branches.items():
        lines.append('\n'+str(arm))
        step=row.get('step');origin=row.get('source_step')
        lines.append(f'Шаг: {step if step is not None else "—"}')
        if number(step) and number(origin) and step>=origin and number(contract.get('batch')):
            tokens=(step-origin)*contract['batch']*256
            lines.append(f'Новые входные токены этой ветки: {tokens/1e6:.3f} млн')
        history=row.get('history',[])
        if history:
            last=history[-1]
            lines.append('Train CE / KL / loss: '+ ' / '.join(short(last.get(k)) for k in ('ce','kl','loss')))
        # Same extractor as the read-only dashboard: never inject BEST as CURRENT.
        if number(origin):
            metric_points=points(report,arm)
            for panel in ('wiki_A','wiki_B','wiki_C','c4_D'):
                key='validation/'+panel+'_nll'
                measured=[p for p in metric_points if key in p]
                if not measured:continue
                last=measured[-1]
                previous=measured[-2] if len(measured)>1 else None
                delta=f' ({last[key]-previous[key]:+.4f})' if previous else ''
                lines.append(f'NLL {panel}: {last[key]:.4f}{delta} @ {last["progress/global_step"]}')
            fresh={}
            for point in metric_points:
                for key,value in point.items():
                    if key.startswith(('validation/en_stories_','validation/ru_wiki_')):
                        fresh[key]=(value,point['progress/global_step'])
            for key,(value,measured_step) in fresh.items():
                lines.append(f'NLL {key.removeprefix("validation/")}: {value:.4f} @ {measured_step}')
            timings=[p.get('performance/input_tokens_per_second') for p in metric_points
                     if 'performance/input_tokens_per_second' in p]
            if timings:lines.append(f'Скорость ядра: {sum(timings[-20:])/len(timings[-20:]):.0f} ток/с')
        if number(row.get('best_nll')):lines.append(f'BEST Wiki A: {row["best_nll"]:.4f} @ {row.get("best_step","—")}')
        if row.get('failed'):lines.append('⚠ Ветка отмечена failed')
        if row.get('durable_step') is not None:lines.append(f'Шаг удалённого сохранения: {row["durable_step"]}')
    lines.append(f'Ожидают HF: {len(report.get("pending_slots",[]))}')
    if report.get('report_sync_error'):lines.append('⚠ Ошибка синхронизации отчёта')
    # Existing runtime reports lack wall-clock start timestamps: do not invent ETA.
    if number(report.get('launcher_elapsed_hours')):lines.append(f'Время запуска: {report["launcher_elapsed_hours"]:.2f} ч')
    return '\n'.join(lines)


HELP='/check — все сессии\n/check blackwell — одна сессия\n/sessions — названия сессий\n/help — команды'


class Commands:
    def __init__(self, rows, chat_id, reports=None):
        self.rows=rows;self.chat_id=str(chat_id);self.reports=reports or Reports()

    def reply(self, message):
        # Private chat only. Matching chat ID alone is insufficient in a group.
        chat=message.get('chat',{})
        if str(chat.get('id'))!=self.chat_id or chat.get('type')!='private':return []
        text=message.get('text','').strip();parts=text.split(maxsplit=1)
        if not parts:return []
        cmd=parts[0].split('@')[0].lower()
        if cmd in ('/help','/start'):return [HELP]
        if cmd=='/sessions':return ['Сессии:\n'+'\n'.join('/check '+r['id'] for r in self.rows)]
        if cmd!='/check':return []
        selected=self.rows
        if len(parts)>1:
            selected=[r for r in self.rows if r['id'].lower()==parts[1].lower()]
            if not selected:return ['Нет такой сессии. Названия: /sessions']
        replies=[]
        for row in selected:
            try:replies.append(format_report(row['id'],*self.reports.get(row)))
            except (OSError,ValueError,TypeError,KeyError,AttributeError):
                replies.append(row['id']+': отчёт пока недоступен или его формат не поддерживается')
        return replies


def chunks(text,limit=3500):
    while text:
        cut=text.rfind('\n',0,limit) if len(text)>limit else len(text)
        if cut<=0:cut=limit
        yield text[:cut];text=text[cut:].lstrip('\n')


def poll(token,handler,state_path,stop,call=api):
    state=read_json(state_path) or {};offset=int(state.get('offset',0));backoff=2
    verified=False
    print('Command listener ready: /check /sessions; no automatic messages',flush=True)
    while not stop.is_set():
        try:
            if not verified:
                if call(token,'getWebhookInfo').get('url'):
                    raise RuntimeError('Bot already has a webhook; not changed. Integrate /check with that bot instead.')
                verified=True
            updates=call(token,'getUpdates',dict(offset=offset,timeout=25,allowed_updates='["message"]'))
            for update in updates:
                message=update.get('message',{})
                # Do not replay forgotten commands from days ago after downtime.
                if time.time()-message.get('date',0)<300:
                    for reply in handler.reply(message):
                        for part in chunks(reply):call(token,'sendMessage',dict(chat_id=handler.chat_id,text=part))
                offset=int(update['update_id'])+1
                write_json(state_path,dict(offset=offset))
            backoff=2
        except TransportError as exc:
            if exc.code in (401,403,409):
                raise RuntimeError('Telegram authentication/access/conflicting poller; HTTP '+str(exc.code)) from None
            delay=max(backoff,min(float(exc.retry_after or 0),3600))+random.uniform(0,1)
            print(f'Telegram temporarily unavailable (HTTP {exc.code}); retrying',flush=True)
            stop.wait(delay);backoff=min(backoff*2,60)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',default=str(Path(__file__).resolve().parent.parent/'engine-core'))
    p.add_argument('--config',help='JSON session registry; re-read on listener restart')
    p.add_argument('--preview',action='store_true',help='print summary only; no Telegram access')
    a=p.parse_args(argv);root=Path(a.work_dir).expanduser().resolve()
    rows=sessions(a.config,root)
    if a.preview:
        handler=Commands(rows,'preview')
        for text in handler.reply(dict(chat=dict(id='preview',type='private'),text='/check')):print(text+'\n')
        return
    token=os.environ.get('TELEGRAM_BOT_TOKEN')
    chat=os.environ.get('TELEGRAM_CHAT_ID') or os.environ.get('TELEGRAM_USER_ID')
    if not token or not chat or not str(chat).isdigit():
        raise ValueError('Set TELEGRAM_BOT_TOKEN and positive private TELEGRAM_CHAT_ID (or TELEGRAM_USER_ID)')
    from filelock import FileLock
    # One listener for this bot per user/host, even with different work directories.
    lock=Path(tempfile.gettempdir())/('engine-status-'+hashlib.sha256(token.encode()).hexdigest()[:20]+'.lock')
    directory=root/'status-bot';directory.mkdir(parents=True,exist_ok=True)
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *unused:stop.set())
    with FileLock(str(lock),timeout=0):poll(token,Commands(rows,chat),directory/'cursor.json',stop)


if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError) as exc:
        # These messages are constructed locally; transport errors never print URLs.
        raise SystemExit(str(exc)) from None
