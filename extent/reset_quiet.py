"""Hard per-invocation Telegram cap. Children never send Telegram messages."""
import time
from scripts.qwen_extended_horizon_campaign import _safe_notify

class ParentNotifier:
    def __init__(self,enabled,experiment,*,clock=time.monotonic,sender=_safe_notify):
        self.enabled=enabled;self.experiment=experiment;self.clock=clock;self.sender=sender
        self.started=clock();self.last_progress=self.started;self.seen=set();self.events=[]
    def emit(self,kind,message):
        now=self.clock()
        if kind not in ('started','progress','failure','finished'):return False
        if kind!='progress' and kind in self.seen:return False
        if kind=='progress' and now-self.last_progress<14400:return False
        if len(self.events)>=(5 if kind=='finished' else 4):return False
        self.seen.add(kind)
        if kind=='progress':self.last_progress=now
        result=self.sender(self.enabled,f'Extent EXP{self.experiment}: {message}')
        self.events.append(dict(kind=kind,elapsed_seconds=now-self.started,result=result))
        return True
