"""Read-only current-state profiling worker; parent owns the time limit/lock."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',required=True);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--batches',default='1,2,4,8,16');p.add_argument('--gpu',default='0')
    a=p.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    import jax
    root=Path(a.work_dir).resolve();output=root/'results/runtime-capacity.json'
    source=root/'results/runtime-report.json';parent=json.loads(source.read_text())
    from extent.capacity_probe import measure_batches,ProbeFinished
    from extent.hidden_distillation_159 import DistillationStep
    from extent.recovery_durability import RecoveryCheckpointStore
    from extent.broad_corpus import PackedCorpus
    from scripts import m3q_corpus_recovery_160 as runner
    from extent import corpus_recovery_160 as core
    from extent.hf_artifact_sync import artifact_config_from_env
    from extent.campaign_checkpoint import write_json_atomic
    os.environ['EXTENT_HF_DOWNLOAD_DEADLINE_UNIX']=str(a.deadline)
    # Persistent executable cache can amortize future probes; no parameters stored.
    jax.config.update('jax_compilation_cache_dir',str(root/'compile-cache'))
    if artifact_config_from_env(include_kaggle_secrets=False) is None:raise ValueError('HF environment credentials required')
    batches=[int(v) for v in a.batches.split(',')]
    corpus=PackedCorpus(root/'data-pack');session_holder={}
    class Session:
        def __init__(self,args,contract,hub):
            if contract!=parent['contract']:raise ValueError('current parent contract differs; probe cannot reinterpret state')
            self.result=copy.deepcopy(parent);self.result['status']='preflight';self.result['pending_slots']=[]
            self.result['cost']=dict(attempted_updates=0,input_tokens=0,ce_targets=0,teacher_input_tokens=0)
            self.store=RecoveryCheckpointStore(root/'states/runtime-report','read-only',None)
            self.wall=time.monotonic()+max(0,a.deadline-time.time());self.deadline=self.wall;self.finished=False
        def check(self):
            if self.finished or time.time()>=a.deadline:raise ProbeFinished('read-only probe finished/deadline')
        def checkpoint(self,*args):pass # NEVER write a training checkpoint
        def emit(self,*args):pass
        def persist(self,**kwargs):return True # NEVER upload or overwrite parent report
    def session(*args):
        value=Session(*args);session_holder['value']=value;return value
    class Kernel:
        def __init__(self,cfg,tc,columns,spec,tx):
            self.parameters=(cfg,tc,columns,spec,tx)
            self.teacher=DistillationStep(cfg,tc,columns,spec,tx);self.check=lambda:None
        def teacher_forward(self,*args):
            self.teacher.check=self.check;return self.teacher.teacher_forward(*args)
        def __call__(self,*args):
            cfg,tc,columns,spec,tx=self.parameters
            # Always profile fresh consecutive C4 windows; weights/Adam belong
            # to the reported profiled arm. Domain/value variation is disclosed.
            offset=max(0,int(args[-1])-118304)
            def windows(batch):
                return __import__('numpy').concatenate([corpus.batch(offset+i) for i in range(batch)],axis=0)
            try:measure_batches(cfg,tc,columns,spec,tx,args,windows,output,batches,a.deadline,self.check)
            finally:session_holder['value'].finished=True
            raise ProbeFinished('batch profile saved; zero accepted updates')
    old_session,old_factory=runner.session,core.make_kernel
    runner.session=session;core.make_kernel=lambda cfg,tc,col,spec,tx,**kw:Kernel(cfg,tc,col,spec,tx)
    options=runner.parser().parse_args(['--execution','server-balanced','--corpus-dir',str(root/'data-pack'),
        '--max-wall-hours','1','--max-steps',str(parent['contract']['horizon']),
        '--cache-dir',str(root/'cache'),'--checkpoint-dir',str(root/'states'),
        '--output-dir',str(root/'results'),'--no-telegram'])
    try:
        result=runner.run(options)
        if result.get('status')=='failed':raise RuntimeError('probe failed before measurements')
    finally:runner.session=old_session;core.make_kernel=old_factory


if __name__=='__main__':main()
