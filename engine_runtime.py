"""Engine core runtime with audited device execution and checkpoint continuation."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--engine',choices=('auto','resident','compiled','legacy'),default='auto')
    options,remaining=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    # Set device selection before importing any scientific/JAX modules.
    gpu=remaining[remaining.index('--gpu')+1] if '--gpu' in remaining else '0'
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if '--work-dir' not in remaining:
        remaining+=['--work-dir',str(Path(__file__).resolve().parent.parent/'engine-core')]
    from scripts.m3q_server_160 import main as launch
    if options.engine=='legacy':return launch(remaining)
    from extent import corpus_recovery_160 as math_module
    from extent.device_execution import AdaptiveDeviceStep
    from scripts import m3q_corpus_recovery_160 as runner
    original_factory=math_module.make_kernel;original_session=runner.session
    def factory(cfg,tc,columns,spec,tx,*,tpu=False):
        if tpu:raise ValueError('device execution is registered for single GPU only')
        return AdaptiveDeviceStep(cfg,tc,columns,spec,tx,mode=options.engine)
    def session(*args):
        value=original_session(*args)
        root=Path(__file__).resolve().parent
        provenance={name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                    for name in ('engine_runtime.py','extent/device_execution.py')}
        value.result.setdefault('execution_history',[]).append(dict(time_unix=time.time(),
            mode=options.engine,code=provenance,contract_policy='original source/optimizer/data contract retained; explicit operational backend extension with first-gradient gate'))
        value.persist(local_only=True);return value
    # Explicit process-local execution adapter; pinned original files and their
    # source checks remain intact, supplemented by execution_history + engine_gate.
    math_module.make_kernel=factory;runner.session=session
    try:return launch(remaining)
    finally:math_module.make_kernel=original_factory;runner.session=original_session


if __name__=='__main__':main()
