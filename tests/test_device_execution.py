import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.test_hidden_distillation_159 import fixture
from extent.hidden_distillation_159 import DistillationStep,optimizer
from extent.device_execution import DeviceStep,AdaptiveDeviceStep,compare_proposals


def arguments():
    p,t,q,spec,tp=fixture(0);tx=optimizer(q,spec)
    return p,tx,(q,tx.init(q),p.base,tp,t,jnp.asarray(118304.,jnp.float32)),spec


def test_device_states_match_original_gradients_parameters_and_adam():
    p,tx,args,spec=arguments()
    before=jax.tree.map(np.array,args[0])
    ref=DistillationStep(p.cfg,p.tc,p.columns,spec,tx,return_gradients=True)
    device=DeviceStep(p.cfg,p.tc,p.columns,spec,tx,return_gradients=True)
    a=jax.block_until_ready(ref(*args));b=jax.block_until_ready(device(*args))
    comparison=compare_proposals(a,b,resident=True);assert comparison['passed'],comparison
    for x,y in zip(jax.tree.leaves(a[:2]),jax.tree.leaves(b[:2])):
        np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-7)
    for x,y in zip(jax.tree.leaves(before),jax.tree.leaves(args[0])):np.testing.assert_array_equal(x,y)


def test_whole_step_compilation_audit_and_next_call_resume_optimizer():
    p,tx,args,spec=arguments()
    engine=AdaptiveDeviceStep(p.cfg,p.tc,p.columns,spec,tx,mode='compiled')
    first=engine(*args)
    audit=first[2]['engine_gate']
    assert audit['checks']['resident']['passed']
    assert audit['checks']['compiled']['passed'],audit
    assert audit['selected']=='compiled' and audit['gradient_evaluations']==12
    q,state=first[:2]
    second=engine(q,state,*args[2:-1],args[-1]+1)
    from extent.decoder_recovery_157 import adam_counts
    assert adam_counts(second[1])==[2]*6
    assert bool(second[2]['finite'])
    # Teacher baseline remains usable after execution selection.
    assert engine.teacher_forward(args[2],args[3],args[4])[0].shape[:2]==args[4].shape


def test_compiled_memory_failure_falls_back_without_accepting_a_bad_proposal():
    p,tx,args,spec=arguments()
    engine=AdaptiveDeviceStep(p.cfg,p.tc,p.columns,spec,tx,mode='compiled')
    def memory(*args):raise MemoryError('synthetic memory guard')
    engine.compiled=memory
    result=engine(*args);audit=result[2]['engine_gate']
    assert audit['selected'] in ('legacy','resident') and not audit['checks']['compiled']['passed']
    assert bool(result[2]['finite'])


def test_execution_adapter_records_provenance_and_restores_process_hooks(tmp_path,monkeypatch):
    import engine_runtime
    from scripts import m3q_server_160 as server
    from scripts import m3q_corpus_recovery_160 as runner
    from extent import corpus_recovery_160 as core
    from pathlib import Path
    original_factory=core.make_kernel;previous_session=runner.session
    class Session:
        result={}
        def persist(self,**kw):pass
    monkeypatch.setattr(runner,'session',lambda *a:Session())
    session_hook=runner.session
    captured=[]
    def launch(argv):
        assert core.make_kernel is not original_factory
        value=runner.session(None)
        captured.extend(value.result['execution_history'])
        return 'ran'
    monkeypatch.setattr(server,'main',launch)
    assert engine_runtime.main(['--engine','resident','--max-wall-hours','21'])=='ran'
    assert core.make_kernel is original_factory and runner.session is session_hook
    assert captured[0]['mode']=='resident' and 'extent/device_execution.py' in captured[0]['code']
