"""Mandatory physical batch1..8 benchmark; probe proposals are never accepted."""
import gc,time
import numpy as np


def memory_error(exc):
    return isinstance(exc,MemoryError) or any(x in str(exc).lower() for x in ('resource_exhausted','out of memory'))


def benchmark(factory,invoke,release,check=lambda:None,record=lambda value:None,clock=time.monotonic):
    trials=[];start=clock()
    # Release executable caches between shapes: an earlier large allocation must
    # not decide whether a later, smaller batch fits.
    for batch in range(8,0,-1):
        check();row=dict(batch=batch,status='testing',completed_probe_proposals=0);engine=None;begun=clock()
        try:
            engine=factory(batch)
            invoke(engine,batch)  # compilation + warm-up; proposals discarded
            row['completed_probe_proposals']+=1
            row['compile_warmup_seconds']=clock()-begun
            samples=[]
            for _ in range(2):
                check();t=clock();invoke(engine,batch);samples.append(max(1e-6,clock()-t));row['completed_probe_proposals']+=1
            seconds=float(np.median(samples))
            row.update(status='measured',samples_seconds=samples,seconds_per_update=seconds,
                input_tokens_per_second=batch*256/seconds)
        except Exception as exc:
            if not memory_error(exc):raise
            row.update(status='memory_limit',error_type=type(exc).__name__)
        finally:
            engine=None;release();gc.collect()
            row['elapsed_seconds']=clock()-begun;trials.append(row);record(dict(trials=trials,elapsed_seconds=clock()-start))
    valid=[r for r in trials if r['status']=='measured']
    if not valid:raise RuntimeError('No compiled batch1..8 fits; refusing silent slow fallback')
    winner=max(valid,key=lambda r:(r['input_tokens_per_second'],-r['batch']))
    result=dict(status='completed',trials=trials,selected_batch=winner['batch'],
        input_tokens_per_second=winner['input_tokens_per_second'],elapsed_seconds=clock()-start,
        accepted_updates=0,scope='fixed actual CURRENT/Adam; compilation excluded from steady samples; no state/data advance')
    record(result);return result


def plan_kernel(cfg,tc,columns,spec,tx,q,state,base,teacher,tokens,cursor,check,record):
    import jax
    import jax.numpy as jnp
    from extent.method_execution_163 import compiled_step
    from extent.reset_streamed_gpu import MemoryCheckedJit
    from extent.method_sweep_163 import MethodStep
    from extent.device_execution import compare_proposals
    from extent.plateau_forensics import plain
    def factory(batch):return MemoryCheckedJit(compiled_step(cfg,tc,columns,spec,tx))
    def invoke(engine,batch):
        result=engine(q,state,base,teacher,tokens[:batch],cursor)
        if not bool(result[2]['finite']):raise FloatingPointError('nonfinite capacity probe')
        del result
    def release():jax.clear_caches()
    result=benchmark(factory,invoke,release,check,record)
    selected=result['selected_batch'];check()
    # Streamed fixed-q reference only for the selected batch, not 8 discarded
    # sets of Adam work. Move its gradient to host before compiling the winner.
    reference=MethodStep(cfg,tc,columns,spec,tx);reference.check=check
    r=reference.accumulated(q,state,base,teacher,tokens[:selected],cursor)
    if not bool(r[2]['finite']):raise FloatingPointError('nonfinite planner reference')
    host_gradient=jax.tree.map(lambda v:np.asarray(jax.device_get(v)),r[3])
    health=plain(r[2]);del r,reference;gc.collect();jax.clear_caches();check()
    kernel=factory(selected)
    actual=kernel(q,state,base,teacher,tokens[:selected],cursor)
    audit=compare_proposals((None,None,health,host_gradient),actual)
    del host_gradient,actual;gc.collect()
    result['gradient_audit']=audit
    result['probe_input_presentations']=sum(r['completed_probe_proposals']*r['batch']*256 for r in result['trials'])+2*selected*256
    # Failed allocation trials may compile, but have no accepted updates.
    result['failed_probe_partial_compute_not_measured']=any(r['status']=='memory_limit' for r in result['trials'])
    result['accepted_updates']=0;record(result)
    if not audit['passed']:raise ValueError('Selected batch gradient audit failed; training not started')
    return kernel,result
