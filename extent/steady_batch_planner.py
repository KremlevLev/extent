"""Steady-state planner: independent BEST and lean proposals, same KD/Adam."""
import gc
import numpy as np
from extent.batch_planner import benchmark


def independent_best(best):
    import jax
    # BEST is needed for durable saves, not for ordinary training. Keep exact
    # host arrays so divergence from CURRENT never adds a GPU weight copy.
    return jax.tree.map(lambda v:v if isinstance(v,np.ndarray) else np.array(jax.device_get(v),copy=True),best)


def plan_kernel(cfg,tc,columns,spec,tx,q,state,base,teacher,tokens,cursor,check,record):
    import jax
    import jax.numpy as jnp
    from extent.steady_execution import proposal_step,gradient_step
    from extent.reset_streamed_gpu import MemoryCheckedJit
    from extent.method_sweep_163 import MethodStep
    from extent.device_execution import compare_proposals
    from extent.plateau_forensics import plain
    def factory(batch):return MemoryCheckedJit(proposal_step(cfg,tc,columns,spec,tx))
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
    gradient=MemoryCheckedJit(gradient_step(cfg,tc,columns,spec,tx))
    actual_gradient,actual_health=gradient(q,state,base,teacher,tokens[:selected],cursor)
    audit=compare_proposals((None,None,health,host_gradient),(None,None,actual_health,actual_gradient))
    del host_gradient,actual_gradient,actual_health,gradient;gc.collect();jax.clear_caches()
    kernel=factory(selected)
    result['gradient_audit']=audit
    result['probe_input_presentations']=sum(r['completed_probe_proposals']*r['batch']*256 for r in result['trials'])+2*selected*256
    # Failed allocation trials may compile, but have no accepted updates.
    result['failed_probe_partial_compute_not_measured']=any(r['status']=='memory_limit' for r in result['trials'])
    result['accepted_updates']=0;record(result)
    if not audit['passed']:raise ValueError('Selected batch gradient audit failed; training not started')
    result['steady_best_buffers']='BEST retained exactly on host RAM before scan and after every quality selection'
    result['proposal_outputs']='parameters, Adam, health; gradient not retained by training executable'
    record(result)
    return kernel,result
