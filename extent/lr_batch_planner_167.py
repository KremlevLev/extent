"""Actual-batch layer-streamed full decoder planner, with complete memory errors."""
import gc,time
import numpy as np
from extent.batch_planner import memory_error
from extent.steady_batch_planner import independent_best


def memory_snapshot():
    import jax,os
    from extent.plateau_forensics import plain
    from engine_dashboard import gpu_snapshot
    return dict(allocator=[dict(device=d.id,platform=d.platform,stats=plain(d.memory_stats() or {})) for d in jax.devices()],
        device_snapshot=gpu_snapshot(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]))


def benchmark(factory,invoke,release,check=lambda:None,record=lambda value:None,clock=time.monotonic,snapshot=memory_snapshot):
    trials=[];start=clock()
    for batch in range(1,9):
        check();row=dict(batch=batch,status='testing',completed_probe_proposals=0,memory_before=snapshot());engine=None;begun=clock()
        try:
            engine=factory(batch);invoke(engine,batch);row['completed_probe_proposals']+=1
            row['compile_warmup_seconds']=clock()-begun;samples=[]
            for _ in range(2):
                check();t=clock();invoke(engine,batch);samples.append(max(1e-6,clock()-t));row['completed_probe_proposals']+=1
            median=float(np.median(samples));row.update(status='measured',samples_seconds=samples,
                seconds_per_update=median,input_tokens_per_second=batch*256/median)
        except Exception as exc:
            row.update(status='memory_limit' if memory_error(exc) else 'failed',error_type=type(exc).__name__,error_message=str(exc)[:8192])
            if not memory_error(exc):raise
            print('Engine stream: memory trial',batch,row['error_type'],row['error_message'],flush=True)
        finally:
            engine=None;release();gc.collect();row['memory_after']=snapshot();row['elapsed_seconds']=clock()-begun
            trials.append(row);record(dict(trials=trials,elapsed_seconds=clock()-start,execution='batched-layer-streamed'))
    valid=[r for r in trials if r['status']=='measured']
    if not valid:raise RuntimeError('No streamed batch1..8 fits; inspect saved error_message and allocator stats')
    matched=[r for r in valid if r['batch']==8]
    if not matched:raise RuntimeError('Matched LR comparison requires physical batch8; no fallback or batch confound')
    best=matched[0]
    result=dict(status='completed',execution='batched-layer-streamed',trials=trials,selected_batch=best['batch'],
        input_tokens_per_second=best['input_tokens_per_second'],elapsed_seconds=clock()-start,accepted_updates=0)
    record(result);return result


def plan_kernel(cfg,tc,columns,spec,tx,q,state,base,teacher,tokens,cursor,check,record):
    import jax
    from extent.method_sweep_163 import MethodReference,MethodStep
    from extent.device_execution import compare_proposals
    from extent.plateau_forensics import plain
    def factory(batch):
        engine=MethodReference(cfg,tc,columns,spec,tx,return_gradients=False);engine.check=check;return engine
    def invoke(engine,batch):
        out=jax.block_until_ready(engine(q,state,base,teacher,tokens[:batch],cursor))
        if not bool(out[2]['finite']):raise FloatingPointError('nonfinite streamed batch probe')
        del out
    result=benchmark(factory,invoke,lambda:jax.clear_caches(),check,record);selected=result['selected_batch'];check()
    # One whole-batch gradient and Adam proposal; compare to mean of single-
    # example gradients at fixed q, not to sequential optimizer updates.
    engine=factory(selected);engine.return_gradients=True
    candidate=jax.block_until_ready(engine(q,state,base,teacher,tokens[:selected],cursor))
    health=plain(candidate[2]);gradient=jax.tree.map(lambda v:np.asarray(jax.device_get(v)),candidate[3]);del candidate
    reference=MethodStep(cfg,tc,columns,spec,tx);reference.check=check
    actual=jax.block_until_ready(reference.accumulated(q,state,base,teacher,tokens[:selected],cursor))
    audit=compare_proposals((None,None,health,gradient),actual)
    del actual,reference,gradient;gc.collect();engine.return_gradients=False
    result.update(gradient_audit=audit,accepted_updates=0,
        probe_input_presentations=sum(r['completed_probe_proposals']*r['batch']*256 for r in result['trials'])+2*selected*256,
        failed_probe_partial_compute_not_measured=any(r['status']=='memory_limit' for r in result['trials']),
        scope='whole physical batch per layer; host activation staging; ONE full Adam update; original q/Adam/data not advanced by probes')
    record(result)
    if not audit['passed']:raise ValueError('Streamed actual-batch gradient audit failed')
    return engine,result
