"""Short, non-mutating batch throughput probes, with incremental results."""
import time
import jax
import numpy as np
from extent.device_execution import compiled_step
from extent.reset_streamed_gpu import MemoryCheckedJit
from extent.campaign_checkpoint import write_json_atomic


def recommend(report):
    valid=[t for t in report.get('trials',[]) if t.get('status')=='measured' and t.get('finite')]
    if not valid:return None
    best=max(t['input_tokens_per_second'] for t in valid)
    # Within3% choose the smaller batch: less memory for negligible speed loss.
    return min((t for t in valid if t['input_tokens_per_second']>=.97*best),key=lambda t:t['batch'])['batch']


class ProbeFinished(TimeoutError):pass


def measure_batches(cfg,tc,columns,spec,tx,args,window_provider,path,candidates,deadline,check=lambda:None):
    """Same input q/Adam for every unaccepted proposal; compile/warmup excluded from TPS.

    Parent process enforces the wall deadline even during uninterruptible XLA compile.
    JAX's allocator high-water mark is cumulative, not isolated per-trial NVML memory.
    """
    q,state,base,tp,original,cursor=args
    device=next(iter(original.devices()))
    report=dict(status='profiling',trials=[],profiled_arm=spec['name'],source_global_step=int(cursor),
                device_kind=device.device_kind,
                sequence_length=original.shape[1],accepted_updates=0,
                inputs='consecutive C4 windows at reported cursor; profiled arm parameters may be Wiki or C4 branch',
                memory_scope='JAX process allocator; cumulative high-water includes initialization, not total GPU utilization',
                effective_batch_change_requires_new_training_contract=True)
    def save():
        report['recommended_batch']=recommend(report);write_json_atomic(path,report)
    save()
    for batch in candidates:
        check()
        if time.time()>=deadline:break
        row=dict(batch=batch,status='compiling',started_unix=time.time());report['trials'].append(row);save()
        engine=MemoryCheckedJit(compiled_step(cfg,tc,columns,spec,tx))
        tokens=jax.device_put(window_provider(batch),original.sharding)
        before=device.memory_stats() or {}
        try:
            begun=time.perf_counter()
            value=jax.block_until_ready(engine(q,state,base,tp,tokens,cursor))
            row['compile_and_warmup_seconds']=time.perf_counter()-begun
            if not bool(value[2]['finite']):row.update(status='nonfinite',finite=False);save();continue
            row['objective']={k:float(value[2][k]) for k in ('loss','ce','kl')}
            # No intermediate proposal is fed to the next call.
            del value
            samples=[]
            for _ in range(3):
                check()
                if time.time()>=deadline:break
                begun=time.perf_counter()
                value=jax.block_until_ready(engine(q,state,base,tp,tokens,cursor))
                samples.append(time.perf_counter()-begun)
                if not bool(value[2]['finite']):raise FloatingPointError('nonfinite timing proposal')
                del value
            if len(samples)<2:
                row.update(status='insufficient_timing_samples',samples_seconds=samples);save();break
            seconds=float(np.median(samples));after=device.memory_stats() or {}
            row.update(status='measured',finite=True,samples_seconds=samples,seconds_per_update=seconds,
                input_tokens_per_second=tokens.size/seconds,next_token_targets_per_second=batch*(tokens.shape[1]-1)/seconds,
                sustained_input_tokens_per_second=len(samples)*tokens.size/sum(samples),
                teacher_input_tokens_per_second=tokens.size/seconds,
                input_tokens_per_update=int(tokens.size),allocator_before=before,allocator_after=after,
                gradient_evaluations=1+len(samples),probe_input_presentations=int(tokens.size)*(1+len(samples)),
                executable_memory=next(iter(engine.memory.values()),[]))
            print('engine-core probe:',batch,'windows,',round(row['input_tokens_per_second'],1),'input tokens/s',flush=True)
        except Exception as exc:
            memory=isinstance(exc,MemoryError) or any(s in str(exc).lower() for s in ('resource_exhausted','out of memory'))
            if not memory:raise
            row.update(status='memory_limit',error_type=type(exc).__name__,memory_diagnostic=str(exc)[:4096]);save();break
        finally:
            del engine,tokens
        save()
    report['status']='completed' if all(t['status'] in ('measured','memory_limit','nonfinite') for t in report['trials']) and len(report['trials'])==len(candidates) else 'partial'
    save();return report
