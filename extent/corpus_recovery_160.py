"""EXP160 keeps EXP159 output-KD math, changes only registered training data."""
from extent.hidden_distillation_159 import (
    source_pin, restore_source, prepare, base_coordinates, adam_counts, Inspector,
    block_forward, replay_report, make_kernel, optimizer,
)
from extent.hidden_distillation_159 import recipes as previous_recipes

TPU_EXPERIMENTS = ()
GPU_EXPERIMENTS = (160,)
LIMITS = {160: 12.}


def recipes(experiment):
    if experiment != 160: raise ValueError('EXP160 only')
    source = previous_recipes(159)[0]
    return [dict(source,name=name,data_stream=stream)
            for name,stream in [('KD-WIKI-CYCLIC','wiki'),('KD-C4-FRESH','c4')]]


def teacher_validation(cfg, engine, base, teacher, windows, check):
    """Original Qwen baseline on the exact same D windows; bounded teacher layers."""
    import jax
    import numpy as np
    from extent.context_recovery import tiled_nll
    from extent.reset_streamed_gpu import MemoryCheckedJit
    head=MemoryCheckedJit(lambda embedding,norm,h,t:tiled_nll(
        cfg,dict(embed_tokens=dict(embedding=embedding),norm=norm),h,t))
    device=next(iter(base['embed_tokens']['embedding'].devices()))
    values=[]
    for window in windows:
        check();tokens=jax.device_put(window[None],device)
        hidden,_=engine.teacher_forward(base,teacher,tokens)
        loss=np.asarray(head(base['embed_tokens']['embedding'],base['norm'],hidden,tokens))
        if not np.all(np.isfinite(loss)):raise FloatingPointError('nonfinite teacher C4 NLL')
        values.append(float(loss.mean()))
    return dict(nll=float(np.mean(values)),windows=values,targets_per_window=windows.shape[1]-1,
                input_tokens=int(windows.size),scope='original Qwen; same D128 windows; evaluation cost only')
