"""EXP153 v2: preserve trained rank32 factors, append zero-effect capacity.

No extra host folding/cast of147 corrections. Total trainable rank64/256,
fresh Adam in both arms. Original149-154 source/runner files stay immutable.
"""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze
from extent.continuation_matrix import restore_source,source_pin,base_coordinates,adam_counts
from extent.continuation_matrix import prepare as old_prepare,recipes as old_recipes
from extent.reset_followup import block_forward as source_block
from extent.recovery_subspace import initialize_corrections

TPU_EXPERIMENTS=()
GPU_EXPERIMENTS=(153,)
LIMITS={153:12.}


def recipes(experiment):
    if experiment!=153:raise ValueError('repaired EXP153 only')
    source=old_recipes(149)[0]
    return [dict(source,name=f'GROW-R{rank}',mode='grow-rank',total_rank=rank,
        correction_rank=rank,old_rank=32,extra_rank=rank-32) for rank in (64,256)]


def grow_coordinates(base,source,cfg,rank):
    """Copy containers, keep every learned source leaf, append new A/zero B."""
    if rank not in (64,256):raise ValueError('registered target ranks64/256 only')
    q=jax.tree.map(lambda value:value,source)
    added=initialize_corrections(base,cfg.mamba_layer_indices,'INOUT-LORA',
        seed=153123,rank=rank-32,head_dim=cfg.mamba.head_dim)
    for i in cfg.mamba_layer_indices:
        layer=q[str(i)]
        if 'extra_lora' in layer:raise ValueError('source already has growth coordinates')
        for factors in layer['lora'].values():
            if factors['a'].shape[1]!=32 or factors['b'].shape[0]!=32:
                raise ValueError('source correction must have rank32')
        layer['extra_lora']={name:added[f'layers_{i}/mamba/{name}/kernel']
            for name in ('in_proj','out_proj')}
    return q


def prepare(patient,spec,payload,pin,ancestor=None):
    # Reconstruct only the already validated145/147 base. Never fold147 here.
    base,source,cursor=old_prepare(patient,dict(spec,mode='exact'),payload,pin,ancestor)
    return base,grow_coordinates(base,source,patient.cfg,spec['total_rank']),cursor


def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    if 'extra_lora' not in q:
        return source_block(cfg,index,p,q,h,columns,spec,strength,details=details)
    p=unfreeze(p)
    for name,factors in q['lora'].items():
        extra=q['extra_lora'][name]
        b=factors['b'];eb=extra['b']
        if name=='in_proj' and not spec['unprotect']:
            mask=(jnp.arange(b.shape[1])<columns)[None]
            b=b*mask;eb=eb*mask
        kernel=p['mamba'][name]['kernel']
        # Preserve source rank32 dot shape and source addition order. Appended
        # component has B=0 at start; only ONE final cast to stored BF16.
        current=kernel.astype(jnp.float32)+factors['a']@b
        p['mamba'][name]['kernel']=(current+extra['a']@eb).astype(kernel.dtype)
    remaining={key:value for key,value in q.items() if key not in ('lora','extra_lora')}
    return source_block(cfg,index,p,remaining,h,columns,spec,strength,details=details)


def make_kernel(cfg,tc,columns,spec,tx,*,tpu=False,return_gradients=False):
    if tpu:raise ValueError('EXP153 v2 is T4 only')
    return GrowthStreamedStep(cfg,tc,columns,spec,tx,return_gradients=return_gradients)


from extent.reset_streamed_gpu import StreamedStep,MemoryCheckedJit

class GrowthStreamedStep(StreamedStep):
    """Use growth block in BOTH forward and VJP; legacy block_fn is ignored."""
    def kernels(self,index):
        attention=index in self.cfg.attention_layer_indices
        if attention not in self.forward:
            selected=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
            def forward(p,r,h,phase):
                return block_forward(self.cfg,selected,p,r,h,self.columns,self.spec,phase)[0]
            self.forward[attention]=MemoryCheckedJit(forward)
            def backward(p,r,h,dy,phase):
                _,vjp=jax.vjp(lambda r,h:forward(p,r,h,phase),r,h)
                return vjp(dy)
            self.backward[attention]=MemoryCheckedJit(backward)
        return self.forward[attention],self.backward[attention]


def replay_report(control,pin,tolerance=.03):
    """Persistable diagnostics, including actual/reference/worst windows on failure."""
    panels={}
    for name in ('A','B','C'):
        actual=np.asarray(control[name]['windows'],np.float64)
        reference=np.asarray(pin['reference'][name]['windows'],np.float64)
        if actual.shape!=reference.shape:raise ValueError('source replay window count differs')
        error=np.abs(actual-reference);worst=int(np.argmax(error))
        finite=bool(np.all(np.isfinite(actual)))
        panels[name]=dict(actual_nll=float(actual.mean()),reference_nll=float(reference.mean()),
            max_error=float(error.max()) if finite else None,worst_window=worst,
            worst_actual=float(actual[worst]) if finite else None,
            worst_reference=float(reference[worst]),passed=bool(finite and error.max()<=tolerance))
    return dict(passed=all(v['passed'] for v in panels.values()),tolerance=tolerance,panels=panels,
        per_panel_max_error={k:v['max_error'] for k,v in panels.items()})


class Inspector:
    """Same layer metrics and consistency gate, using growth-aware block math."""
    def __init__(self,cfg,base,columns,spec,check=lambda:None):
        self.cfg,self.base,self.columns,self.spec,self.check=cfg,base,columns,spec,check
        self.cache={}
    def profile(self,q,tokens,step):
        from extent.plateau_forensics import plain
        cfg=self.cfg
        h=self.base['embed_tokens']['embedding'][jnp.asarray(tokens)].astype(getattr(jnp,cfg.compute_dtype))
        rows=[];phase=jnp.asarray(1.,jnp.float32)
        for i in range(cfg.num_layers):
            self.check();attention=i in cfg.attention_layer_indices
            if attention not in self.cache:
                index=cfg.attention_layer_indices[0] if attention else cfg.mamba_layer_indices[0]
                normal=jax.jit(lambda p,r,x,phase,index=index:block_forward(cfg,index,p,r,x,self.columns,self.spec,phase)[0])
                detailed=jax.jit(lambda p,r,x,phase,index=index:block_forward(cfg,index,p,r,x,self.columns,self.spec,phase,details=True))
                self.cache[attention]=(normal,detailed)
            normal,detailed=self.cache[attention];p=self.base[f'layers_{i}'];r=q[str(i)]
            ordinary=normal(p,r,h,phase);diagnostic,_,metrics=detailed(p,r,h,phase)
            error=float(jnp.max(jnp.abs(ordinary.astype(jnp.float32)-diagnostic.astype(jnp.float32)))/jnp.maximum(jnp.max(jnp.abs(ordinary)),1e-30))
            rows.append(dict(layer=i,relative_output_discrepancy=error if np.isfinite(error) else str(error),
                accepted=bool(np.isfinite(error) and error<=.02),**plain(metrics)))
            h=ordinary
        return dict(layers=rows,tolerance=.02,interpretation='growth-aware diagnostic vs ordinary; only one profile window')
