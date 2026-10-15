"""EXP125--127: causal token-wise bounds and full-chain MLP adaptation.

No monkey patches to earlier scientific modules. Derivatives pass through
both RMS estimates and the limiter; no straight-through/detached gate.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
import optax
from dataclasses import replace
from flax.core import unfreeze
from extent.t4_scale_recovery import Engine as ScaleEngine, initialize as old_initialize, corrected_layer, log_rms
from extent.plateau_forensics import norm,mixer,mlp,plain
from extent.stable_gradient_clip import stable_clip_by_global_norm

TARGET_STEPS=32768
RAMP_STEPS=2048
EVAL_STEPS=(0,128,512,1024,2048,4096,8192,16384,24576,32768)

def arm_for(experiment,worker):
    arms={
      125:[('R32-CONTROL',None,None,False),('R32-REL05',.5,None,False)],
      126:[('R32-REL05-ABS10',.5,10.,False),('R32-REL05-ABS100',.5,100.,False)],
      127:[('R32-REL05-MLP32',.5,None,True),('R32-REL05-ABS10-MLP32',.5,10.,True)]}
    name,relative,absolute,adapt_mlp=arms[experiment][worker]
    return dict(name=name,lora=True,scales=False,alpha=0.,penalty=0.,
        relative=relative,absolute=absolute,mlp=adapt_mlp,mlp_rank=32,mlp_lr=1e-4)

def token_log_rms(x):
    # Only hidden channels of the CURRENT token: never future/batch statistics.
    value=x.astype(jnp.float32)
    scale=jnp.maximum(jnp.max(jnp.abs(value),axis=-1,keepdims=True),1e-6)
    # Smooth numerical floor RMS~1e-12 at zero; finite derivatives there.
    return jnp.log(scale)+.5*jnp.log(jnp.mean((value/scale)**2,axis=-1,keepdims=True)+1e-12)

def relative_factor(x,m,rho,strength):
    reference=jnp.maximum(token_log_rms(x),math.log(1e-3))
    target=jnp.minimum(0.,math.log(rho)+reference-token_log_rms(m))
    return jnp.exp(strength*target)

def absolute_factor(x,ceiling,strength):
    return jnp.exp(strength*jnp.minimum(0.,math.log(ceiling)-token_log_rms(x)))

def limit_absolute(x,ceiling,strength):
    factor=jnp.ones(x.shape[:-1]+(1,),jnp.float32) if ceiling is None else absolute_factor(x,ceiling,strength)
    return (x.astype(jnp.float32)*factor).astype(x.dtype),factor

def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    p=corrected_layer(p,q,columns)
    if 'mlp_lora' in q:
        p=unfreeze(p);f=q['mlp_lora'];kernel=p['mlp']['down_proj']['kernel']
        p['mlp']['down_proj']['kernel']=(kernel.astype(jnp.float32)+f['a']@f['b']).astype(kernel.dtype)
    mixed=mixer(norm(h,p['input_layernorm'],cfg),p,cfg,index)
    factor=jnp.ones(mixed.shape[:-1]+(1,),jnp.float32)
    if index in cfg.mamba_layer_indices and spec['relative'] is not None:
        factor=relative_factor(h,mixed,spec['relative'],strength)
    scaled=(mixed.astype(jnp.float32)*factor).astype(mixed.dtype)
    raw_middle=h+scaled
    middle,middle_factor=limit_absolute(raw_middle,spec['absolute'],strength)
    contribution=mlp(norm(middle,p['post_attention_layernorm'],cfg),p,cfg)
    raw_out=middle+contribution
    out,out_factor=limit_absolute(raw_out,spec['absolute'],strength)
    penalty=jax.lax.stop_gradient(jnp.maximum(log_rms(scaled)-log_rms(h),0.)**2)
    if not details:return out,penalty
    def rms(x):return jnp.exp(log_rms(x))
    return out,penalty,dict(input_rms=rms(h),raw_mixer_rms=rms(mixed),scaled_mixer_rms=rms(scaled),
        pre_mlp_rms=rms(middle),output_rms=rms(out),output_max=jnp.max(jnp.abs(out.astype(jnp.float32))),
        mlp_rms=rms(contribution),relative_factor_min=jnp.min(factor),relative_factor_mean=jnp.mean(factor),
        relative_factor_max=jnp.max(factor),relative_active_fraction=jnp.mean(factor<.999999),
        middle_absolute_active_fraction=jnp.mean(middle_factor<.999999),output_absolute_active_fraction=jnp.mean(out_factor<.999999),
        max_scaled_to_input_rms=jnp.max(jnp.exp(token_log_rms(scaled)-jnp.maximum(token_log_rms(h),math.log(1e-3)))),
        max_output_token_rms=jnp.max(jnp.exp(token_log_rms(out))),
        mlp_addition_absorbed_fraction=jnp.sum((raw_out==middle)&(contribution!=0))/jnp.maximum(jnp.sum(contribution!=0),1),
        mixer_addition_absorbed_fraction=jnp.sum((raw_middle==h)&(scaled!=0))/jnp.maximum(jnp.sum(scaled!=0),1))

def initialize(patient,spec):
    result=old_initialize(patient,spec)
    if spec['mlp']:
        for i in range(patient.cfg.num_layers):
            kernel=patient.base[f'layers_{i}']['mlp']['down_proj']['kernel']
            key=jax.random.fold_in(jax.random.key(127123),i)
            result[str(i)]['mlp_lora']=dict(a=jax.random.normal(key,(kernel.shape[0],spec['mlp_rank']),jnp.float32)/math.sqrt(kernel.shape[0]),
                b=jnp.zeros((spec['mlp_rank'],kernel.shape[1]),jnp.float32))
    return result

def optimizer(q,spec):
    labels=jax.tree.map_with_path(lambda path,x:'mlp' if any(getattr(k,'key',None)=='mlp_lora' for k in path) else 'mamba',q)
    def transformation(peak):
        return optax.chain(optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8),
            optax.scale_by_schedule(lambda n:-peak*jnp.minimum((n+1)/256.,1.)))
    return optax.chain(stable_clip_by_global_norm(1.),optax.multi_transform(
        dict(mamba=transformation(3e-4),mlp=transformation(spec['mlp_lr'])),labels))

class Engine(ScaleEngine):
    def __init__(self,patient,spec):
        super().__init__(patient,spec)
        self.set_step(0)
    def set_step(self,step):
        self.phase=self.p.placed(np.asarray(min(max(int(step),0)/RAMP_STEPS,1.),np.float32))
    def kernels(self,layer,q):
        attention=layer in self.cfg.attention_layer_indices
        key=(attention,'lora' in q,'mlp_lora' in q)
        if key not in self.forward_kernels:
            index=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
            def fn(p,q,h,phase):return block_forward(self.cfg,index,p,q,h,self.p.columns,self.spec,phase)
            self.forward_kernels[key]=jax.jit(fn)
            def backward(p,q,h,dy,phase):
                _,pullback=jax.vjp(lambda q,h:fn(p,q,h,phase),q,h)
                return pullback((dy,jnp.array(0.,jnp.float32)))
            self.backward_kernels[key]=jax.jit(backward)
        return (lambda p,q,h:self.forward_kernels[key](p,q,h,self.phase),
                lambda p,q,h,dy:self.backward_kernels[key](p,q,h,dy,self.phase))
    def inspect(self,q,tokens):
        h=self.p.embed(tokens,self.cfg.compute_dtype);rows=[]
        for layer in range(self.cfg.num_layers):
            self.p.check();current=q[str(layer)]
            attention=layer in self.cfg.attention_layer_indices
            key=(attention,'lora' in current,'mlp_lora' in current)
            if key not in self.profile_kernels:
                index=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
                self.profile_kernels[key]=jax.jit(lambda p,q,h,phase,index=index:block_forward(
                    self.cfg,index,p,q,h,self.p.columns,self.spec,phase,details=True))
            parameters=self.p.placed(self.parameters(layer));forward,_=self.kernels(layer,current)
            ordinary,_=forward(parameters,current,h)
            diagnostic,_,metrics=self.profile_kernels[key](parameters,current,h,self.phase)
            error=float(jnp.max(jnp.abs(ordinary.astype(jnp.float32)-diagnostic.astype(jnp.float32)))/jnp.maximum(jnp.max(jnp.abs(ordinary)),1e-30))
            metrics=plain(jax.block_until_ready(metrics))
            required=float(self.phase)>=1.
            relative_ok=not required or attention or self.spec['relative'] is None or metrics['max_scaled_to_input_rms']<=self.spec['relative']*1.02
            absolute_ok=not required or self.spec['absolute'] is None or metrics['max_output_token_rms']<=self.spec['absolute']*1.02
            rows.append(dict(layer=layer,bounds_required=required,relative_bound_ok=relative_ok,absolute_bound_ok=absolute_ok,relative_output_discrepancy=error if math.isfinite(error) else str(error),
                accepted=math.isfinite(error) and error<=.02,**metrics))
            h=ordinary
        return dict(layers=rows,ramp_strength=float(self.phase),relative_output_tolerance=.02,
            interpretation='One A window; token-wise bounds measured, BF16 rounding allows small bound error; no recurrence trace claim')
