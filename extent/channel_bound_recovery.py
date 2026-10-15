"""EXP128: component-wise hard/soft bounds preserving uncapped channels.

No monkey patches to earlier scientific modules. Derivatives pass through
the input RMS, component magnitudes and the limiter; no straight-through/detached gate.
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
from extent.dynamic_recovery import token_log_rms,initialize,optimizer

TARGET_STEPS=32768
RAMP_STEPS=2048
EVAL_STEPS=(0,128,512,1024,2048,4096,8192,16384,24576,32768)

def arm_for(experiment,worker):
    if experiment!=128 or worker not in (0,1):raise ValueError('EXP128 workers0/1 required')
    return dict(name=('R32-CHANNEL-HARD-MLP32','R32-CHANNEL-TANH-MLP32')[worker],
        lora=True,scales=False,alpha=0.,penalty=0.,relative=.5,absolute=10.,
        mlp=True,mlp_rank=32,mlp_lr=1e-4,channel_mode=('hard','tanh')[worker])


def channel_factor(x,log_bound,strength,mode):
    value=x.astype(jnp.float32)
    logmag=jnp.log(jnp.maximum(jnp.abs(value),1e-12))
    logz=logmag-log_bound
    if mode=='hard':target=jnp.minimum(0.,-logz)
    elif mode=='tanh':
        # log(tanh(z)/z), stable at zero and at huge z. For z>10 tanh
        # rounds to1 in FP32; the denominator still uses the true logz.
        safe_z=jnp.exp(jnp.clip(logz,math.log(.01),math.log(10.)))
        ordinary=jnp.log(jnp.tanh(safe_z))-logz
        small=jnp.exp(jnp.minimum(logz,math.log(.01)))
        small_log=jnp.log1p(-small**2/3+2*small**4/15)
        target=jnp.minimum(0.,jnp.where(logz<math.log(.01),small_log,ordinary))
    else:raise ValueError(mode)
    return jnp.exp(strength*target)


def relative_factor(x,m,rho,strength,mode):
    reference=jnp.maximum(token_log_rms(x),math.log(1e-3))
    return channel_factor(m,math.log(rho)+reference,strength,mode)


def limit_absolute(x,ceiling,strength,mode):
    factor=channel_factor(x,math.log(ceiling),strength,mode)
    return (x.astype(jnp.float32)*factor).astype(x.dtype),factor


def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    p=corrected_layer(p,q,columns)
    if 'mlp_lora' in q:
        p=unfreeze(p);f=q['mlp_lora'];kernel=p['mlp']['down_proj']['kernel']
        p['mlp']['down_proj']['kernel']=(kernel.astype(jnp.float32)+f['a']@f['b']).astype(kernel.dtype)
    mixed=mixer(norm(h,p['input_layernorm'],cfg),p,cfg,index)
    factor=jnp.ones(mixed.shape[:-1]+(1,),jnp.float32)
    if index in cfg.mamba_layer_indices and spec['relative'] is not None:
        factor=relative_factor(h,mixed,spec['relative'],strength,spec['channel_mode'])
    scaled=(mixed.astype(jnp.float32)*factor).astype(mixed.dtype)
    raw_middle=h+scaled
    middle,middle_factor=limit_absolute(raw_middle,spec['absolute'],strength,spec['channel_mode'])
    contribution=mlp(norm(middle,p['post_attention_layernorm'],cfg),p,cfg)
    raw_out=middle+contribution
    out,out_factor=limit_absolute(raw_out,spec['absolute'],strength,spec['channel_mode'])
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
            peak_ok=not required or metrics['output_max']<=self.spec['absolute']*1.02
            rows.append(dict(layer=layer,component_peak_bound_ok=peak_ok,bounds_required=required,relative_bound_ok=relative_ok,absolute_bound_ok=absolute_ok,relative_output_discrepancy=error if math.isfinite(error) else str(error),
                accepted=math.isfinite(error) and error<=.02,**metrics))
            h=ordinary
        return dict(layers=rows,ramp_strength=float(self.phase),active_fraction_scope='components',relative_output_tolerance=.02,
            interpretation='One A window; token-wise bounds measured, BF16 rounding allows small bound error; no recurrence trace claim')
