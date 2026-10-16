"""EXP130--134: isolated adaptations of arianraje's inspected recipe.

Pinned attribution: results/EXP-129-pregate-protocol.md. No external code is
executed. Legacy129 and all earlier scientific contracts remain untouched.
"""
import math
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import unfreeze
import optax
from extent.pregate_recovery import pregate_scan
from extent.plateau_forensics import project_mamba,norm,mixer,mlp,plain
from extent.layers.common import dtype_from_name
from extent.t4_scale_recovery import Engine as ScaleEngine,initialize as old_initialize,log_rms
from extent.dynamic_recovery import limit_absolute,relative_factor,initialize as mlp_initialize
from extent.stable_gradient_clip import stable_clip_by_global_norm

TARGET_STEPS=32768
RAMP_STEPS=2048
EVAL_STEPS=(0,128,512,1024,2048,2560,3072,3584,4096,4608,5120,6144,8192,12288,16384,24576,32768)


def arm_for(experiment,worker):
    s=dict(lora=True,scales=False,alpha=0.,penalty=0.,pregate=True,tanh=True,
        reset=False,unprotect=False,mlp=False,mlp_rank=32,mlp_lr=1e-4,
        norm_lr=1e-4,mamba_lr=3e-4,dense=False,relative=None,absolute=None,
        ramp_steps=2048,train_offset=0,temperature=1.,beta2=.999,lr_warmup=256,decay_steps=0,min_lr_frac=1.)
    if experiment==130:
        s.update(name=('NORM-TANH','NORM-TANH-RESET')[worker],reset=bool(worker))
    elif experiment==131:
        s.update(name=('NORM-TANH-UNLOCK','NORM-TANH-UNLOCK-MLP')[worker],unprotect=True,mlp=bool(worker))
    elif experiment==132:
        s.update(name=('NORM-TANH-UNLOCK-MLP-CE','NORM-TANH-UNLOCK-MLP-KL1')[worker],unprotect=True,mlp=True,alpha=float(worker))
    elif experiment==133:
        s.update(name=('CAP100-LR3e4','CAP100-LR3e5','CAP100-LR3e5-MLP')[worker],
            pregate=False,tanh=False,relative=.5,absolute=100.,ramp_steps=0,
            mamba_lr=3e-4 if worker==0 else 3e-5,mlp=worker==2,mlp_lr=1e-5,train_offset=4096)
    elif experiment==134:
        s.update(name=('DENSE-MAMBA-MLP-CE','DENSE-MAMBA-MLP-KL1')[worker],dense=True,lora=False,
            mlp=True,mamba_lr=2e-4,mlp_lr=2e-5,norm_lr=2e-5,alpha=float(worker),
            beta2=.95,lr_warmup=492,decay_steps=16384,min_lr_frac=.1)
    else:raise ValueError('experiments130--134 required')
    return s


def phase_at(spec,step):
    return 1. if spec['ramp_steps']==0 else min(max(int(step),0)/spec['ramp_steps'],1.)


def with_reset_targets(p,cfg,index):
    p=unfreeze(p);c=cfg.mamba;H=int(cfg.hidden_size*c.expand)//c.head_dim
    rng=np.random.default_rng(130123+index)
    dt=np.exp(rng.uniform(math.log(c.dt_min),math.log(c.dt_max),H)).astype(np.float32)
    dt=np.maximum(dt,c.dt_init_floor)
    a=rng.normal(size=(H,c.mimo_rank,c.head_dim)).astype(np.float32)
    a=np.exp(a-a.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
    p['_reset']=dict(dt_bias=jnp.asarray(dt+np.log(-np.expm1(-dt))),mimo_x=jnp.asarray(a))
    return p


def prepare_base(base,cfg,spec):
    if not spec['reset']:return base
    b=unfreeze(base)
    for i in cfg.mamba_layer_indices:b[f'layers_{i}']=with_reset_targets(b[f'layers_{i}'],cfg,i)
    return b


def initialize(patient,spec):
    q=mlp_initialize(patient,spec)
    H=int(patient.cfg.hidden_size*patient.cfg.mamba.expand)//patient.cfg.mamba.head_dim
    for i in patient.cfg.mamba_layer_indices:
        if spec['unprotect']:
            b=q[str(i)]['lora']['in_proj']['b'];q[str(i)]['lora']['in_proj']['b']=b*(jnp.arange(b.shape[1])<patient.columns)[None]
        if spec['pregate']:q[str(i)]['pregate_weight']=jnp.ones((H,patient.cfg.mamba.head_dim),jnp.float32)
        if spec['dense']:q[str(i)]['dense_mamba']=jax.tree.map(lambda x:jnp.asarray(x,jnp.float32),patient.current[f'layers_{i}']['mamba'])
    return q


def optimizer(q,spec):
    def label(path,x):
        names=[getattr(k,'key',None) for k in path]
        return 'mlp' if 'mlp_lora' in names else 'norm' if 'pregate_weight' in names else 'mamba'
    labels=jax.tree.map_with_path(label,q)
    def schedule(peak):
        def fn(n):
            warm=jnp.minimum((n+1)/spec['lr_warmup'],1.)
            progress=jnp.clip((n-spec['lr_warmup'])/max(1,spec['decay_steps']-spec['lr_warmup']),0.,1.)
            decay=spec['min_lr_frac']+(1.-spec['min_lr_frac'])*.5*(1.+jnp.cos(jnp.pi*progress)) if spec['decay_steps'] else 1.
            return -peak*warm*decay
        return fn
    def tx(peak):return optax.chain(optax.scale_by_adam(b1=.9,b2=spec['beta2'],eps=1e-8),optax.scale_by_schedule(schedule(peak)))
    return optax.chain(stable_clip_by_global_norm(1.),optax.multi_transform(
        {k:tx(spec[k+'_lr']) for k in ('mamba','norm','mlp')},labels))


def effective_layer(p,q,columns,spec,strength):
    p=unfreeze(p)
    if 'dense_mamba' in q:
        p['mamba']=jax.tree.map(lambda x,old:x.astype(old.dtype),q['dense_mamba'],p['mamba'])
    if 'lora' in q:
        for name,f in q['lora'].items():
            b=f['b']
            if name=='in_proj' and not spec['unprotect']:b=b*(jnp.arange(b.shape[1])<columns)[None]
            k=p['mamba'][name]['kernel'];p['mamba'][name]['kernel']=(k.astype(jnp.float32)+f['a']@b).astype(k.dtype)
    if spec['reset'] and 'mamba' in p:
        m=p['mamba'];k=m['in_proj']['kernel']
        k=k.at[:,columns:].set((k[:,columns:].astype(jnp.float32)*(1.-strength)).astype(k.dtype))
        m['in_proj']['kernel']=k
        for name in ('dt_bias','mimo_x'):
            old=m[name];m[name]=((1.-strength)*old.astype(jnp.float32)+strength*p['_reset'][name]).astype(old.dtype)
    if 'mlp_lora' in q:
        f=q['mlp_lora'];k=p['mlp']['down_proj']['kernel']
        p['mlp']['down_proj']['kernel']=(k.astype(jnp.float32)+f['a']@f['b']).astype(k.dtype)
    return p


def normalized_mixer(inputs,p,cfg,weight,strength,spec,*,details=False):
    values,_=project_mamba(inputs,p,cfg);values=list(values)
    if spec['tanh']:
        raw_angle=values[-1].astype(jnp.float32)
        values[-1]=(1.-strength)*raw_angle+strength*jnp.tanh(raw_angle)
    c=cfg.mamba
    y=pregate_scan(*values,p['mimo_x'],p['mimo_z'],p['mimo_o'],p['D'],int(c.d_state*c.rope_fraction)//2,
        weight,strength if spec['pregate'] else 0.,details=details)
    if details:y,metrics=y
    features=y.reshape(inputs.shape[0],inputs.shape[1],-1)
    out=nn.Dense(cfg.hidden_size,use_bias=False,dtype=dtype_from_name(cfg.compute_dtype),
        param_dtype=dtype_from_name(cfg.param_dtype)).apply({'params':p['out_proj']},features)
    return (out,metrics) if details else out


def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    p=effective_layer(p,q,columns,spec,strength);n=norm(h,p['input_layernorm'],cfg);internal={}
    is_mamba=index in cfg.mamba_layer_indices
    modified=is_mamba and (spec['pregate'] or spec['tanh'])
    if modified:
        H=int(cfg.hidden_size*cfg.mamba.expand)//cfg.mamba.head_dim
        weight=q.get('pregate_weight',jnp.ones((H,cfg.mamba.head_dim),jnp.float32))
        if details:mixed,internal=normalized_mixer(n,p['mamba'],cfg,weight,strength,spec,details=True)
        else:mixed=jax.lax.cond(strength==0.,lambda _:mixer(n,p,cfg,index),
            lambda _:normalized_mixer(n,p['mamba'],cfg,weight,strength,spec),None)
    else:mixed=mixer(n,p,cfg,index)
    factor=jnp.ones(mixed.shape[:-1]+(1,),jnp.float32)
    if is_mamba and spec['relative'] is not None:factor=relative_factor(h,mixed,spec['relative'],strength)
    scaled=(mixed.astype(jnp.float32)*factor).astype(mixed.dtype)
    raw_middle=h+scaled;middle,middle_factor=limit_absolute(raw_middle,spec['absolute'],strength)
    contribution=mlp(norm(middle,p['post_attention_layernorm'],cfg),p,cfg)
    raw_out=middle+contribution;out,out_factor=limit_absolute(raw_out,spec['absolute'],strength)
    penalty=jax.lax.stop_gradient(jnp.maximum(log_rms(scaled)-log_rms(h),0.)**2)
    if not details:return out,penalty
    metrics=dict(input_rms=jnp.exp(log_rms(h)),raw_mixer_rms=jnp.exp(log_rms(mixed)),
        scaled_mixer_rms=jnp.exp(log_rms(scaled)),output_rms=jnp.exp(log_rms(out)),
        output_max=jnp.max(jnp.abs(out.astype(jnp.float32))),mlp_rms=jnp.exp(log_rms(contribution)),
        relative_active_fraction=jnp.mean(factor<.999999),
        middle_absolute_active_fraction=jnp.mean(middle_factor<.999999),output_absolute_active_fraction=jnp.mean(out_factor<.999999),
        mlp_addition_absorbed_fraction=jnp.sum((raw_out==middle)&(contribution!=0))/jnp.maximum(jnp.sum(contribution!=0),1),
        mixer_addition_absorbed_fraction=jnp.sum((raw_middle==h)&(scaled!=0))/jnp.maximum(jnp.sum(scaled!=0),1))
    metrics.update(internal)
    if 'pregate_weight' in q:metrics.update(norm_weight_min=jnp.min(q['pregate_weight']),norm_weight_max=jnp.max(q['pregate_weight']))
    return out,penalty,metrics


def head_tile(cfg,h,embedding,scale,labels,teacher_h,alpha,normalizer=1):
    # Exact full-vocabulary forward KL at temperature1, matching stage2a's
    # objective family, rather than the old T2/partial-KL experiments.
    dtype=embedding.dtype
    z=(norm(h,{'scale':scale},cfg).astype(dtype)@embedding.T).astype(dtype).astype(jnp.float32)
    sl=jax.nn.log_softmax(z,-1);ce=-jnp.take_along_axis(sl,labels[...,None],-1)[...,0]
    if alpha:
        tz=(norm(teacher_h,{'scale':scale},cfg).astype(dtype)@embedding.T).astype(dtype).astype(jnp.float32)
        tl=jax.lax.stop_gradient(jax.nn.log_softmax(tz,-1));kl=jnp.sum(jnp.exp(tl)*(tl-sl),-1)
    else:kl=jnp.zeros_like(ce)
    return jnp.sum((1-alpha)*ce+alpha*kl)/normalizer,(jnp.sum(ce)/normalizer,jnp.sum(kl)/normalizer)


class Engine(ScaleEngine):
    def __init__(self,patient,spec):
        super().__init__(patient,spec)
        self.set_step(0)
        self.head=jax.jit(jax.value_and_grad(lambda h,e,s,l,t,n:head_tile(self.cfg,h,e,s,l,t,spec['alpha'],n),argnums=0,has_aux=True))
    def set_step(self,step):
        self.phase=self.p.placed(np.asarray(phase_at(self.spec,step),np.float32))
    def parameters(self,layer):
        p=super().parameters(layer)
        return with_reset_targets(p,self.cfg,layer) if self.spec['reset'] and layer in self.cfg.mamba_layer_indices else p
    def kernels(self,layer,q):
        attention=layer in self.cfg.attention_layer_indices
        key=(attention,'lora' in q,'pregate_weight' in q)
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
            key=(attention,'lora' in current,'pregate_weight' in current)
            if key not in self.profile_kernels:
                index=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
                self.profile_kernels[key]=jax.jit(lambda p,q,h,phase,index=index:block_forward(
                    self.cfg,index,p,q,h,self.p.columns,self.spec,phase,details=True))
            parameters=self.p.placed(self.parameters(layer));forward,_=self.kernels(layer,current)
            ordinary,_=forward(parameters,current,h)
            diagnostic,_,metrics=self.profile_kernels[key](parameters,current,h,self.phase)
            error=float(jnp.max(jnp.abs(ordinary.astype(jnp.float32)-diagnostic.astype(jnp.float32)))/jnp.maximum(jnp.max(jnp.abs(ordinary)),1e-30))
            metrics=plain(jax.block_until_ready(metrics))
            rows.append(dict(layer=layer,relative_output_discrepancy=error if math.isfinite(error) else str(error),
                accepted=math.isfinite(error) and error<=.02,**metrics))
            h=ordinary
        return dict(layers=rows,ramp_strength=float(self.phase),relative_output_tolerance=.02,
            interpretation='One A window; competitor-informed surgery; no artificial residual bound or recurrence-trace claim')
