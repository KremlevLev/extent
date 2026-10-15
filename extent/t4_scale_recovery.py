"""End-to-end CE/KL with layer-wise recomputation and HOST activations.

Only corrections/scalars have Adam state. Frozen BF16 patient kernels are
streamed one layer at a time. No local teacher-mixer fitting objective.
"""
from dataclasses import replace
import math
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze
from extent.plateau_forensics import norm, mixer, mlp
from extent.stable_gradient_clip import stable_clip_by_global_norm,scaled_norm_parts
from extent.recovery_subspace import coordinates_finite

all_finite=jax.jit(coordinates_finite)

TARGET_STEPS = 65536
EVAL_STEPS = (0,128,512,1024,2048,4096,8192,16384,32768,49152,65536)


def arm_for(experiment, worker):
    if experiment == 121:
        return dict(name=('SCALE-CE','SCALE-CE-PENALTY')[worker], scales=True, lora=False,
                    penalty=(0.,.01)[worker], alpha=0., scale_lr=.01)
    if experiment == 122:
        return dict(name=('R32-CE','R32-SCALE-CE')[worker], scales=bool(worker), lora=True,
                    penalty=0., alpha=0., scale_lr=.01)
    if experiment == 123:
        return dict(name=('R32-SCALE-KL10','R32-SCALE-KL50')[worker], scales=True, lora=True,
                    penalty=0., alpha=(.1,.5)[worker], scale_lr=.01)
    raise ValueError('training experiments are121--123')


def log_rms(x):
    value=x.astype(jnp.float32)
    scale=jnp.maximum(jnp.max(jnp.abs(value)),1e-30)
    return jnp.log(scale)+.5*jnp.log(jnp.maximum(jnp.mean((value/scale)**2),1e-30))


def corrected_layer(p,q,columns):
    if 'lora' not in q:return p
    result=unfreeze(p)
    for projection,factors in q['lora'].items():
        b=factors['b']
        if projection=='in_proj':b=b*(jnp.arange(b.shape[1])<columns)[None]
        kernel=p['mamba'][projection]['kernel']
        result['mamba'][projection]['kernel']=(kernel.astype(jnp.float32)+factors['a']@b).astype(kernel.dtype)
    return result


def block_forward(cfg,index,p,q,h,columns,*,direct=False,details=False):
    """Identical functional block for forward, recomputation and controls."""
    p=corrected_layer(p,q,columns)
    n=norm(h,p['input_layernorm'],cfg)
    mixed=mixer(n,p,cfg,index)
    if 'gain' in q:
        factor=q['gain'] if direct else jnp.exp(jnp.clip(q['gain'],-30.,3.))
        scaled=(mixed.astype(jnp.float32)*factor).astype(mixed.dtype)
    else:scaled=mixed
    penalty=jnp.maximum(log_rms(scaled)-jax.lax.stop_gradient(log_rms(h)),0.)**2
    middle=h+scaled
    contribution=mlp(norm(middle,p['post_attention_layernorm'],cfg),p,cfg)
    out=middle+contribution
    if details:
        return out,penalty,dict(input_rms=jnp.exp(log_rms(h)),raw_mixer_rms=jnp.exp(log_rms(mixed)),
            scaled_mixer_rms=jnp.exp(log_rms(scaled)),output_rms=jnp.exp(log_rms(out)),mlp_rms=jnp.exp(log_rms(contribution)),
            output_max=jnp.max(jnp.abs(out.astype(jnp.float32))),
            mlp_addition_absorbed_fraction=jnp.sum((out==middle)&(contribution!=0))/jnp.maximum(jnp.sum(contribution!=0),1),
            mixer_addition_absorbed_fraction=jnp.sum((middle==h)&(scaled!=0))/jnp.maximum(jnp.sum(scaled!=0),1))
    return out,penalty


def initialize(patient,spec):
    result={}
    for layer in range(patient.cfg.num_layers):
        q={}
        if layer in patient.cfg.mamba_layer_indices:
            if spec['scales']:q['gain']=jnp.array(0.,jnp.float32)
            if spec['lora']:
                q['lora']={name:jax.tree.map(lambda a:jnp.asarray(a,jnp.float32),
                    patient.coordinates[f'layers_{layer}/mamba/{name}/kernel']) for name in ('in_proj','out_proj')}
        result[str(layer)]=q
    return result


def optimizer(q,spec):
    def schedule(peak):return lambda n:-peak*jnp.minimum((n+1)/256.,1.)
    labels=jax.tree.map_with_path(lambda path,x:'gain' if str(path[-1].key)=='gain' else 'lora',q)
    transforms={name:optax.chain(optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8),
        optax.scale_by_schedule(schedule(peak))) for name,peak in (('gain',spec['scale_lr']),('lora',3e-4))}
    # Clip jointly BEFORE splitting into LR groups. Fresh Adam for every arm;
    # EXP111 moments are deliberately not claimed to be preserved.
    return optax.chain(stable_clip_by_global_norm(1),optax.multi_transform(transforms,labels))


def head_tile(cfg,h,embedding,scale,labels,teacher_h,alpha,normalizer=1):
    dtype=embedding.dtype
    z=(norm(h,{'scale':scale},cfg).astype(dtype)@embedding.T).astype(dtype).astype(jnp.float32)
    ce=-jnp.take_along_axis(jax.nn.log_softmax(z,-1),labels[...,None],-1)[...,0]
    # Exact full-vocabulary KL at T2; teacher arrays are fixed, not top-k.
    if alpha:
        tz=(norm(teacher_h,{'scale':scale},cfg).astype(dtype)@embedding.T).astype(dtype).astype(jnp.float32)
        teacher_log=jax.lax.stop_gradient(jax.nn.log_softmax(tz/2.,-1))
        student_log=jax.nn.log_softmax(z/2.,-1)
        kl=4*jnp.sum(jnp.exp(teacher_log)*(teacher_log-student_log),-1)
    else:kl=jnp.zeros_like(ce)
    return jnp.sum((1-alpha)*ce+alpha*kl)/normalizer,(jnp.sum(ce)/normalizer,jnp.sum(kl)/normalizer)


class Engine:
    def __init__(self,patient,spec,*,direct=False):
        self.p,self.spec,self.direct=patient,spec,direct
        self.forward_kernels={};self.backward_kernels={};self.profile_kernels={}
        cfg=replace(patient.cfg,remat_policy='none')
        self.cfg=cfg
        self.embedding=patient.placed(patient.current['embed_tokens']['embedding'])
        self.final_scale=patient.placed(patient.current['norm']['scale'])
        self.head=jax.jit(jax.value_and_grad(lambda h,e,s,l,t,n:head_tile(cfg,h,e,s,l,t,spec['alpha'],n),argnums=0,has_aux=True))

    def parameters(self,layer):
        return (self.p.base if self.spec.get('lora') else self.p.current)[f'layers_{layer}']

    def kernels(self,layer,q):
        attention=layer in self.cfg.attention_layer_indices
        key=(attention,'gain' in q,'lora' in q)
        if key not in self.forward_kernels:
            index=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
            def fn(p,q,h):return block_forward(self.cfg,index,p,q,h,self.p.columns,direct=self.direct)
            self.forward_kernels[key]=jax.jit(fn)
            def backward(p,q,h,dy):
                _,pullback=jax.vjp(lambda q,h:fn(p,q,h),q,h)
                # Mean over Mamba layers; denominator is fixed, cannot be
                # gamed by increasing the incoming residual norm.
                weight=jnp.asarray(0. if attention else self.spec['penalty']/len(self.cfg.mamba_layer_indices),jnp.float32)
                return pullback((dy,weight))
            self.backward_kernels[key]=jax.jit(backward)
        return self.forward_kernels[key],self.backward_kernels[key]

    def forward(self,q,tokens,*,record=False):
        h=self.p.embed(tokens,self.cfg.compute_dtype);states=[];penalties=[]
        for layer in range(self.cfg.num_layers):
            self.p.check()
            if record:states.append(np.asarray(jax.device_get(h)))
            function,_=self.kernels(layer,q[str(layer)])
            h,penalty=function(self.p.placed(self.parameters(layer)),q[str(layer)],h)
            h,penalty=jax.block_until_ready((h,penalty))
            if layer in self.cfg.mamba_layer_indices:penalties.append(float(penalty))
        return h,states,float(np.mean(penalties))

    def loss_and_grad(self,q,tokens):
        h,states,penalty=self.forward(q,tokens,record=True)
        teacher=self.p.forward_hidden(tokens,teacher=True) if self.spec['alpha'] else h
        cotangent=jnp.zeros_like(h);total=0.;ce=0.;kl=0.
        count=tokens.shape[0]*(tokens.shape[1]-1)
        for start in range(0,tokens.shape[1]-1,32):
            self.p.check();end=min(start+32,tokens.shape[1]-1)
            (value,(c,k)),gradient=self.head(h[:,start:end],self.embedding,self.final_scale,
                self.p.placed(tokens[:,start+1:end+1]),teacher[:,start:end],count)
            value,c,k,gradient=jax.block_until_ready((value,c,k,gradient))
            total+=float(value);ce+=float(c);kl+=float(k)
            cotangent=cotangent.at[:,start:end].set(gradient)
        gradients={}
        for layer in reversed(range(self.cfg.num_layers)):
            self.p.check();_,backward=self.kernels(layer,q[str(layer)])
            gradient,cotangent=backward(self.p.placed(self.parameters(layer)),q[str(layer)],
                self.p.placed(states[layer]),cotangent)
            gradient,cotangent=jax.block_until_ready((gradient,cotangent))
            gradients[str(layer)]=gradient
            states[layer]=None
        report=dict(loss=total+self.spec['penalty']*penalty,ce=ce,kl=kl,activation_penalty=penalty)
        finite=all(np.isfinite(v) for v in report.values()) and bool(all_finite(gradients))
        if not finite:raise FloatingPointError('nonfinite full-model forward/gradient; proposal rejected')
        return report,gradients

    def evaluate(self,q,windows,*,tail=None,profile=False):
        values=[];penalties=[];maxima=[]
        for t in windows:
            t=np.asarray(t,np.int32)[None]
            h,_,penalty=self.forward(q,t)
            losses=self.p.token_losses(h,t)[0]
            values.append(float(np.mean(losses if tail is None else losses[-tail:])))
            penalties.append(penalty)
            if profile:maxima.append(float(jnp.max(jnp.abs(h.astype(jnp.float32)))))
        if not np.all(np.isfinite(values)):raise FloatingPointError('nonfinite evaluation')
        return dict(nll=float(np.mean(values)),windows=values,
            activation_penalty=float(np.mean(penalties)),final_hidden_max=maxima)

    def inspect(self,q,tokens):
        from extent.plateau_forensics import plain
        h=self.p.embed(tokens,self.cfg.compute_dtype);rows=[]
        for layer in range(self.cfg.num_layers):
            self.p.check();current=q[str(layer)];attention=layer in self.cfg.attention_layer_indices
            key=(attention,'gain' in current,'lora' in current)
            if key not in self.profile_kernels:
                index=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
                self.profile_kernels[key]=jax.jit(lambda p,q,h,index=index:block_forward(self.cfg,index,p,q,h,
                    self.p.columns,direct=self.direct,details=True))
            parameters=self.p.placed(self.parameters(layer));forward,_=self.kernels(layer,current)
            ordinary,_=forward(parameters,current,h)
            diagnostic,_,metrics=self.profile_kernels[key](parameters,current,h)
            error=jnp.max(jnp.abs(ordinary.astype(jnp.float32)-diagnostic.astype(jnp.float32)))/jnp.maximum(jnp.max(jnp.abs(ordinary)),1e-30)
            metrics=plain(jax.block_until_ready(metrics));error=float(error)
            rows.append(dict(layer=layer,relative_output_discrepancy=error,accepted=math.isfinite(error) and error<=.02,**metrics))
            h=ordinary
        return dict(layers=rows,relative_output_tolerance=.02,
            interpretation='Inspect-executable BF16 discrepancies checked per layer; profile uses one explicit A window')


def make_update(tx):
    @jax.jit
    def proposal(q,state,gradient):
        changes,after=tx.update(gradient,state,q)
        proposed=optax.apply_updates(q,changes)
        maximum,_,lognorm=scaled_norm_parts(gradient)
        health=dict(finite=coordinates_finite((gradient,proposed,after)),grad_max_abs=maximum,log10_grad_norm=lognorm)
        return proposed,after,health
    def apply(q,state,gradient):
        proposed,after,health=jax.block_until_ready(proposal(q,state,gradient))
        if not bool(health['finite']):raise FloatingPointError('nonfinite optimizer proposal rejected')
        return proposed,after,{k:float(v) for k,v in health.items() if k!='finite'}
    return apply


def update(tx,q,state,gradient):
    proposed,after,_=make_update(tx)(q,state,gradient)
    return proposed,after


def gain_report(q):
    return {layer:float(jnp.exp(jnp.clip(row['gain'],-30.,3.)))
            for layer,row in q.items() if 'gain' in row}
