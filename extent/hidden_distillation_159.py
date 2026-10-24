"""EXP159: output KD versus output KD plus normalized intermediate targets."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from extent.decoder_recovery_157 import source_pin,restore_source,prepare,base_coordinates,adam_counts,Inspector
from extent.rank_growth_153 import block_forward,replay_report,GrowthStreamedStep
from extent.competitor_recovery import optimizer
from extent.reset_streamed_gpu import StreamedStep,MemoryCheckedJit
ROOT=Path(__file__).resolve().parents[1]
GPU_EXPERIMENTS=(159,)
TPU_EXPERIMENTS=()
LIMITS={159:12.}

def recipes(experiment):
    if experiment!=159:raise ValueError('EXP159 only')
    s=source_pin()['metadata']['contract']['recipes'][0]
    return [dict(s,name=name,source='155-current118304',mode='fresh-adam',decoder_dense=False,
        alpha=.5,hidden_beta=beta,align_layers=[6,13,20,27],ramp_steps=0,reset=False,
        lr_warmup=512,mamba_lr=3e-4,norm_lr=1e-4)
        for name,beta in [('CE-KD',0.),('CE-KD-HIDDEN',.1)]]

def normalized_hidden_mse(student,teacher):
    """Tokenwise direction loss, all prediction positions, FP32 scale-safe RMS."""
    def unit(x):
        x=x.astype(jnp.float32)
        scale=jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(x),axis=-1,keepdims=True),1e-6))
        u=x/scale
        return u*jax.lax.rsqrt(jnp.mean(u*u,axis=-1,keepdims=True)+1e-6)
    target=jax.lax.stop_gradient(unit(teacher[:,:-1]))
    return jnp.mean((unit(student[:,:-1])-target)**2)

class DistillationStep(GrowthStreamedStep):
    def __init__(self,cfg,tc,columns,spec,tx,*,return_gradients=False):
        super().__init__(cfg,tc,columns,spec,tx,return_gradients=return_gradients)
        self.selected=tuple(spec['align_layers']);self.beta=spec['hidden_beta']
        if not self.selected or len(set(self.selected))!=len(self.selected) or any(i<0 or i>=cfg.num_layers for i in self.selected):raise ValueError('invalid hidden target layers')
        self.aux=MemoryCheckedJit(jax.value_and_grad(normalized_hidden_mse,argnums=0))
        self.last_hidden={}
    def teacher_forward(self,base,tp,tokens):
        # Embeddings and head are source-identical and frozen (runner verifies).
        h=self.embed(base['embed_tokens']['embedding'],tokens);targets={}
        device=next(iter(base['embed_tokens']['embedding'].devices()))
        for i in range(self.cfg.num_layers):
            self.check();layer=jax.device_put(tp[f'layers_{i}'],device)
            h=self.teacher(layer,h)
            if i in self.selected and self.beta:targets[i]=np.asarray(jax.device_get(h))
            del layer
        return h,targets
    def __call__(self,q,state,base,tp,tokens,cursor):
        # Teacher parameters never enter differentiation or the optimizer tree.
        th,targets=self.teacher_forward(base,tp,tokens)
        h=self.embed(base['embed_tokens']['embedding'],tokens);states=[];layouts=[];auxiliary={};hidden=0.;layer_metrics={}
        phase=jnp.asarray(1.,jnp.float32)
        for i in range(self.cfg.num_layers):
            self.check();states.append(np.asarray(jax.device_get(h)));layouts.append(h.sharding)
            forward,_=self.kernels(i);h=forward(base[f'layers_{i}'],q[str(i)],h,phase)
            if self.beta and i in targets:
                v,g=self.aux(h,jax.device_put(targets.pop(i),h.sharding))
                layer_metrics[str(i)]=float(v);hidden+=float(v)/len(self.selected)
                auxiliary[i]=np.asarray(jax.device_get(g*(self.beta/len(self.selected))))
        count=tokens.shape[0]*(tokens.shape[1]-1);cotangent=jnp.zeros_like(h);loss=ce=kl=0.
        for start in range(0,tokens.shape[1]-1,32):
            self.check();end=min(start+32,tokens.shape[1]-1)
            (v,(c,k)),gradient=self.head(h[:,start:end],base['embed_tokens']['embedding'],base['norm']['scale'],tokens[:,start+1:end+1],th[:,start:end],jnp.asarray(count,jnp.float32))
            loss+=float(v);ce+=float(c);kl+=float(k);cotangent=cotangent.at[:,start:end].set(gradient)
        del h,th
        gradients={}
        for i in reversed(range(self.cfg.num_layers)):
            self.check()
            if i in auxiliary:
                cotangent=(cotangent.astype(jnp.float32)+jax.device_put(auxiliary.pop(i),cotangent.sharding).astype(jnp.float32)).astype(cotangent.dtype)
            _,backward=self.kernels(i)
            gradients[str(i)],cotangent=backward(base[f'layers_{i}'],q[str(i)],jax.device_put(states[i],layouts[i]),cotangent,phase)
            states[i]=None
        gradients=jax.tree.map(lambda g,p:jax.device_put(g,p.sharding),gradients,q)
        loss+=self.beta*hidden
        result=self.update(q,state,gradients,jnp.asarray(loss,jnp.float32),jnp.asarray(ce,jnp.float32),jnp.asarray(kl,jnp.float32))
        health=dict(result[2],hidden_mse=hidden,hidden_contribution=self.beta*hidden,hidden_by_layer=layer_metrics)
        result=result[:2]+(health,)
        return result+(gradients,) if self.return_gradients else result

def make_kernel(cfg,tc,columns,spec,tx,*,tpu=False):
    if tpu:raise ValueError('EXP159 is two independent GPU workers')
    return DistillationStep(cfg,tc,columns,spec,tx)
