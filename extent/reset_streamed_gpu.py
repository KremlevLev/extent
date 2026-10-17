"""Bounded-memory, host-activation chain rule for EXP135-138 on T4.
No objective, batch, source, optimizer or intervention change; execution changes.
"""
from dataclasses import replace
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
from extent import competitor_tpu as core
from extent.reset_followup import block_forward
from extent.competitor_recovery import head_tile
from extent.plateau_forensics import norm,mixer,mlp
from extent.stable_gradient_clip import scaled_norm_parts


class MemoryCheckedJit:
    """Per-layer compile guard; include numeric evidence instead of opaque errors."""
    def __init__(self,fn):self.function=jax.jit(fn);self.cache={};self.memory={}
    def __call__(self,*args):
        key=tuple((v.shape,str(v.dtype),str(getattr(v,'sharding',None))) for v in jax.tree.leaves(args))
        if key not in self.cache:
            executable=self.function.lower(*args).compile();m=executable.memory_analysis()
            devices=list(executable.runtime_executable().local_devices())
            reports=[]
            for d in devices:
                stats=d.memory_stats() or {};limit=stats.get('bytes_limit',stats.get('bytes_reservable_limit'))
                row=dict(device=d.id,platform=d.platform,limit=limit,in_use=stats.get('bytes_in_use',0))
                if m is not None:
                    row.update(argument_bytes=m.argument_size_in_bytes,output_bytes=m.output_size_in_bytes,temp_bytes=m.temp_size_in_bytes,alias_bytes=m.alias_size_in_bytes)
                    extra=m.output_size_in_bytes+m.temp_size_in_bytes-m.alias_size_in_bytes
                    if limit is not None and (m.argument_size_in_bytes+extra>limit or extra>limit-row['in_use']):
                        raise MemoryError('streamed executable HBM guard: '+json.dumps(row))
                reports.append(row)
            self.memory[key]=reports;self.cache[key]=executable
        return jax.block_until_ready(self.cache[key](*args))


class StreamedStep:
    def __init__(self,cfg,tc,columns,spec,tx,*,return_gradients=False):
        if spec.get('penalty',0)!=0:raise ValueError('streamed reset runner registers zero auxiliary penalty')
        self.cfg=replace(cfg,remat_policy='none');self.tc=tc;self.columns=columns;self.spec=spec;self.return_gradients=return_gradients
        self.forward={};self.backward={};self.check=lambda:None
        self.embed=MemoryCheckedJit(lambda embedding,t:embedding[t].astype(getattr(jnp,cfg.compute_dtype)))
        self.head=MemoryCheckedJit(jax.value_and_grad(lambda h,e,s,l,t,n:head_tile(cfg,h,e,s,l,t,spec['alpha'],n),argnums=0,has_aux=True))
        def update(q,state,g,loss,ce,kl):
            changes,after=tx.update(g,state,q);proposed=optax.apply_updates(q,changes)
            maximum,_,lognorm=scaled_norm_parts(g)
            health=core.proposal_health(loss,dict(ce=ce,kl=kl),q,state,g,changes,proposed,after)
            return proposed,after,dict(loss=loss,ce=ce,kl=kl,**health,grad_max_abs=maximum,log10_grad_norm=lognorm)
        self.update=MemoryCheckedJit(update)
        def teacher(p,h):
            h=h+mixer(norm(h,p['input_layernorm'],self.cfg),p,self.cfg,0,self.tc)
            return h+mlp(norm(h,p['post_attention_layernorm'],self.cfg),p,self.cfg)
        self.teacher=MemoryCheckedJit(teacher)

    def kernels(self,index):
        attention=index in self.cfg.attention_layer_indices
        if attention not in self.forward:
            selected=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
            def forward(p,r,h,phase):return block_forward(self.cfg,selected,p,r,h,self.columns,self.spec,phase)[0]
            self.forward[attention]=MemoryCheckedJit(forward)
            def backward(p,r,h,dy,phase):
                _,vjp=jax.vjp(lambda r,h:forward(p,r,h,phase),r,h)
                return vjp(dy)
            self.backward[attention]=MemoryCheckedJit(backward)
        return self.forward[attention],self.backward[attention]

    def __call__(self,q,state,base,tp,tokens,cursor):
        phase=jnp.asarray(1.,jnp.float32) if not self.spec['ramp_steps'] else jnp.clip(cursor/self.spec['ramp_steps'],0.,1.)
        h=self.embed(base['embed_tokens']['embedding'],tokens);states=[];layouts=[]
        for i in range(self.cfg.num_layers):
            self.check();states.append(np.asarray(jax.device_get(h)));layouts.append(h.sharding)
            forward,_=self.kernels(i);h=forward(base[f'layers_{i}'],q[str(i)],h,phase)
        if self.spec['alpha']:
            th=self.embed(tp['embed_tokens']['embedding'],tokens)
            for i in range(self.cfg.num_layers):self.check();th=self.teacher(tp[f'layers_{i}'],th)
            th=jax.lax.stop_gradient(th)
        else:th=h
        count=tokens.shape[0]*(tokens.shape[1]-1);cotangent=jnp.zeros_like(h)
        loss=ce=kl=0.
        for start in range(0,tokens.shape[1]-1,32):
            self.check();end=min(start+32,tokens.shape[1]-1)
            (v,(c,k)),gradient=self.head(h[:,start:end],base['embed_tokens']['embedding'],base['norm']['scale'],tokens[:,start+1:end+1],th[:,start:end],jnp.asarray(count,jnp.float32))
            loss+=float(v);ce+=float(c);kl+=float(k)
            cotangent=cotangent.at[:,start:end].set(gradient)
        del h,th
        gradients={}
        for i in reversed(range(self.cfg.num_layers)):
            self.check();_,backward=self.kernels(i)
            gradients[str(i)],cotangent=backward(base[f'layers_{i}'],q[str(i)],jax.device_put(states[i],layouts[i]),cotangent,phase)
            states[i]=None
        gradients=jax.tree.map(lambda g,p:jax.device_put(g,p.sharding),gradients,q)
        result=self.update(q,state,gradients,jnp.asarray(loss,jnp.float32),jnp.asarray(ce,jnp.float32),jnp.asarray(kl,jnp.float32))
        return result+(gradients,) if self.return_gradients else result


def make_step(cfg,tc,columns,spec,tx,*,block_fn=None,return_gradients=False):
    return StreamedStep(cfg,tc,columns,spec,tx,return_gradients=return_gradients)
