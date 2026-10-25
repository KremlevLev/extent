"""Execution-only alternatives for EXP160; original scientific source files stay pinned.

No batch, objective, parameter, optimizer or data-cursor change. Selection is an
explicit first-proposal audit, not an assumption of GPU numerical identity.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from extent.hidden_distillation_159 import DistillationStep
from extent.reset_streamed_gpu import MemoryCheckedJit
from extent.rank_growth_153 import block_forward


def compiled_step(cfg,tc,columns,spec,tx):
    """Same tiled objective/Adam; retain intermediates instead of recomputing blocks."""
    import optax
    from extent.competitor_tpu import hidden,teacher_hidden,proposal_health
    from extent.competitor_recovery import head_tile
    from extent.stable_gradient_clip import scaled_norm_parts
    def step(q,state,base,tp,tokens,cursor):
        target=teacher_hidden(cfg,tc,tp,tokens)
        def objective(r):
            h=hidden(cfg,base,r,tokens,columns,spec,cursor,remat=False,block_fn=block_forward)
            count=tokens.shape[0]*(tokens.shape[1]-1);tile=32;pad=(-count)%tile
            def pack(v):return jnp.pad(v.reshape(count,-1),((0,pad),(0,0))).reshape(-1,tile,v.shape[-1])
            features=pack(h[:,:-1]);teacher=pack(target[:,:-1])
            labels=jnp.pad(tokens[:,1:].reshape(count),(0,pad)).reshape(-1,tile)
            mask=(jnp.arange(count+pad)<count).reshape(-1,tile)
            def body(_,data):
                x,y,t,valid=data
                def one(x,y,t):
                    return head_tile(cfg,x[None],base['embed_tokens']['embedding'],base['norm']['scale'],
                                     y[None],t[None],spec['alpha'],1)
                loss,(ce,kl)=jax.vmap(one)(x,y,t)
                return None,jnp.stack((jnp.sum(loss*valid),jnp.sum(ce*valid),jnp.sum(kl*valid)))
            _,values=jax.lax.scan(body,None,(features,labels,teacher,mask))
            loss,ce,kl=jnp.sum(values,axis=0)/count
            return loss,dict(ce=ce,kl=kl)
        (loss,parts),g=jax.value_and_grad(objective,has_aux=True)(q)
        changes,after=tx.update(g,state,q);proposed=optax.apply_updates(q,changes)
        maximum,_,lognorm=scaled_norm_parts(g)
        health=proposal_health(loss,parts,q,state,g,changes,proposed,after)
        return proposed,after,dict(loss=loss,**parts,**health,grad_max_abs=maximum,log10_grad_norm=lognorm),g
    return step


class QueuedJit(MemoryCheckedJit):
    """Existing compile/memory guard, without per-call host synchronization."""
    @classmethod
    def from_checked(cls,checked):
        result=object.__new__(cls);result.__dict__.update(checked.__dict__);return result

    def __call__(self,*args):
        key=tuple((v.shape,str(v.dtype),str(getattr(v,'sharding',None))) for v in jax.tree.leaves(args))
        if key not in self.cache:return super().__call__(*args)
        return self.cache[key](*args)


class DeviceStep(DistillationStep):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.beta:raise ValueError('device execution registered only for EXP160 hidden_beta0')
        for name in ('embed','head','update','teacher'):
            setattr(self,name,QueuedJit.from_checked(getattr(self,name)))

    def kernels(self,index):
        forward,backward=super().kernels(index)
        attention=index in self.cfg.attention_layer_indices
        if not isinstance(forward,QueuedJit):
            self.forward[attention]=QueuedJit.from_checked(forward)
            self.backward[attention]=QueuedJit.from_checked(backward)
        return self.forward[attention],self.backward[attention]

    def __call__(self,q,state,base,tp,tokens,cursor):
        th,_=self.teacher_forward(base,tp,tokens)
        h=self.embed(base['embed_tokens']['embedding'],tokens);states=[]
        phase=jnp.asarray(1.,jnp.float32)
        for i in range(self.cfg.num_layers):
            self.check();states.append(h)
            forward,_=self.kernels(i);h=forward(base[f'layers_{i}'],q[str(i)],h,phase)
        count=tokens.shape[0]*(tokens.shape[1]-1);cotangent=jnp.zeros_like(h);loss=ce=kl=0.
        # Same tile shapes, order, casts and chain rule as the pinned baseline.
        for start in range(0,tokens.shape[1]-1,32):
            self.check();end=min(start+32,tokens.shape[1]-1)
            (v,(c,k)),gradient=self.head(h[:,start:end],base['embed_tokens']['embedding'],
                base['norm']['scale'],tokens[:,start+1:end+1],th[:,start:end],jnp.asarray(count,jnp.float32))
            loss+=float(v);ce+=float(c);kl+=float(k)
            cotangent=cotangent.at[:,start:end].set(gradient)
        del h,th
        gradients={}
        for i in reversed(range(self.cfg.num_layers)):
            self.check();_,backward=self.kernels(i)
            gradients[str(i)],cotangent=backward(base[f'layers_{i}'],q[str(i)],states[i],cotangent,phase)
            states[i]=None
        gradients=jax.tree.map(lambda g,p:jax.device_put(g,p.sharding),gradients,q)
        result=self.update(q,state,gradients,jnp.asarray(loss,jnp.float32),jnp.asarray(ce,jnp.float32),jnp.asarray(kl,jnp.float32))
        health=dict(result[2],hidden_mse=0.,hidden_contribution=0.,hidden_by_layer={})
        return result[:2]+(health,)+(gradients,) if self.return_gradients else result[:2]+(health,)


def compare_proposals(reference,actual,*,resident=False):
    from extent.reset_followup import compare_gradients
    comparison=compare_gradients(reference[3],actual[3])
    ce_error=abs(float(reference[2]['ce'])-float(actual[2]['ce']))
    # Resident layer/tile arithmetic should closely match, not merely meet the
    # existing compiled-vs-streamed BF16 gate used by the TPU experiments.
    tolerance=1e-5 if resident else .03
    passed=comparison['passed'] and ce_error<=tolerance and bool(actual[2]['finite'])
    if resident:passed=passed and comparison.get('relative_l2',1.)<=1e-5
    return dict(comparison,passed=bool(passed),ce_error=ce_error,ce_tolerance=tolerance,
                scope='one unaccepted actual-batch gradient; not bitwise or long-trajectory identity')


class AdaptiveDeviceStep:
    def __init__(self,cfg,tc,columns,spec,tx,*,mode='auto'):
        self.reference=DistillationStep(cfg,tc,columns,spec,tx,return_gradients=True)
        self.resident=DeviceStep(cfg,tc,columns,spec,tx,return_gradients=True)
        self.teacher_actor=self.resident
        self.compiled=MemoryCheckedJit(compiled_step(cfg,tc,columns,spec,tx))
        self.mode=mode;self.check=lambda:None;self.selected=None;self.audit=None

    def teacher_forward(self,*args):
        self.teacher_actor.check=self.check;return self.teacher_actor.teacher_forward(*args)

    def __call__(self,*args):
        for engine in (self.reference,self.resident):
            if engine is not None:engine.check=self.check
        if self.selected is not None:
            self.check();return jax.block_until_ready(self.selected(*args))
        self.check();reference=jax.block_until_ready(self.reference(*args));evaluations=1
        if not bool(reference[2]['finite']):return reference
        candidates={'legacy':self.reference};results={'legacy':reference};checks={}
        for name,engine in [('resident',self.resident)]+([('compiled',self.compiled)] if self.mode in ('auto','compiled') else []):
            self.check()
            print('engine-core: auditing execution',name,flush=True)
            try:
                actual=jax.block_until_ready(engine(*args));evaluations+=1
            except Exception as exc:
                memory=isinstance(exc,MemoryError) or any(s in str(exc).lower() for s in ('resource_exhausted','out of memory'))
                if not memory:raise
                checks[name]=dict(passed=False,reason='execution memory guard/OOM; proposal not accepted',error_type=type(exc).__name__);continue
            comparison=compare_proposals(reference,actual,resident=name=='resident');checks[name]=comparison
            if comparison['passed']:candidates[name]=engine;results[name]=actual
        timings={}
        for name,engine in candidates.items():
            samples=[]
            for _ in range(3):
                self.check();begun=time.perf_counter()
                value=jax.block_until_ready(engine(*args));evaluations+=1
                if not bool(value[2]['finite']):raise FloatingPointError('nonfinite execution timing proposal')
                samples.append(time.perf_counter()-begun)
            timings[name]=float(np.median(samples))
        eligible=candidates
        if self.mode in ('resident','compiled') and self.mode in candidates:eligible={self.mode:candidates[self.mode]}
        chosen=min(eligible,key=lambda n:timings[n]);self.selected=candidates[chosen]
        tokens=args[4]
        self.audit=dict(selected=chosen,checks=checks,steady_seconds=timings,
            compiled_accepted=chosen=='compiled',
            gradient_evaluations=evaluations,audit_extra_input_tokens=(evaluations-1)*tokens.size,
            audit_extra_teacher_input_tokens=(evaluations-1)*tokens.size,
            objective_batch_optimizer_unchanged=True,
            scope='all probes reuse SAME unaccepted q/Adam/batch; only one proposal returned for acceptance')
        print('engine-core: selected execution',chosen,'seconds/update',timings,flush=True)
        result=results[chosen]
        # Release unselected executables; no old/new optimizer values are mixed.
        if chosen!='legacy':self.reference=None
        if chosen!='resident':self.resident=None
        if chosen!='compiled':self.compiled=None
        return result[:2]+(dict(result[2],engine_gate=self.audit),)+result[3:]
