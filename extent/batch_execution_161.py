"""One batch6 Adam update; first gradient checked against six fixed-q microsteps."""
import jax
import jax.numpy as jnp
from extent.hidden_distillation_159 import DistillationStep
from extent.device_execution import compiled_step,compare_proposals
from extent.reset_streamed_gpu import MemoryCheckedJit


class BatchStep:
    def __init__(self,cfg,tc,columns,spec,tx):
        self.reference=DistillationStep(cfg,tc,columns,spec,tx,return_gradients=True)
        self.compiled=MemoryCheckedJit(compiled_step(cfg,tc,columns,spec,tx))
        self.check=lambda:None;self.selected=None

    def teacher_forward(self,*args):
        self.reference.check=self.check;return self.reference.teacher_forward(*args)

    def accumulated(self,q,state,base,tp,tokens,cursor):
        """Mean gradient at FIXED q, then ONE Adam update (not six updates)."""
        self.reference.check=self.check;n=tokens.shape[0];gradient=None;loss=ce=kl=0.
        for i in range(n):
            self.check()
            value=self.reference(q,state,base,tp,tokens[i:i+1],cursor)
            if not bool(value[2]['finite']):return value
            g=jax.tree.map(lambda x:x.astype(jnp.float32)/n,value[3])
            gradient=g if gradient is None else jax.tree.map(lambda a,b:a+b,gradient,g)
            loss+=float(value[2]['loss'])/n;ce+=float(value[2]['ce'])/n;kl+=float(value[2]['kl'])/n
            del value,g
        result=self.reference.update(q,state,gradient,jnp.asarray(loss,jnp.float32),jnp.asarray(ce,jnp.float32),jnp.asarray(kl,jnp.float32))
        return result+(gradient,)

    def __call__(self,*args):
        if args[4].shape[0]!=6:raise ValueError('batch6 input required')
        self.reference.check=self.check;self.check()
        if self.selected is not None:return jax.block_until_ready(self.selected(*args))
        # Allocate the largest executable FIRST. Probing smaller shapes first
        # previously fragmented BFC and caused a spurious batch4 allocation fail.
        candidate=None;memory_reason=None
        try:candidate=jax.block_until_ready(self.compiled(*args))
        except Exception as exc:
            memory=isinstance(exc,MemoryError) or any(s in str(exc).lower() for s in ('resource_exhausted','out of memory'))
            if not memory:raise
            memory_reason=dict(type=type(exc).__name__,message=str(exc)[:2048])
        self.check();reference=jax.block_until_ready(self.accumulated(*args))
        if not bool(reference[2]['finite']):return reference
        comparison=compare_proposals(reference,candidate) if candidate is not None else dict(passed=False,reason='compiled memory guard',memory=memory_reason)
        chosen='compiled-batch6' if comparison['passed'] else 'accumulated-batch6'
        self.selected=self.compiled if comparison['passed'] else self.accumulated
        result=candidate if comparison['passed'] else reference
        gate=dict(selected=chosen,comparison=comparison,gradient_evaluations=2 if candidate is not None else 1,
            evaluation_unit='one full batch6 equivalent',reference_microsteps=6,
            batch=6,optimizer_updates_accepted_by_audit=0,
            scope='candidate and six single-example gradients use same q/Adam/input; one proposal returned for caller acceptance')
        print('engine-core batch6: gradient audit',gate,flush=True)
        return result[:2]+(dict(result[2],engine_gate=gate),)+result[3:]
