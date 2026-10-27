"""EXP165 compiled objective for the selected recipe's expanded coordinates.

Within each recipe, compiled and reference execution preserve the batch,
objective, parameter/optimizer coordinates and data clock. Selection is audited.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
from extent.hidden_distillation_159 import DistillationStep
from extent.reset_streamed_gpu import MemoryCheckedJit
from extent.method_sweep_163 import block_forward


def gradient_step(cfg,tc,columns,spec,tx):
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
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(g)]))
        finite=finite & jnp.all(jnp.isfinite(jnp.stack([loss,parts['ce'],parts['kl']])))
        return g,dict(loss=loss,**parts,finite=finite)
    return step


def proposal_step(cfg,tc,columns,spec,tx):
    from extent.method_execution_163 import compiled_step
    original=compiled_step(cfg,tc,columns,spec,tx)
    def step(*args):return original(*args)[:3]
    return step
