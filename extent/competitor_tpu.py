"""Full-chain sharded TPU math for EXP133/134; no GPU subprocesses."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import optax
from extent.competitor_recovery import block_forward,head_tile
from extent.plateau_forensics import norm,mixer,mlp
from extent.recovery_subspace import coordinates_finite
from extent.stable_gradient_clip import scaled_norm_parts


def hidden(cfg,base,q,tokens,columns,spec,step,*,remat=True):
    cfg=replace(cfg,remat_policy='none')
    h=base['embed_tokens']['embedding'][tokens].astype(getattr(jnp,cfg.compute_dtype))
    phase=jnp.asarray(1.,jnp.float32) if spec['ramp_steps']==0 else jnp.clip(step/spec['ramp_steps'],0.,1.)
    for i in range(cfg.num_layers):
        def fn(p,r,h,i=i):return block_forward(cfg,i,p,r,h,columns,spec,phase)[0]
        h=(jax.checkpoint(fn,prevent_cse=False) if remat else fn)(base[f'layers_{i}'],q[str(i)],h)
    return h


def teacher_hidden(cfg,tc,tp,tokens):
    cfg=replace(cfg,remat_policy='none')
    h=tp['embed_tokens']['embedding'][tokens].astype(getattr(jnp,cfg.compute_dtype))
    for i in range(cfg.num_layers):
        p=tp[f'layers_{i}'];n=norm(h,p['input_layernorm'],cfg)
        h=h+mixer(n,p,cfg,i,tc)
        h=h+mlp(norm(h,p['post_attention_layernorm'],cfg),p,cfg)
    return jax.lax.stop_gradient(h)


def objective(cfg,base,q,tokens,columns,spec,step,teacher_h=None,tile=32):
    h=hidden(cfg,base,q,tokens,columns,spec,step)
    count=tokens.shape[0]*(tokens.shape[1]-1);pad=(-count)%tile
    def pack(v):return jnp.pad(v.reshape(count,-1),((0,pad),(0,0))).reshape(-1,tile,v.shape[-1])
    features=pack(h[:,:-1]);labels=jnp.pad(tokens[:,1:].reshape(count),(0,pad)).reshape(-1,tile)
    target=pack(teacher_h[:,:-1]) if spec['alpha'] else features
    mask=(jnp.arange(count+pad)<count).reshape(-1,tile)
    def body(_,data):
        x,y,t,valid=data
        def one(x,y,t):
            return head_tile(cfg,x[None],base['embed_tokens']['embedding'],base['norm']['scale'],
                y[None],t[None],spec['alpha'],1)
        loss,(ce,kl)=jax.vmap(one)(x,y,t)
        return None,jnp.stack((jnp.sum(loss*valid),jnp.sum(ce*valid),jnp.sum(kl*valid)))
    _,values=jax.lax.scan(jax.checkpoint(body,prevent_cse=False),None,(features,labels,target,mask))
    loss,ce,kl=jnp.sum(values,axis=0)/count
    return loss,dict(ce=ce,kl=kl)


def make_step(cfg,tc,columns,spec,tx):
    def step(q,state,base,tp,tokens,cursor):
        th=teacher_hidden(cfg,tc,tp,tokens) if spec['alpha'] else None
        (loss,parts),g=jax.value_and_grad(lambda r:objective(cfg,base,r,tokens,columns,spec,cursor,th),has_aux=True)(q)
        changes,after=tx.update(g,state,q);proposed=optax.apply_updates(q,changes)
        maximum,_,lognorm=scaled_norm_parts(g)
        return proposed,after,dict(loss=loss,**parts,finite=coordinates_finite((loss,g,proposed,after)),
            grad_max_abs=maximum,log10_grad_norm=lognorm)
    return step


def token_nll(cfg,base,q,tokens,columns,spec,step):
    from extent.context_recovery import tiled_nll
    return tiled_nll(cfg,base,hidden(cfg,base,q,tokens,columns,spec,step),tokens)


def reconcile(row,cursor,best_step):
    if row.get('step',0)!=cursor:row.setdefault('rollbacks',[]).append(dict(reported=row.get('step',0),restored=cursor))
    for key in ('history','evaluations'):row[key]=[v for v in row.get(key,[]) if v['step']<=cursor]
    row['step']=cursor;row['best_step']=best_step
    if row.get('final',{}).get('current_step')!=cursor:row.pop('final',None)


def training_batch(pool,step,experiment,offset=0):
    import numpy as np
    from extent.t4_scale_data import training_index
    pool=np.asarray(pool)
    if pool.shape!=(16384,256):raise ValueError('expected pinned 4194304-token pool')
    if experiment==134:
        blocks=pool.reshape(4096,1024)
        return blocks[(int(step)*7919+120123)%4096][None]
    return pool[training_index(step+offset)][None]
