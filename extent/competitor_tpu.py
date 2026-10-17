"""Full-chain sharded TPU math for EXP133/134; no GPU subprocesses."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import optax
from extent.competitor_recovery import block_forward,head_tile
from extent.plateau_forensics import norm,mixer,mlp
from extent.recovery_subspace import coordinates_finite
from extent.stable_gradient_clip import scaled_norm_parts


def hidden(cfg,base,q,tokens,columns,spec,step,*,remat=True,block_fn=block_forward):
    cfg=replace(cfg,remat_policy='none')
    h=base['embed_tokens']['embedding'][tokens].astype(getattr(jnp,cfg.compute_dtype))
    phase=jnp.asarray(1.,jnp.float32) if spec['ramp_steps']==0 else jnp.clip(step/spec['ramp_steps'],0.,1.)
    for i in range(cfg.num_layers):
        def fn(p,r,h,i=i):return block_fn(cfg,i,p,r,h,columns,spec,phase)[0]
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


def objective(cfg,base,q,tokens,columns,spec,step,teacher_h=None,tile=32,*,block_fn=block_forward):
    h=hidden(cfg,base,q,tokens,columns,spec,step,block_fn=block_fn)
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


def tree_health(tree, *, detailed=False):
    """Small on-device diagnostics; no full gradient/state arrays leave the device."""
    entries = {}
    for path, value in jax.tree.leaves_with_path(tree):
        finite = jnp.isfinite(value)
        entries[jax.tree_util.keystr(path)] = dict(
            finite=jnp.all(finite),
            nan_count=jnp.sum(jnp.isnan(value), dtype=jnp.int32),
            inf_count=jnp.sum(jnp.isinf(value), dtype=jnp.int32),
            max_finite_abs=jnp.max(jnp.where(finite, jnp.abs(value.astype(jnp.float32)), 0.)),
        )
    result = dict(finite=jnp.all(jnp.stack([v['finite'] for v in entries.values()])) if entries else jnp.asarray(True))
    if detailed:
        result['leaves'] = entries
    return result


def proposal_health(loss, parts, q, state, gradients, updates, proposed, after, *, detailed=False):
    stages = dict(objective=dict(loss=loss, **parts), parameters_before=q,
        optimizer_before=state, gradients=gradients, updates=updates,
        parameters_after=proposed, optimizer_after=after)
    health = {name: tree_health(value, detailed=detailed) for name, value in stages.items()}
    return dict(finite=jnp.all(jnp.stack([v['finite'] for v in health.values()])), stages=health)


def make_step(cfg,tc,columns,spec,tx,*,detailed=False,block_fn=block_forward,return_gradients=False):
    def step(q,state,base,tp,tokens,cursor):
        th=teacher_hidden(cfg,tc,tp,tokens) if spec['alpha'] else None
        (loss,parts),g=jax.value_and_grad(lambda r:objective(cfg,base,r,tokens,columns,spec,cursor,th,block_fn=block_fn),has_aux=True)(q)
        changes,after=tx.update(g,state,q);proposed=optax.apply_updates(q,changes)
        maximum,_,lognorm=scaled_norm_parts(g)
        health=proposal_health(loss,parts,q,state,g,changes,proposed,after,detailed=detailed)
        result=(proposed,after,dict(loss=loss,**parts,**health,
            grad_max_abs=maximum,log10_grad_norm=lognorm))
        return result+(g,) if return_gradients else result
    return step


def token_nll(cfg,base,q,tokens,columns,spec,step,*,block_fn=block_forward):
    from extent.context_recovery import tiled_nll
    return tiled_nll(cfg,base,hidden(cfg,base,q,tokens,columns,spec,step,block_fn=block_fn),tokens)


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
