"""EXP124: matched-target short/long context, full-chain TPU autodiff."""
from dataclasses import replace
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import optax
from extent.t4_scale_recovery import initialize, optimizer, block_forward, gain_report
from extent.plateau_forensics import norm
from extent.recovery_subspace import coordinates_finite
from extent.stable_gradient_clip import scaled_norm_parts

ARMS=('SHORT','LONG')
HORIZON=32768
EVAL_STEPS=(0,256,1024,4096,8192,16384,24576,32768)
SPEC=dict(scales=True,lora=True,scale_lr=.01)

def initial_coordinates(config,old):
    return {k:v for k,v in initialize(SimpleNamespace(cfg=config,coordinates=old),SPEC).items() if v}

def training_batch(pool,step,arm):
    if arm not in ARMS:raise ValueError(arm)
    blocks=np.asarray(pool).reshape(-1,1024)
    if len(blocks)!=4096:raise ValueError('expected pinned 4194304-token pool')
    block=blocks[(int(step)*7919+120123)%4096]
    if arm=='SHORT':return block.reshape(4,256),np.ones((4,255),np.float32)
    mask=np.ones((1,1023),np.float32);mask[0,[255,511,767]]=0
    return block.reshape(1,1024),mask

def hidden(config,base,q,tokens,columns,*,remat=True):
    cfg=replace(config,remat_policy='none')
    h=base['embed_tokens']['embedding'][tokens].astype(getattr(jnp,cfg.compute_dtype))
    for layer in range(cfg.num_layers):
        def fn(p,r,x,layer=layer):return block_forward(cfg,layer,p,r,x,columns)[0]
        h=(jax.checkpoint(fn,prevent_cse=False) if remat else fn)(base[f'layers_{layer}'],q.get(str(layer),{}),h)
    return h

def tiled_nll(config,base,h,tokens,tile=32):
    """Full-vocabulary CE; at most32 positions of logits live per head tile."""
    count=tokens.shape[0]*(tokens.shape[1]-1)
    features=h[:,:-1].reshape(count,config.hidden_size)
    labels=tokens[:,1:].reshape(count)
    padding=(-count)%tile
    features=jnp.pad(features,((0,padding),(0,0))).reshape(-1,tile,config.hidden_size)
    labels=jnp.pad(labels,(0,padding)).reshape(-1,tile)
    def body(_,values):
        x,y=values
        x=norm(x,base['norm'],config).astype(getattr(jnp,config.param_dtype))
        logits=(x@base['embed_tokens']['embedding'].T).astype(getattr(jnp,config.compute_dtype)).astype(jnp.float32)
        nll=-jnp.take_along_axis(jax.nn.log_softmax(logits,-1),y[:,None],-1)[:,0]
        return None,nll
    _,loss=jax.lax.scan(jax.checkpoint(body,prevent_cse=False),None,(features,labels))
    return loss.reshape(-1)[:count].reshape(tokens.shape[0],tokens.shape[1]-1)

def objective(config,base,q,tokens,mask,columns):
    values=tiled_nll(config,base,hidden(config,base,q,tokens,columns),tokens)
    return jnp.sum(values*mask)/jnp.maximum(jnp.sum(mask),1)

def make_step(config,columns,tx):
    def step(q,state,base,tokens,mask):
        loss,g=jax.value_and_grad(lambda r:objective(config,base,r,tokens,mask,columns))(q)
        changes,after=tx.update(g,state,q);proposal=optax.apply_updates(q,changes)
        maximum,_,lognorm=scaled_norm_parts(g)
        return proposal,after,dict(loss=loss,finite=coordinates_finite((loss,g,proposal,after)),
            grad_max_abs=maximum,log10_grad_norm=lognorm)
    return step

def reconcile(row,cursor):
    if row.get('step',0)!=cursor:
        row.setdefault('rollbacks',[]).append(dict(reported=row.get('step',0),restored=cursor))
        row['evaluations']=[e for e in row.get('evaluations',[]) if e['step']<=cursor]
        row['history']=[e for e in row.get('history',[]) if e['step']<=cursor]
        row.pop('final',None)
    row['step']=cursor
