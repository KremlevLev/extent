"""EXP115: retain the trained rank32 block, add zero-effect independent cohorts."""
import math
import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze
from extent.full_model_distillation import causal_cross_entropy
from extent.stable_gradient_clip import stable_clip_by_global_norm,scaled_norm_parts
from extent.recovery_subspace import coordinates_finite

SEED=123
PILOT=4096
FINAL=16384
# (additional rank, activation at additional-update cursor, new peak LR)
ARMS={
    'R32':dict(freeze_old=False,cohorts=[]),
    'G64':dict(freeze_old=False,cohorts=[(32,0,3e-4)]),
    'G128':dict(freeze_old=False,cohorts=[(96,0,3e-4)]),
    'G256':dict(freeze_old=False,cohorts=[(224,0,3e-4)]),
    'ADD256':dict(freeze_old=True,cohorts=[(224,0,3e-4)]),
    'FAST256':dict(freeze_old=False,cohorts=[(224,0,1e-3)]),
    'LADDER256':dict(freeze_old=False,cohorts=[(32,1024,3e-4),(64,2048,3e-4),(128,3072,3e-4)]),
}

def initialize(old,arm):
    spec=ARMS[arm];q={'old':jax.tree.map(lambda x:x.astype(jnp.float32),old)}
    offset=0
    for i,(rank,start,peak) in enumerate(spec['cohorts']):
        group={}
        for index,(name,factors) in enumerate(sorted(old.items())):
            # Same random prefix across capacities and the progressive cohorts.
            a=jax.random.normal(jax.random.fold_in(jax.random.key(115123),index),
                (factors['a'].shape[0],224),jnp.float32)/math.sqrt(factors['a'].shape[0])
            group[name]={'a':a[:,offset:offset+rank],
                         'b':jnp.zeros((rank,factors['b'].shape[1]),jnp.float32)}
        q[f'new{i}']=group;offset+=rank
    return q

def materialize(base,q,columns,spec,step):
    result=unfreeze(base)
    for name,old in q['old'].items():
        path=name.split('/');parent=result
        for piece in path[:-1]:parent=parent[piece]
        kernel=parent[path[-1]]
        mask=(jnp.arange(old['b'].shape[1])<columns)[None,:] if path[-2]=='in_proj' else 1
        old_f=jax.tree.map(jax.lax.stop_gradient,old) if spec['freeze_old'] else old
        value=kernel.astype(jnp.float32)+old_f['a']@(old_f['b']*mask)
        for i,(_,start,_) in enumerate(spec['cohorts']):
            block=q[f'new{i}'][name]
            # Inactive cohorts have no gradient or Adam-time advance.
            active=jnp.asarray(step)>=start
            contribution=block['a']@(block['b']*mask)
            value=value+jnp.where(active,contribution,jax.lax.stop_gradient(jnp.zeros_like(contribution)))
        parent[path[-1]]=value.astype(kernel.dtype)
    return result

def group_optimizer(peak=3e-4,warmup=0):
    # Same four-state layout as EXP111; clipping is applied jointly outside.
    return optax.chain(optax.identity(),optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8),
        optax.identity(),optax.scale_by_schedule(lambda n:-peak*(jnp.minimum(n/float(warmup),1.) if warmup else 1.)))

def potential_coordinate_count(q,columns,freeze_old=False):
    return sum(f['a'].size+f['b'].shape[0]*(min(columns,f['b'].shape[1]) if '/in_proj/' in name else f['b'].shape[1])
        for group,leaves in q.items() if not (group=='old' and freeze_old)
        for name,f in leaves.items())

def optimizers(arm):
    return {'old':group_optimizer(),**{f'new{i}':group_optimizer(peak,128)
        for i,(_,_,peak) in enumerate(ARMS[arm]['cohorts'])}}

def make_step(model,columns,arm):
    spec=ARMS[arm];txs=optimizers(arm);clip=stable_clip_by_global_norm(1)
    def step(q,states,base,tokens,cursor):
        def loss(params):
            logits=model.apply({'params':materialize(base,params,columns,spec,cursor)},tokens)
            return causal_cross_entropy(logits,tokens),jnp.all(jnp.isfinite(logits))
        (value,forward_finite),g=jax.value_and_grad(loss,has_aux=True)(q)
        clipped,_=clip.update(g,optax.EmptyState())
        next_q={};next_s={}
        for name,tx in txs.items():
            if name=='old':active=jnp.array(not spec['freeze_old'])
            else:active=cursor>=spec['cohorts'][int(name[3:])][1]
            def update(args):
                params,state,gradient=args
                updates,following=tx.update(gradient,state,params)
                return optax.apply_updates(params,updates),following
            next_q[name],next_s[name]=jax.lax.cond(active,update,
                lambda args:(args[0],args[1]),(q[name],states[name],clipped[name]))
        maximum,_,lognorm=scaled_norm_parts(g)
        finite=forward_finite & jnp.isfinite(value) & coordinates_finite(g) & coordinates_finite(next_q) & coordinates_finite(next_s)
        return next_q,next_s,dict(loss=value,finite=finite,grad_max_abs=maximum,log10_grad_norm=lognorm)
    return step

def select(branches):
    # Mean of the last two fixed validation observations reduces single-spike selection.
    eligible=[]
    for arm,row in branches.items():
        if arm=='R32' or row.get('failed') or row.get('step',0)<PILOT:continue
        points={int(r['step']):r['validation']['nll'] for r in row['evaluations']}
        if all(s in points and math.isfinite(points[s]) for s in (3072,4096)):
            eligible.append(((points[3072]+points[4096])/2,arm))
    return [arm for score,arm in sorted(eligible)[:2]],{arm:score for score,arm in sorted(eligible)}
