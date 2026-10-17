"""EXP142 full-decoder recovery. Existing 135-141 math/contracts stay untouched."""
import jax
import jax.numpy as jnp
import optax
from extent import competitor_tpu as core
from extent import reset_followup as previous
from extent.stable_gradient_clip import stable_clip_by_global_norm,scaled_norm_parts


def recipes(experiment):
    if experiment!=142:raise ValueError('EXP142 only')
    s=dict(previous.recipes(139)[1],name='RESET130-FULL-DECODER',full_decoder=True,backbone_lr=3e-5,backbone_norm_lr=1e-5)
    return [s]


def prepare(patient,spec,source_payload=None,clean=None):
    base,old,cursor=previous.prepare(patient,spec,source_payload,clean)
    q={str(i):dict(dense_layer=jax.tree.map(lambda v:jnp.asarray(v,jnp.float32),base[f'layers_{i}'])) for i in range(patient.cfg.num_layers)}
    for i in patient.cfg.mamba_layer_indices:q[str(i)]['pregate_weight']=old[str(i)]['pregate_weight']
    expected={'embed_tokens','norm'}|{f'layers_{i}' for i in range(patient.cfg.num_layers)}
    if set(base)!=expected:raise ValueError('unexpected base parameter groups; cannot claim full-decoder coverage')
    q['_global']={k:jax.tree.map(lambda v:jnp.asarray(v,jnp.float32),base[k]) for k in ('embed_tokens','norm')}
    return base,q,cursor


def cast_like(values,reference):
    return jax.tree.map(lambda v,ref:v.astype(ref.dtype),values,reference)


def effective_base(base,q):
    # One tied embedding serves input and readout, so both gradients accumulate.
    return dict(base,**{k:cast_like(q['_global'][k],base[k]) for k in ('embed_tokens','norm')})


def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    p=cast_like(q['dense_layer'],p)
    r={k:v for k,v in q.items() if k!='dense_layer'}
    return previous.block_forward(cfg,index,p,r,h,columns,spec,strength,details=details)


def optimizer(q,spec):
    # Same Adam/clip/warmup/flat schedule as139; newly unlocked dense kernels
    # use3e-5; their original norm scales1e-5. Shared139 rates stay unchanged.
    def label(path,_):
        keys=[getattr(p,'key',None) for p in path]
        if 'pregate_weight' in keys:return 'norm'
        if 'mamba' in keys:return 'mamba'
        return 'backbone_norm' if keys[-1]=='scale' else 'backbone'
    labels=jax.tree.map_with_path(label,q)
    def schedule(peak):
        return lambda n:-peak*jnp.minimum((n+1)/spec['lr_warmup'],1.)
    def tx(peak):return optax.chain(optax.scale_by_adam(b1=.9,b2=spec['beta2'],eps=1e-8),optax.scale_by_schedule(schedule(peak)))
    return optax.chain(stable_clip_by_global_norm(1.),optax.multi_transform({k:tx(spec[k+'_lr']) for k in ('mamba','norm','backbone','backbone_norm')},labels))


def objective(cfg,base,q,tokens,columns,spec,step):
    return core.objective(cfg,effective_base(base,q),q,tokens,columns,spec,step,block_fn=block_forward)


def parameter_group(path):
    keys=[getattr(v,'key',None) for v in path]
    if 'pregate_weight' in keys:return 'pregate_norm'
    if 'mamba' in keys:return 'mamba'
    if keys[-1]=='scale':return 'decoder_norm'
    if 'embed_tokens' in keys:return 'tied_embedding'
    if 'mlp' in keys:return 'mlp'
    if 'self_attn' in keys:return 'gqa'
    raise ValueError(f'unclassified full-decoder parameter: {keys}')


def group_health(q,gradients,proposed):
    groups={}
    for (path,p),g,new in zip(jax.tree.leaves_with_path(q),jax.tree.leaves(gradients),jax.tree.leaves(proposed)):
        group=parameter_group(path)
        row=groups.setdefault(group,dict(parameter_count=0,grad_max_abs=jnp.asarray(0.),master_changed=0,compute_changed=0))
        row['parameter_count']+=p.size
        row['grad_max_abs']=jnp.maximum(row['grad_max_abs'],jnp.max(jnp.abs(g)))
        row['master_changed']+=jnp.sum(new!=p,dtype=jnp.int32)
        row['compute_changed']+=jnp.sum(new.astype(jnp.bfloat16)!=p.astype(jnp.bfloat16),dtype=jnp.int32)
    return groups


def make_step(cfg,tc,columns,spec,tx,*,block_fn=None,return_gradients=False):
    if spec['alpha'] or spec['decay_steps']:raise ValueError('EXP142 requires CE and the registered flat schedule')
    def step(q,state,base,tp,tokens,cursor):
        (loss,parts),g=jax.value_and_grad(lambda r:objective(cfg,base,r,tokens,columns,spec,cursor),has_aux=True)(q)
        changes,after=tx.update(g,state,q);proposed=optax.apply_updates(q,changes)
        maximum,_,lognorm=scaled_norm_parts(g)
        health=core.proposal_health(loss,parts,q,state,g,changes,proposed,after)
        return proposed,after,dict(loss=loss,**parts,**health,grad_max_abs=maximum,log10_grad_norm=lognorm,groups=group_health(q,g,proposed))
    return step


def token_nll(cfg,base,q,tokens,columns,spec,step,*,block_fn=None):
    return core.token_nll(cfg,effective_base(base,q),q,tokens,columns,spec,step,block_fn=globals()['block_forward'])


class Inspector:
    def __init__(self,cfg,base,columns,spec,check=lambda:None):
        self.cfg,self.base,self.columns,self.spec,self.check=cfg,base,columns,spec,check
    def profile(self,q,tokens,step):
        # The profiler must see learned MLP/GQA/norm/embedding values, not frozen sources.
        base=effective_base(self.base,q)
        for i in range(self.cfg.num_layers):base[f'layers_{i}']=cast_like(q[str(i)]['dense_layer'],self.base[f'layers_{i}'])
        r={str(i):{k:v for k,v in q[str(i)].items() if k!='dense_layer'} for i in range(self.cfg.num_layers)}
        return previous.Inspector(self.cfg,base,self.columns,self.spec,self.check).profile(r,tokens,step)
