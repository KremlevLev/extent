"""EXP157: fresh-Adam rank256 versus full decoder from pinned155 CURRENT118304."""
from pathlib import Path
from dataclasses import replace
import hashlib,json
import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze
from extent.rank_growth_153 import block_forward as source_block, replay_report
from extent.continuation_matrix import prepare as reconstruct_base, base_coordinates, adam_counts, VerifiedTPUStep
from extent.rank_continue_155 import GrowthVerifiedTPUStep
from extent.competitor_recovery import optimizer as source_optimizer
from extent.stable_gradient_clip import stable_clip_by_global_norm
ROOT=Path(__file__).resolve().parents[1]
TPU_EXPERIMENTS=(157,)
GPU_EXPERIMENTS=()
LIMITS={157:9.}

def source_pin():
    return json.loads((ROOT/'results/EXP-157-source-pin.json').read_text())

def recipes(experiment):
    if experiment!=157:raise ValueError('EXP157 only')
    s=source_pin()['metadata']['contract']['recipes'][0]
    return [dict(s,name=name,mode='fresh-adam',source='155-current118304',decoder_dense=dense,
                 reset=False,ramp_steps=0,lr_warmup=512,mamba_lr=1e-4 if dense else 3e-4,
                 mlp_lr=1e-5,norm_lr=1e-5 if dense else s['norm_lr'])
            for name,dense in [('R256-FRESH-ADAM',False),('FULL-DECODER',True)]]

def restore_source(key,hub,cache,check=lambda:None):
    if key!='155-current118304':raise ValueError('pinned155 CURRENT118304 only')
    from extent.hf_artifact_sync import restore_artifact
    from extent.t4_patient import read_parameter_chunks
    from extent.reset_long_fixed_sources import complete_coordinates
    p=source_pin();h=replace(hub,repo_id=p['repo_id'],repo_type='dataset',revision=p['revision'])
    directory=Path(cache)/key;directory.mkdir(parents=True,exist_ok=True);path=directory/'checkpoint.json'
    check()
    if not path.exists():restore_artifact(path,p['path'],h)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=p['manifest_sha256']:raise ValueError('source SHA differs')
    meta=json.loads(path.read_text())
    if meta!=p['metadata'] or meta['step']!=118304:raise ValueError('source metadata/step differs')
    for name,expected in meta['contract']['code'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:raise ValueError('source code differs: '+name)
    q=complete_coordinates(read_parameter_chunks(meta,directory,p['path'].rsplit('/',1)[0],h,check),meta['contract']['model'])
    # Parameters only: BOTH arms intentionally start fresh optimizer states.
    return dict(parameters=q,best=q,best_step=jnp.asarray(118304,jnp.int32)),dict(p,best_step=118304,adam_step=0)

def dense_coordinates(base,source,cfg,columns,spec):
    q={}
    for i in range(cfg.num_layers):
        p=unfreeze(base[f'layers_{i}']);r=source[str(i)]
        if 'lora' in r:
            for name,f in r['lora'].items():
                b=f['b'];eb=r['extra_lora'][name]['b']
                if name=='in_proj' and not spec['unprotect']:
                    mask=(jnp.arange(b.shape[1])<columns)[None];b=b*mask;eb=eb*mask
                k=p['mamba'][name]['kernel']
                p['mamba'][name]['kernel']=(k.astype(jnp.float32)+f['a']@b+r['extra_lora'][name]['a']@eb).astype(k.dtype)
        # Unused historical reset targets are not trainable model parameters.
        p={k:v for k,v in p.items() if not k.startswith('_')}
        q[str(i)]={'dense_layer':jax.tree.map(lambda v:jnp.asarray(v,jnp.float32),p)}
        if 'pregate_weight' in r:q[str(i)]['pregate_weight']=r['pregate_weight']
    return q

def prepare(patient,spec,payload,pin,ancestor=None):
    base,q,cursor=reconstruct_base(patient,dict(spec,mode='exact'),payload,pin,ancestor)
    if spec.get('decoder_dense'):q=dense_coordinates(base,q,patient.cfg,patient.columns,spec)
    return base,q,cursor

def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    if 'dense_layer' not in q:return source_block(cfg,index,p,q,h,columns,spec,strength,details=details)
    template={k:v for k,v in unfreeze(p).items() if not k.startswith('_')}
    p=jax.tree.map(lambda v,old:v.astype(old.dtype),q['dense_layer'],template)
    remaining={k:v for k,v in q.items() if k!='dense_layer'}
    return source_block(cfg,index,p,remaining,h,columns,spec,strength,details=details)

def optimizer(q,spec):
    if not spec.get('decoder_dense'):return source_optimizer(q,spec)
    def label(path,value):
        names=[getattr(k,'key',None) for k in path]
        if 'pregate_weight' in names or 'input_layernorm' in names or 'post_attention_layernorm' in names:return 'norm'
        return 'mlp' if 'mlp' in names else 'mamba'
    labels=jax.tree.map_with_path(label,q)
    def tx(rate):
        schedule=lambda n:-rate*jnp.minimum((n+1)/spec['lr_warmup'],1.)
        return optax.chain(optax.scale_by_adam(b1=.9,b2=spec['beta2'],eps=1e-8),optax.scale_by_schedule(schedule))
    return optax.chain(stable_clip_by_global_norm(1.),optax.multi_transform({k:tx(spec[k+'_lr']) for k in ('mamba','norm','mlp')},labels))

from extent.reset_streamed_gpu import StreamedStep,MemoryCheckedJit
class DecoderStreamedStep(StreamedStep):
    def kernels(self,index):
        kind=index in self.cfg.attention_layer_indices
        if kind not in self.forward:
            selected=self.cfg.attention_layer_indices[0] if kind else self.cfg.mamba_layer_indices[0]
            def fn(p,r,h,phase):return block_forward(self.cfg,selected,p,r,h,self.columns,self.spec,phase)[0]
            self.forward[kind]=MemoryCheckedJit(fn)
            def backward(p,r,h,dy,phase):
                _,pull=jax.vjp(lambda r,h:fn(p,r,h,phase),r,h);return pull(dy)
            self.backward[kind]=MemoryCheckedJit(backward)
        return self.forward[kind],self.backward[kind]

class DecoderVerifiedStep(VerifiedTPUStep):
    def __init__(self,cfg,tc,columns,spec,tx):
        from extent.competitor_tpu import make_step
        from extent.plateau_forensics import GuardedJit
        self.compiled=GuardedJit(make_step(cfg,tc,columns,spec,tx,block_fn=block_forward,return_gradients=True))
        self.streamed=DecoderStreamedStep(cfg,tc,columns,spec,tx,return_gradients=True)
        self.check=lambda:None;self.selected=None;self.audit=None

def make_kernel(cfg,tc,columns,spec,tx,*,tpu):
    if not tpu:raise ValueError('eight TPU devices required')
    return DecoderVerifiedStep(cfg,tc,columns,spec,tx)

class Inspector:
    def __init__(self,*args):pass
    def profile(self,*args):return {'scope':'No layer-profile claim; quality and numerical gates are primary'}
