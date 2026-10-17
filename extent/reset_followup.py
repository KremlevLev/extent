"""EXP135-141 recipes, immutable EXP130 restoration and clean-start controls.
Legacy experiment recipes are not modified.
"""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze
from extent.competitor_recovery import arm_for as old_arm, initialize, optimizer, with_reset_targets
from extent.competitor_recovery import effective_layer, block_forward as old_block
from extent.recovery_subspace import initialize_corrections
from extent.hf_artifact_sync import restore_artifact
from extent.recovery_durability import RecoveryCheckpointStore
from extent.efficiency_sources import tree_digest
ROOT=Path(__file__).resolve().parents[1]


def recipes(experiment):
    s=old_arm(130,1)
    s.update(start='111',reset=False,reset_parts='all',dense=False,mlp=False)
    if experiment==135:
        return [dict(s,name='RESET130-CONTINUE',start='130-exact',reset=True)]
    if experiment==136:
        return [dict(s,name='RESET-ALL',reset=True),dict(s,name='RESET-PROJECTIONS',reset=True,reset_parts='projections')]
    if experiment==137:
        return [dict(s,name='RESET-DT',reset=True,reset_parts='dt'),dict(s,name='RESET-MIMO-X',reset=True,reset_parts='mimo_x')]
    if experiment==138:
        return [dict(s,name='RESET130-CE',start='130-folded',ramp_steps=0),dict(s,name='RESET130-KL1',start='130-folded',ramp_steps=0,alpha=1.)]
    if experiment==139:
        return [dict(s,name='RESET130-R32',start='130-folded',ramp_steps=0),dict(s,name='RESET130-DENSE',start='130-folded',ramp_steps=0,dense=True,lora=False)]
    if experiment in (140,141):
        dense=experiment==141
        return [dict(s,name=('CLEAN' if start=='clean' else 'PREP072')+('-DENSE' if dense else '-R32'),start=start,reset=False,initial_reset=True,ramp_steps=0,dense=dense,lora=not dense) for start in ('clean','072')]
    raise ValueError('experiments135-141 only')


def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    if spec['reset'] and spec.get('reset_parts','all')!='all' and 'mamba' in p:
        p=unfreeze(p);m=p['mamba'];part=spec['reset_parts']
        if part=='projections':
            k=m['in_proj']['kernel'];m['in_proj']['kernel']=k.at[:,columns:].set((k[:,columns:].astype(jnp.float32)*(1.-strength)).astype(k.dtype))
        else:
            old=m['dt_bias' if part=='dt' else 'mimo_x'];key='dt_bias' if part=='dt' else 'mimo_x'
            m[key]=((1.-strength)*old.astype(jnp.float32)+strength*p['_reset'][key]).astype(old.dtype)
        spec=dict(spec,reset=False)
    return old_block(cfg,index,p,q,h,columns,spec,strength,details=details)


def source130(patient,hub,cache,check=lambda:None):
    pin=json.loads((ROOT/'results/EXP-135-source-pin.json').read_text())
    sourcehub=replace(hub,repo_id=pin['repo_id'],repo_type='dataset',revision=pin['revision'])
    spec=old_arm(130,1)
    with jax.default_device(jax.devices('cpu')[0]):
        q=initialize(patient,spec);state=optimizer(q,spec).init(q)
    template=dict(parameters=q,optimizer=state,best=q,best_step=jnp.asarray(-1,jnp.int32))
    prefix=pin['path'].rsplit('/MAIN/',1)[0]
    store=RecoveryCheckpointStore(Path(cache)/'source130',prefix,sourcehub)
    pointer=store.directory('MAIN')/'checkpoint.json'
    check()
    if not pointer.exists():restore_artifact(pointer,pin['path'],sourcehub)
    if hashlib.sha256(pointer.read_bytes()).hexdigest()!=pin['manifest_sha256']:raise ValueError('EXP130 manifest SHA differs')
    saved=store.restore('MAIN',pin['metadata']['contract'],template)
    if saved is None or saved[1]!=pin['metadata']:raise ValueError('EXP130 immutable metadata differs')
    payload=saved[0]
    if tree_digest(payload['parameters'])!=pin['parameters_sha256']:raise ValueError('EXP130 parameter digest differs')
    counts=[int(np.asarray(jax.device_get(v)).item()) for path,v in jax.tree.leaves_with_path(payload['optimizer']) if getattr(path[-1],'name',getattr(path[-1],'key',None))=='count']
    if not counts or any(v!=pin['source_step'] for v in counts):raise ValueError('EXP130 Adam/schedule clock differs from3434')
    # Chunk reader verifies ALL optimizer and best chunks, not only current weights.
    return payload,pin


def folded130(patient,payload):
    spec=old_arm(130,1);base=unfreeze(patient.base)
    for i in range(patient.cfg.num_layers):
        p=base[f'layers_{i}']
        if i in patient.cfg.mamba_layer_indices:p=with_reset_targets(p,patient.cfg,i)
        base[f'layers_{i}']=jax.device_get(effective_layer(p,payload['parameters'][str(i)],patient.columns,spec,jnp.asarray(1.)))
        base[f'layers_{i}'].pop('_reset',None)
    return base


def clean_transplant(patient):
    from extent.layers.mamba3 import Mamba3MIMO
    from extent.mamba3_transplant import build_qwen3_to_mamba3_transplant_variants
    from extent.qwen3_parity import load_mixer_arrays
    cfg=patient.cfg;base=unfreeze(patient.teacher)
    model=Mamba3MIMO(cfg.hidden_size,cfg.mamba,dtype=jnp.bfloat16,param_dtype=jnp.bfloat16)
    reports={}
    with jax.default_device(jax.devices('cpu')[0]):
        for i in cfg.mamba_layer_indices:
            patient.check()
            random=model.init(jax.random.fold_in(jax.random.key(123),i),jnp.zeros((1,1,cfg.hidden_size),jnp.bfloat16))['params']
            variants,info=build_qwen3_to_mamba3_transplant_variants(random,load_mixer_arrays(patient.reader,patient.tc,i),patient.tc,cfg.mamba,i)
            base[f'layers_{i}'].pop('self_attn');base[f'layers_{i}']['mamba']=jax.device_get(jax.tree.map(lambda v:jnp.asarray(v,jnp.bfloat16),variants['INIT-K-balanced-qkvo-lift']))
            reports[str(i)]=info['INIT-K-balanced-qkvo-lift'].__dict__
    return base,reports


def prepare(patient,spec,source_payload=None,clean=None):
    # Base is explicit in each arm; no inherited EXP111 training in the clean/prep matrix.
    start=spec['start']
    if start=='130-exact':base=patient.base;old=patient.coordinates
    elif start=='130-folded':base=folded130(patient,source_payload);old=None
    elif start=='clean':base=clean;old=None
    elif start=='072':base=patient.base;old=None
    elif start=='111':base=patient.base;old=patient.coordinates
    else:raise ValueError(start)
    if spec.get('initial_reset'):
        base=unfreeze(base)
        for i in patient.cfg.mamba_layer_indices:
            p=with_reset_targets(base[f'layers_{i}'],patient.cfg,i)
            base[f'layers_{i}']=jax.device_get(effective_layer(p,{},patient.columns,old_arm(130,1),jnp.asarray(1.)))
            base[f'layers_{i}'].pop('_reset',None)
    if old is None:
        old=initialize_corrections(base,patient.cfg.mamba_layer_indices,'INOUT-LORA',seed=123,rank=32,head_dim=patient.cfg.mamba.head_dim)
    current=base
    if spec['dense'] and start=='111':current=patient.current
    p=SimpleNamespace(cfg=patient.cfg,base=base,current=current,coordinates=old,columns=patient.columns)
    q=initialize(p,spec)
    if start=='130-exact':q=source_payload['parameters']
    if start=='130-folded':
        for i in patient.cfg.mamba_layer_indices:q[str(i)]['pregate_weight']=source_payload['parameters'][str(i)]['pregate_weight']
    # Dense replaces the effective Mamba, so never retain a redundant LoRA tree.
    if spec['dense']:
        for i in patient.cfg.mamba_layer_indices:q[str(i)].pop('lora',None)
    base=unfreeze(base)
    if spec['reset']:
        for i in patient.cfg.mamba_layer_indices:base[f'layers_{i}']=with_reset_targets(base[f'layers_{i}'],patient.cfg,i)
    cursor=3434 if start=='130-exact' else 0
    return base,q,cursor


class Inspector:
    """One-window profiles; validate diagnostic vs ordinary layer computation."""
    def __init__(self,cfg,base,columns,spec,check=lambda:None):
        self.cfg,self.base,self.columns,self.spec,self.check=cfg,base,columns,spec,check
        self.cache={}
    def profile(self,q,tokens,step):
        from extent.plateau_forensics import plain
        phase=jnp.asarray(1. if self.spec['ramp_steps']==0 else min(step/self.spec['ramp_steps'],1.),jnp.float32)
        h=self.base['embed_tokens']['embedding'][jnp.asarray(tokens)].astype(getattr(jnp,self.cfg.compute_dtype));rows=[]
        for i in range(self.cfg.num_layers):
            self.check();attention=i in self.cfg.attention_layer_indices
            if attention not in self.cache:
                index=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
                normal=jax.jit(lambda p,r,x,phase,index=index:block_forward(self.cfg,index,p,r,x,self.columns,self.spec,phase)[0])
                detailed=jax.jit(lambda p,r,x,phase,index=index:block_forward(self.cfg,index,p,r,x,self.columns,self.spec,phase,details=True))
                self.cache[attention]=(normal,detailed)
            normal,detailed=self.cache[attention];p=self.base[f'layers_{i}'];r=q[str(i)]
            ordinary=normal(p,r,h,phase);diagnostic,_,metrics=detailed(p,r,h,phase)
            error=float(jnp.max(jnp.abs(ordinary.astype(jnp.float32)-diagnostic.astype(jnp.float32)))/jnp.maximum(jnp.max(jnp.abs(ordinary)),1e-30))
            rows.append(dict(layer=i,relative_output_discrepancy=error if np.isfinite(error) else str(error),accepted=bool(np.isfinite(error) and error<=.02),**plain(metrics)))
            h=ordinary
        return dict(layers=rows,tolerance=.02,interpretation='one diagnostic window; failed profile checks exclude corresponding mechanistic claims, ordinary NLL remains quality metric')


def compare_gradients(reference,actual):
    """Robust host comparison of a single gradient pair; never modifies either."""
    a,ta=jax.tree.flatten(reference);b,tb=jax.tree.flatten(actual)
    if ta!=tb:raise ValueError('gradient tree differs')
    x=[np.asarray(jax.device_get(v),np.float64) for v in a]
    y=[np.asarray(jax.device_get(v),np.float64) for v in b]
    if any(u.shape!=v.shape for u,v in zip(x,y)):raise ValueError('gradient shape differs')
    if not all(np.all(np.isfinite(v)) for v in x+y):return dict(passed=False,reason='nonfinite gradient in comparison')
    scale=max([float(np.max(np.abs(v))) for v in x+y]+[1e-300])
    xx=sum(float(np.sum((v/scale)**2)) for v in x);yy=sum(float(np.sum((v/scale)**2)) for v in y)
    diff=sum(float(np.sum(((v-u)/scale)**2)) for u,v in zip(x,y))
    dot=sum(float(np.sum((v/scale)*(u/scale))) for u,v in zip(x,y))
    relative=(diff/max(xx,1e-300))**.5
    cosine=dot/(xx*yy)**.5 if xx>0 and yy>0 else 1. if xx==yy else 0.
    return dict(passed=bool(relative<=.15 and cosine>=.98),relative_l2=relative,cosine=cosine,relative_l2_tolerance=.15,minimum_cosine=.98)


class ReferenceProbe:
    """Compare actual EXP135 gradient to legacy streamed130 before acceptance."""
    def __init__(self,patient,device):
        import copy
        from extent.competitor_recovery import Engine
        p=copy.copy(patient);p.device=device;p.functions={}
        self.device=device
        with jax.default_device(device):self.engine=Engine(p,old_arm(130,1))
    def compare(self,q,tokens,step,loss,gradient):
        with jax.default_device(self.device):
            self.engine.set_step(step)
            report,reference=self.engine.loss_and_grad(jax.device_put(q,self.device),tokens)
        result=compare_gradients(reference,gradient)
        result.update(reference_ce=report['ce'],compiled_ce=float(loss),ce_error=abs(report['ce']-float(loss)),ce_tolerance=.03)
        result['passed']=result['passed'] and result['ce_error']<=.03
        result['scope']='one actual batch; bounded numerical agreement, not bitwise or long-trajectory equivalence'
        return result
