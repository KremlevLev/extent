"""EXP114 pure diagnostic math. Patient parameters are never donated/mutated."""
from dataclasses import replace
import math
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from extent.layers.common import RMSNorm,SwiGLU,dtype_from_name
from extent.layers.mamba3 import Mamba3MIMO,ScanState,heavy_tail_activation,_rotate_mimo_halves
from extent.model import HybridDecoderLayer
from extent.qwen3_teacher import Qwen3GQAAttention
from extent.efficiency_diagnostics import stats,relative
from extent.full_model_distillation import causal_cross_entropy
from extent.stable_gradient_clip import stable_clip_by_global_norm,scaled_norm_parts

def state_digest(tree):
    from extent.efficiency_sources import tree_digest
    return tree_digest(serialization.to_state_dict(tree))

def plain(tree):
    def finite_list(v):
        if isinstance(v,list):return [finite_list(x) for x in v]
        return v if not isinstance(v,float) or np.isfinite(v) else str(v)
    def convert(x):
        a=np.asarray(jax.device_get(x))
        if a.ndim:return finite_list(a.tolist())
        v=a.item()
        return v if not isinstance(v,float) or np.isfinite(v) else str(v)
    return jax.tree.map(convert,tree)

class GuardedJit:
    """Compile once per shape/layout and reject a known excessive HBM plan."""
    def __init__(self,function):self.function=jax.jit(function);self.cache={}
    def __call__(self,*args):
        key=tuple((x.shape,str(x.dtype),str(getattr(x,'sharding',None))) for x in jax.tree.leaves(args))
        if key not in self.cache:
            exe=self.function.lower(*args).compile(); m=exe.memory_analysis()
            device=jax.devices()[0].memory_stats() or {}; limit=device.get('bytes_limit',device.get('bytes_reservable_limit'))
            if m is not None and limit is not None:
                extra=m.output_size_in_bytes+m.temp_size_in_bytes-m.alias_size_in_bytes
                if m.argument_size_in_bytes+extra>limit or extra>limit-device.get('bytes_in_use',0):
                    raise MemoryError('EXP114 executable exceeds reported HBM')
            self.cache[key]=exe
        return jax.block_until_ready(self.cache[key](*args))

def token_metrics(logits,tokens):
    z=logits[:,:-1].astype(jnp.float32); labels=tokens[:,1:]
    lp=jax.nn.log_softmax(z,-1); truth=jnp.take_along_axis(z,labels[...,None],-1)[...,0]
    nll=-jnp.take_along_axis(lp,labels[...,None],-1)[...,0]
    # Optimistic rank: 1 + number of STRICTLY greater logits, ties documented.
    rank=1+jnp.sum(z>truth[...,None],-1)
    other=jnp.max(jnp.where(jnp.arange(z.shape[-1])==labels[...,None],-jnp.inf,z),-1)
    return dict(nll=nll,rank=rank,top1=(jnp.argmax(z,-1)==labels),top5=rank<=5,
        entropy=-jnp.sum(jnp.exp(lp)*lp,-1),true_logit=truth,margin=truth-other)

def losses(logits,tokens,tail=None):
    z=jax.nn.log_softmax(logits[:,:-1].astype(jnp.float32),-1)
    nll=-jnp.take_along_axis(z,tokens[:,1:,None],-1)[...,0]
    return jnp.mean(nll if tail is None else nll[:,-tail:])

def norm(x,p,config):
    return RMSNorm(config.hidden_size,config.rms_norm_eps,dtype_from_name(config.param_dtype)).apply({'params':p},x)

def mixer(x,p,config,j,teacher_config=None):
    if teacher_config is not None:
        return Qwen3GQAAttention(teacher_config).apply({'params':p['self_attn']},x,jnp.arange(x.shape[1])[None],None)
    if j in config.attention_layer_indices:
        return Qwen3GQAAttention(replace(teacher_config_for(config),remat_policy=config.remat_policy)).apply(
            {'params':p['self_attn']},x,jnp.arange(x.shape[1])[None],None)
    return Mamba3MIMO(config.hidden_size,config.mamba,dtype_from_name(config.compute_dtype),
        dtype_from_name(config.param_dtype)).apply({'params':p['mamba']},x)

def teacher_config_for(config):
    from extent.model import _attention_source_config
    return _attention_source_config(config)

def mlp(x,p,config):
    return SwiGLU(config.hidden_size,config.intermediate_size,dtype_from_name(config.compute_dtype),
        dtype_from_name(config.param_dtype)).apply({'params':p['mlp']},x)

def parts(x,p,config,j,tc=None):
    n=norm(x,p['input_layernorm'],config); mixed=mixer(n,p,config,j,tc)
    middle=x+mixed; post=norm(middle,p['post_attention_layernorm'],config)
    contribution=mlp(post,p,config)
    return dict(input=x,normalized=n,mixed=mixed,middle=middle,post_normalized=post,mlp=contribution,output=middle+contribution)

def layer_profile(p,q,x,y,config,j,tc):
    a=parts(x,p,config,j); b=parts(y,q,config,j,tc)
    on_teacher=parts(y,p,config,j); q_on_student=parts(x,q,config,j,tc)
    local=on_teacher['output']-b['output']; shift=a['output']-on_teacher['output']
    denominator=jnp.maximum(jnp.max(jnp.abs(local)),jnp.max(jnp.abs(shift)))
    denominator=jnp.maximum(denominator,1e-30)
    return dict(hidden=stats(a['output']),input=stats(x),normalized=stats(a['normalized']),
        post_normalized=stats(a['post_normalized']),teacher_hidden=stats(b['output']),
        input_drift=relative(x,y),mixed=stats(a['mixed']),mlp=stats(a['mlp']),
        mixer_teacher_input_error=relative(on_teacher['mixed'],b['mixed']),
        mixer_student_input_error=relative(a['mixed'],q_on_student['mixed']),
        mlp_teacher_input_error=relative(on_teacher['mlp'],b['mlp']),
        mlp_student_input_error=relative(a['mlp'],q_on_student['mlp']),
        local_error=stats(local),input_shift_effect=stats(shift),
        total_error=stats(a['output']-b['output']),
        normalized_cross_term=jnp.mean((local/denominator)*(shift/denominator)),
        decomposition_identity_max=jnp.max(jnp.abs(local+shift-(a['output']-b['output']))),
        amplification=jnp.sqrt(jnp.mean(a['output'].astype(jnp.float32)**2))/jnp.maximum(jnp.sqrt(jnp.mean(x.astype(jnp.float32)**2)),1e-30))

def project_mamba(inputs,p,config):
    c=config.mamba; inner=int(config.hidden_size*c.expand);heads=inner//c.head_dim
    width=2*inner+2*c.mimo_rank*c.groups*c.d_state+3*heads+int(c.d_state*c.rope_fraction)//2
    projected=nn.Dense(width,use_bias=False,dtype=dtype_from_name(config.compute_dtype),
        param_dtype=dtype_from_name(config.param_dtype)).apply({'params':p['in_proj']},inputs)
    sizes=(inner,inner,c.mimo_rank*c.groups*c.d_state,c.mimo_rank*c.groups*c.d_state,
        heads,heads,heads,int(c.d_state*c.rope_fraction)//2)
    z,x,b,cc,rd,ra,trap,angle=jnp.split(projected,np.cumsum(sizes[:-1]).tolist(),-1)
    batch,length=inputs.shape[:2]
    x=x.reshape(batch,length,heads,c.head_dim);z=z.reshape(batch,length,heads,c.head_dim)
    b=b.reshape(batch,length,c.mimo_rank,c.groups,c.d_state);cc=cc.reshape(b.shape)
    if c.groups==1:
        b=jnp.broadcast_to(b,(batch,length,c.mimo_rank,heads,c.d_state));cc=jnp.broadcast_to(cc,b.shape)
    else:b=jnp.repeat(b,heads//c.groups,3);cc=jnp.repeat(cc,heads//c.groups,3)
    dtype=dtype_from_name(config.param_dtype)
    b=RMSNorm(c.d_state,1e-5,dtype).apply({'params':p['b_norm']},b)+jnp.transpose(p['b_bias'],(1,0,2))[None,None]
    cc=RMSNorm(c.d_state,1e-5,dtype).apply({'params':p['c_norm']},cc)+jnp.transpose(p['c_bias'],(1,0,2))[None,None]
    dt=jax.nn.softplus(rd.astype(jnp.float32)+p['dt_bias'].astype(jnp.float32))
    decay=jnp.minimum(-heavy_tail_activation(ra),-c.a_floor)
    return (x,z,b,cc,dt,decay,trap,angle),dict(raw_dt=rd,raw_decay=ra)

def traced_scan(values,p,rotary_pairs,reset_period=0):
    """Same official recurrence; summarize each state without storing state history."""
    x,z,b,c,dt,decay,trap,angle=values
    batch,length,heads,dim=x.shape;rank=b.shape[2];state_dim=b.shape[-1]
    initial=ScanState(jnp.zeros((batch,heads,dim,state_dim),jnp.float32),
        jnp.zeros((batch,rank,heads,state_dim),jnp.float32),
        jnp.zeros((batch,heads,dim),jnp.float32),jnp.zeros((batch,heads,rotary_pairs),jnp.float32))
    mx=jnp.swapaxes(p['mimo_x'],0,1).astype(jnp.float32)
    mz=jnp.swapaxes(p['mimo_z'],0,1).astype(jnp.float32)
    mo=jnp.swapaxes(p['mimo_o'],0,1).astype(jnp.float32)
    def step(state,data):
        index,x,z,b,c,dt,decay,trap,angle=data
        if reset_period:
            state=jax.tree.map(lambda s:jnp.where((index%reset_period)==0,jnp.zeros_like(s),s),state)
        dt=dt.astype(jnp.float32);alpha=jnp.exp(decay.astype(jnp.float32)*dt)
        gate=jax.nn.sigmoid(trap.astype(jnp.float32));gamma=gate*dt;beta=(1-gate)*dt*alpha
        increment=jnp.pi*angle[:,None,:].astype(jnp.float32)*dt[...,None]
        phase=state.angle+increment
        br=_rotate_mimo_halves(b.astype(jnp.float32),phase,rotary_pairs)
        cr=_rotate_mimo_halves(c.astype(jnp.float32),phase,rotary_pairs)
        now=x[:,None].astype(jnp.float32)*mx[None];prev=state.previous_v[:,None]*mx[None]
        inject=jnp.einsum('brhp,brhn->bhpn',gamma[:,None,:,None]*now,br)
        previous=jnp.einsum('brhp,brhn->bhpn',beta[:,None,:,None]*prev,state.previous_k)
        carry=alpha[:,:,None,None]*state.ssm;ssm=carry+inject+previous
        read=jnp.einsum('bhpn,brhn->brhp',ssm,cr)
        skip=p['D'][None,None,:,None].astype(jnp.float32)*now
        g=jax.nn.silu(z[:,None].astype(jnp.float32)*mz[None])
        # Preserve the official FP32 operation order before the BF16 cast.
        # Distributing gate/output factors over read and skip changes rounding.
        rs=read*g*mo[None];ks=skip*g*mo[None]
        ranks=(read+skip)*g*mo[None]
        y=jnp.sum(ranks,1)
        # Normalize BEFORE covariance/energy to avoid unscaled squares overflow.
        maximum=jnp.maximum(jnp.max(jnp.abs(ranks)),1e-30)
        flat=jnp.swapaxes(ranks/maximum,0,1).reshape(rank,-1)
        cov=flat@flat.T/jnp.maximum(flat.shape[1],1)
        participation=jnp.trace(cov)**2/jnp.maximum(jnp.sum(cov*cov),1e-30)
        trace=dict(state=stats(ssm),carry=stats(carry),injection=stats(inject+previous),
            readout=stats(read),skip=stats(skip),skip_state_cosine=relative(jnp.sum(rs,1),jnp.sum(ks,1))['cosine'],
            alpha_min=jnp.min(alpha),alpha_max=jnp.max(alpha),alpha_mean=jnp.mean(alpha),
            alpha_small=jnp.mean(alpha<1e-3),alpha_near_one=jnp.mean(alpha>0.99),alpha_above_one=jnp.mean(alpha>1),
            gamma=stats(gamma),beta=stats(beta),phase=stats(phase),increment=stats(increment),
            rank_energy=jnp.diag(cov),rank_participation=participation,
            cancellation=jnp.linalg.norm(jnp.sum(flat,0))/jnp.maximum(jnp.sum(jnp.linalg.norm(flat,axis=1)),1e-30))
        return ScanState(ssm,br,x.astype(jnp.float32),phase),(y.astype(x.dtype),trace)
    _,(y,trace)=jax.lax.scan(step,initial,(jnp.arange(length),)+tuple(jnp.swapaxes(v,0,1) for v in values))
    return jnp.swapaxes(y,0,1),trace

def mamba_trace(inputs,p,config,reset_period=0):
    values,raw=project_mamba(inputs,p,config)
    y,trace=traced_scan(values,p,int(config.mamba.d_state*config.mamba.rope_fraction)//2,reset_period)
    features=y.reshape(inputs.shape[0],inputs.shape[1],-1)
    output=nn.Dense(config.hidden_size,use_bias=False,dtype=dtype_from_name(config.compute_dtype),
        param_dtype=dtype_from_name(config.param_dtype)).apply({'params':p['out_proj']},features)
    def distribution(v):
        f=v.astype(jnp.float32)
        return dict(p01=jnp.quantile(f,.01),p50=jnp.quantile(f,.5),p99=jnp.quantile(f,.99),health=stats(f))
    summary={name:distribution(v) for name,v in zip(('value','gate','B','C','dt','decay','trap','angle'),values)}
    summary.update({k:distribution(v) for k,v in raw.items()})
    summary['gate_abs_above_10']=jnp.mean(jnp.abs(values[1])>10)
    return output,dict(distributions=summary,time=trace)

def compress_trace(report,width=32):
    """Explicit time bins, retaining extrema; no raw activations/state payloads."""
    def bins(x):
        return {name:jnp.stack([function(x[i:i+width].astype(jnp.float32),axis=0)
            for i in range(0,x.shape[0],width)]) for name,function in
            (('mean',jnp.mean),('min',jnp.min),('max',jnp.max))}
    return dict(distributions=report['distributions'],time_bins=jax.tree.map(bins,report['time']),
        bin_width=width,binning_note='Each time bin preserves mean/min/max; no interpolation or raw states persisted.')

class ForensicLM(nn.Module):
    """Same parameter names and output path; intervention masks are dynamic."""
    config:object
    teacher_config:object
    mode:str='identity'
    period:int=0

    @nn.compact
    def __call__(self,tokens,reference,mask,boundaries):
        cfg=self.config; dtype=dtype_from_name(cfg.compute_dtype);pd=dtype_from_name(cfg.param_dtype)
        embedding=nn.Embed(cfg.vocab_size,cfg.hidden_size,dtype=dtype,param_dtype=pd,
            embedding_init=nn.initializers.normal(.02),name='embed_tokens')
        x=embedding(tokens); positions=jnp.arange(tokens.shape[1])[None]
        layer_type=nn.remat(HybridDecoderLayer,prevent_cse=False) if cfg.remat_policy=='full' else HybridDecoderLayer
        for j in range(cfg.num_layers):
            before=x;out=layer_type(cfg,j,name=f'layers_{j}')(x,positions,None)
            if self.mode=='rollback':
                out=jax.lax.cond(mask[j],lambda _:HybridDecoderLayer(cfg,j).apply({'params':reference[f'layers_{j}']},before,positions,None),lambda _:out,None)
            elif self.mode=='teacher':
                if j not in cfg.attention_layer_indices:
                    def substitute(_):
                        p=self.variables['params'][f'layers_{j}'];n=norm(before,p['input_layernorm'],cfg)
                        mixed=Qwen3GQAAttention(self.teacher_config).apply({'params':reference[f'layers_{j}']['self_attn']},n,positions,None)
                        middle=before+mixed;return middle+mlp(norm(middle,p['post_attention_layernorm'],cfg),p,cfg)
                    out=jax.lax.cond(mask[j],substitute,lambda _:out,None)
            elif self.mode=='boundary':out=jnp.where(mask[j],boundaries[j],out)
            elif self.mode=='reset' and j not in cfg.attention_layer_indices:
                def substitute(_):
                    p=self.variables['params'][f'layers_{j}'];n=norm(before,p['input_layernorm'],cfg)
                    mixed=mamba_trace(n,p['mamba'],cfg,self.period)[0]
                    middle=before+mixed;return middle+mlp(norm(middle,p['post_attention_layernorm'],cfg),p,cfg)
                out=jax.lax.cond(mask[j],substitute,lambda _:out,None)
            x=out
        x=RMSNorm(cfg.hidden_size,cfg.rms_norm_eps,pd,name='norm')(x)
        if cfg.tie_word_embeddings:logits=embedding.attend(x.astype(pd))
        else:logits=nn.Dense(cfg.vocab_size,use_bias=False,dtype=dtype,param_dtype=pd,name='lm_head')(x)
        return logits.astype(dtype_from_name(cfg.logits_dtype))

def readout_init(hidden,seed,rank=32):
    return dict(a=.01*jax.random.normal(jax.random.key(seed),(hidden,rank),jnp.float32),
        b=jnp.zeros((rank,hidden),jnp.float32))

def readout_logits(features,correction,vocabulary,config):
    dtype=dtype_from_name(config.param_dtype)
    # BF16 effective forward, FP32 masters; B0 gives exact output identity.
    a=correction['a'].astype(dtype);b=correction['b'].astype(dtype)
    transformed=features+(features.astype(dtype)@a)@b
    if config.tie_word_embeddings:
        embed=nn.Embed(config.vocab_size,config.hidden_size,dtype=dtype_from_name(config.compute_dtype),param_dtype=dtype)
        return embed.apply({'params':vocabulary['embed_tokens']},transformed.astype(dtype),method=embed.attend).astype(dtype_from_name(config.logits_dtype))
    return nn.Dense(config.vocab_size,use_bias=False,dtype=dtype_from_name(config.compute_dtype),param_dtype=dtype).apply(
        {'params':vocabulary['lm_head']},transformed).astype(dtype_from_name(config.logits_dtype))

def readout_optimizer():
    return optax.chain(stable_clip_by_global_norm(1),optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8),
        optax.scale_by_schedule(lambda n:-1e-3*jnp.minimum(n/128,1.)))

def readout_step(config):
    tx=readout_optimizer()
    def step(c,state,h,v,t):
        loss,g=jax.value_and_grad(lambda q:causal_cross_entropy(readout_logits(h,q,v,config),t))(c)
        u,proposed=tx.update(g,state,c);after=optax.apply_updates(c,u)
        finite=jnp.isfinite(loss)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((g,after,proposed))]))
        return after,proposed,dict(loss=loss,finite=finite,gradient_log10_norm=scaled_norm_parts(g)[2])
    return step

def select_layers(rows):
    """A only, deterministic tie by layer. Negative deltas are improvements."""
    selected={};pair=[]
    for family in ('rollback','teacher','boundary'):
        candidates=sorted(((r['delta_nll'],int(r['layer'])) for r in rows if r['family']==family))
        if not candidates:raise ValueError('incomplete A sweep')
        selected[family]=candidates[0][1]
        if family=='teacher':pair=[j for _,j in candidates[:2]]
    return dict(single=selected,pair=pair)

def paired_summary(current,altered):
    d=np.asarray(altered,float)-np.asarray(current,float)
    if not np.all(np.isfinite(d)):raise FloatingPointError('nonfinite paired evaluation')
    clusters=[float(np.mean(d[i:i+8])) for i in range(0,len(d),8)]
    return dict(delta_nll=float(np.mean(d)),per_window_delta=d.tolist(),
        fraction_improved=float(np.mean(d<0)),cluster8_means=clusters,
        nll=float(np.mean(altered)),baseline_nll=float(np.mean(current)))
