"""Clean teacher-head initialization; EXP170 engineering building block, not a runner.

Independent adaptation of the head-preserving recipe documented by arianraje:
https://huggingface.co/arianraje/qwen3-4b-mamba3-hybrid-stage2b-kd-bias1
revision b946069890837bbe9b5dd75e2d118a7e98d2d193, Apache-2.0.
Qwen1.7B has2 query heads/KV group, whereas the inspected4B has4. With MIMO4,
extra C slots duplicate queries; a one-hot readout still selects each own head.
Preserves projection ownership, NOT original softmax attention or full outputs.
"""
import math
import numpy as np


def geometry(source,mamba):
    h=source.num_attention_heads;g=source.num_key_value_heads;n=source.head_dim
    if h*n!=source.hidden_size or h%g:raise ValueError('unsupported teacher head geometry')
    if (mamba.head_dim,mamba.d_state,mamba.groups,int(source.hidden_size*mamba.expand))!=(n,n,g,h*n):
        raise ValueError('head-preserving transplant requires donor P,N,KV groups and full query width; no crop/pooling')
    if mamba.mimo_rank%(h//g):raise ValueError('MIMO rank must be a multiple of queries per KV group')
    return h,g,n,h//g


def initialize(arrays,source,mamba,layer,*,seed=123,bc_bias=1.):
    from extent.mamba3_transplant import mamba3_projection_slices
    h,g,n,per_group=geometry(source,mamba);r=mamba.mimo_rank;d=source.hidden_size
    prefix=f'model.layers.{layer}.self_attn'
    def matrix(name,shape):
        value=np.asarray(arrays[prefix+'.'+name+'.weight'],np.float32).T
        if value.shape!=shape:raise ValueError('teacher tensor shape mismatch: '+name)
        return value
    q=matrix('q_proj',(d,h*n)).reshape(d,h,n)
    k=matrix('k_proj',(d,g*n)).reshape(d,g,n)
    v=matrix('v_proj',(d,g*n)).reshape(d,g,n)
    o=matrix('o_proj',(h*n,d))
    slices=mamba3_projection_slices(d,mamba);width=slices['angle'].stop
    rng=np.random.default_rng(seed+layer);kernel=np.zeros((d,width),np.float32)
    kernel[:,slices['z']]=rng.normal(0,.02,(d,h*n)).astype(np.float32)
    kernel[:,slices['x']]=np.repeat(v,per_group,axis=1).reshape(d,h*n)
    kernel[:,slices['b']]=np.broadcast_to(k[:,None],(d,r,g,n)).reshape(d,r*g*n)
    queries=np.stack([q[:,np.arange(g)*per_group+(rank%per_group)] for rank in range(r)],axis=1)
    kernel[:,slices['c']]=queries.reshape(d,r*g*n)
    # All token-dependent dynamics start at zero; no inherited Adam/adapter.
    logits=rng.normal(size=(h,r,n));simplex=np.exp(logits-logits.max(axis=1,keepdims=True));simplex/=simplex.sum(axis=1,keepdims=True)
    own=np.zeros((h,r,n),np.float32);own[np.arange(h),np.arange(h)%r,:]=1.
    dt=np.exp(rng.uniform(math.log(mamba.dt_min),math.log(mamba.dt_max),h)).astype(np.float32)
    dt=np.maximum(dt,mamba.dt_init_floor)
    scales={}
    for target,name in [('b_norm','k_norm'),('c_norm','q_norm')]:
        scale=np.asarray(arrays[prefix+'.'+name+'.weight'],np.float32)
        if scale.shape!=(n,):raise ValueError('teacher q/k norm width differs')
        scales[target]={'scale':scale.copy()}
    return dict(in_proj={'kernel':kernel},out_proj={'kernel':o.copy()},**scales,
        b_bias=np.full((h,r,n),bc_bias,np.float32),c_bias=np.full((h,r,n),bc_bias,np.float32),
        dt_bias=dt+np.log(-np.expm1(-dt)),mimo_x=simplex.astype(np.float32),
        mimo_o=own,mimo_z=np.ones_like(own),D=np.ones(h,np.float32),
        readout_norm={'scale':np.ones((h,n),np.float32)})


def forward(inputs,p,cfg):
    """Explicit full-strength normalization/tanh from the first clean step."""
    import jax
    import jax.numpy as jnp
    from extent.mamba3_transplant import mamba3_projection_slices
    from extent.layers.mamba3 import heavy_tail_activation
    from extent.pregate_recovery import pregate_scan
    c=cfg.mamba;h=int(cfg.hidden_size*c.expand)//c.head_dim;r=c.mimo_rank;g=c.groups;n=c.d_state
    slices=mamba3_projection_slices(cfg.hidden_size,c);dtype=getattr(jnp,cfg.compute_dtype)
    p=jax.tree.map(lambda value:jnp.asarray(value,dtype),p)
    proj=inputs.astype(dtype)@jnp.asarray(p['in_proj']['kernel'],dtype)
    parts={name:proj[...,s] for name,s in slices.items()};batch,length=inputs.shape[:2]
    x=parts['x'].reshape(batch,length,h,c.head_dim);z=parts['z'].reshape(x.shape)
    def bc(name,scale,bias):
        value=parts[name].reshape(batch,length,r,g,n).astype(jnp.float32)
        value=value*jax.lax.rsqrt(jnp.mean(value*value,axis=-1,keepdims=True)+cfg.rms_norm_eps)
        value=(value*jnp.asarray(p[scale]['scale'])).astype(dtype)
        value=jnp.repeat(value,h//g,axis=3).astype(jnp.float32)
        return value+jnp.asarray(p[bias]).transpose(1,0,2)[None,None]
    b=bc('b','b_norm','b_bias');cc=bc('c','c_norm','c_bias')
    dt=jax.nn.softplus(parts['dt'].astype(jnp.float32)+jnp.asarray(p['dt_bias']))
    decay=jnp.minimum(-heavy_tail_activation(parts['a']),-c.a_floor)
    y=pregate_scan(x,z,b,cc,dt,decay,parts['trap'],jnp.tanh(parts['angle'].astype(jnp.float32)),
        p['mimo_x'],p['mimo_z'],p['mimo_o'],p['D'],int(c.d_state*c.rope_fraction)//2,
        p['readout_norm']['scale'],jnp.asarray(1.,jnp.float32))
    return y.reshape(batch,length,-1).astype(dtype)@jnp.asarray(p['out_proj']['kernel'],dtype)


def alignment_loss(p,teacher_layer,cfg,teacher_input,teacher_output):
    """Frozen original Qwen residual/MLP; only clean Mamba coordinates differentiate.

    Explicit equal-relative-layer objective; not a claimed recovery of the
    competitor's unpublished trainer loss reduction.
    """
    import jax
    import jax.numpy as jnp
    from extent.plateau_forensics import norm,mlp
    middle=teacher_input+forward(norm(teacher_input,teacher_layer['input_layernorm'],cfg),p,cfg)
    prediction=middle+mlp(norm(middle,teacher_layer['post_attention_layernorm'],cfg),teacher_layer,cfg)
    target=jax.lax.stop_gradient(teacher_output.astype(jnp.float32))
    scale=jax.lax.stop_gradient(jnp.sqrt(jnp.mean(target*target)+1e-8))
    return jnp.mean(((prediction.astype(jnp.float32)-target)/scale)**2)
