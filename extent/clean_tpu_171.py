"""Clean QKVO alignment algebra. No Patient, checkpoint reconstruction or torch."""
import math
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import optax


def rms(x, p, eps):
    return (x.astype(jnp.float32)*jax.lax.rsqrt(jnp.mean(x.astype(jnp.float32)**2,-1,keepdims=True)+eps)*p['scale']).astype(x.dtype)


def mlp(h, p):
    dtype=h.dtype
    return (jax.nn.silu(h@p['gate_proj']['kernel'].astype(dtype))*(h@p['up_proj']['kernel'].astype(dtype)))@p['down_proj']['kernel'].astype(dtype)


def attention(h,p,c):
    from extent.qwen3_teacher import apply_qwen3_rope
    b,t,_=h.shape;heads=c.num_attention_heads;groups=c.num_key_value_heads;n=c.head_dim
    q=(h@p['q_proj']['kernel'].astype(h.dtype)).reshape(b,t,heads,n)
    k=(h@p['k_proj']['kernel'].astype(h.dtype)).reshape(b,t,groups,n)
    v=(h@p['v_proj']['kernel'].astype(h.dtype)).reshape(b,t,groups,n)
    positions=jnp.arange(t)[None]
    q=apply_qwen3_rope(rms(q,p['q_norm'],c.rms_norm_eps),positions,c.rope_theta)
    k=apply_qwen3_rope(rms(k,p['k_norm'],c.rms_norm_eps),positions,c.rope_theta)
    y=jax.nn.dot_product_attention(q,k,v,is_causal=True,implementation='xla')
    return y.reshape(b,t,-1)@p['o_proj']['kernel'].astype(h.dtype)


def teacher_block(h,p,c):
    middle=h+attention(rms(h,p['input_layernorm'],c.rms_norm_eps),p['self_attn'],c)
    return middle+mlp(rms(middle,p['post_attention_layernorm'],c.rms_norm_eps),p['mlp'])


def rotate(value,phase,n):
    pad=n//2-phase.shape[-1]
    cosine=jnp.pad(jnp.cos(phase),((0,0),(0,0),(0,0),(0,pad)),constant_values=1)[:,:,None]
    sine=jnp.pad(jnp.sin(phase),((0,0),(0,0),(0,0),(0,pad)))[:,:,None]
    a,b=jnp.split(value,2,-1)
    return jnp.concatenate((a*cosine-b*sine,b*cosine+a*sine),-1)


def ssd_chunk(carry,inputs,p):
    state,pk,pv,phase=carry
    x,z,b,c,dt,decay,trap,angles=inputs
    batch,width,h,n=x.shape;r=b.shape[2];groups=b.shape[3]
    logalpha=dt*decay;prefix=jnp.cumsum(logalpha,axis=1).transpose(0,2,1)
    phases=phase[:,None]+jnp.cumsum(jnp.pi*jnp.tanh(angles)[:,:,None]*dt[...,None],axis=1)
    kb=rotate(jnp.repeat(b,h//groups,3)+p['b_bias'].transpose(1,0,2)[None,None],phases,n)
    qc=rotate(jnp.repeat(c,h//groups,3)+p['c_bias'].transpose(1,0,2)[None,None],phases,n)
    prevk=jnp.concatenate((pk[:,None],kb[:,:-1]),1)
    prevx=jnp.concatenate((pv[:,None],x[:,:-1].astype(jnp.float32)),1)
    now=x.astype(jnp.float32)[:,:,None]*p['mimo_x'].transpose(1,0,2)[None,None]
    prev=prevx[:,:,None]*p['mimo_x'].transpose(1,0,2)[None,None]
    fraction=jax.nn.sigmoid(trap);gamma=fraction*dt;beta=(1-fraction)*dt*jnp.exp(logalpha)
    def flat(v):return v.transpose(0,3,1,2,4).reshape(batch,h,width*r,n)
    q=flat(qc)
    diff=prefix[:,:,:,None]-prefix[:,:,None,:]
    mask=jnp.exp(jnp.where(jnp.tril(jnp.ones((width,width),bool)),diff,-jnp.inf))
    def read(keys,values,coefficient):
        weights=mask*coefficient.transpose(0,2,1)[:,:,None,:]
        weights=jnp.repeat(jnp.repeat(weights,r,-2),r,-1)
        score=(q@flat(keys).swapaxes(-1,-2))*weights
        return score@flat(values)
    carry_read=q@state.swapaxes(-1,-2)
    carry_read*=jnp.repeat(jnp.exp(prefix),r,-1)[...,None]
    readout=(read(kb,now,gamma)+read(prevk,prev,beta)+carry_read).reshape(batch,h,width,r,n).transpose(0,2,3,1,4)
    readout+=p['D'][None,None,None,:,None]*now
    readout=readout*jax.lax.rsqrt(jnp.mean(readout**2,-1,keepdims=True)+1e-5)*p['readout_norm']['scale'][None,None,None]
    output=jnp.sum(readout*jax.nn.silu(z.astype(jnp.float32)[:,:,None]*p['mimo_z'].transpose(1,0,2)[None,None])*p['mimo_o'].transpose(1,0,2)[None,None],2)
    end_weights=jnp.exp(prefix[:,:,-1,None]-prefix).transpose(0,2,1)
    end=jnp.exp(prefix[:,:,-1])[...,None,None]*state
    for keys,values,coefficient in ((kb,now,gamma),(prevk,prev,beta)):
        end+=flat(values).swapaxes(-1,-2)@flat(keys*(end_weights*coefficient)[:,:,None,:,None])
    return (end,kb[:,-1],x[:,-1].astype(jnp.float32),phases[:,-1]),output.astype(x.dtype)


def mamba_forward(u,p,c,*,block=64):
    from extent.mamba3_transplant import mamba3_projection_slices
    m=c.mamba;h=int(c.hidden_size*m.expand)//m.head_dim;r=m.mimo_rank;g=m.groups;n=m.d_state
    batch,length=u.shape[:2]
    if length%block:raise ValueError('context must be divisible by SSD chunk')
    slices=mamba3_projection_slices(c.hidden_size,m)
    projected=u@p['in_proj']['kernel'].astype(u.dtype)
    parts={name:projected[...,s] for name,s in slices.items()}
    x=parts['x'].reshape(batch,length,h,n);z=parts['z'].reshape(x.shape)
    def bc(name,scale):
        value=parts[name].reshape(batch,length,r,g,n)
        return rms(value,p[scale],c.rms_norm_eps).astype(jnp.float32)
    dt=jax.nn.softplus(parts['dt'].astype(jnp.float32)+p['dt_bias'])
    a=parts['a'].astype(jnp.float32)
    decay=-jnp.maximum(jnp.maximum(a,0)+1/(1-jnp.minimum(a,0)),1e-4)
    values=(x,z,bc('b','b_norm'),bc('c','c_norm'),dt,decay,parts['trap'].astype(jnp.float32),parts['angle'].astype(jnp.float32))
    chunks=tuple(v.reshape(batch,length//block,block,*v.shape[2:]).swapaxes(0,1) for v in values)
    carry=(jnp.zeros((batch,h,n,n),jnp.float32),jnp.zeros((batch,r,h,n),jnp.float32),
        jnp.zeros((batch,h,n),jnp.float32),jnp.zeros((batch,h,n//4),jnp.float32))
    _,output=jax.lax.scan(jax.checkpoint(lambda state,data:ssd_chunk(state,data,p)),carry,chunks)
    output=output.swapaxes(0,1).reshape(batch,length,h*n)
    return output@p['out_proj']['kernel'].astype(u.dtype)


def student_block(h,p,q,c):
    middle=h+mamba_forward(rms(h,p['input_layernorm'],c.rms_norm_eps),q,c)
    return middle+mlp(rms(middle,p['post_attention_layernorm'],c.rms_norm_eps),p['mlp'])


def relative_loss(pred,target):
    energy=jax.lax.stop_gradient(jnp.maximum(jnp.mean(target.astype(jnp.float32)**2,(1,2)),1e-8))
    return jnp.mean(jnp.mean((pred.astype(jnp.float32)-target.astype(jnp.float32))**2,(1,2))/energy)


def make_grad(c,tc):
    def gradients(q,teacher,embedding,tokens):
        hidden=embedding[tokens].astype(getattr(jnp,c.compute_dtype))
        def layer(h,inputs):
            p,m=inputs
            target=jax.lax.stop_gradient(teacher_block(h,p,tc))
            def loss(m):return relative_loss(student_block(h,p,m,c),target)
            value,grad=jax.value_and_grad(jax.checkpoint(loss))(m)
            return target,(value,grad)
        _,(losses,grads)=jax.lax.scan(layer,hidden,(teacher,q))
        grads=jax.tree.map(lambda v:v/len(losses),grads)
        return jnp.mean(losses),grads,losses
    return gradients


def optimizer():
    from extent.stable_gradient_clip import stable_clip_by_global_norm
    return optax.chain(stable_clip_by_global_norm(1.),optax.scale_by_adam(b1=.9,b2=.95,eps=1e-8))


def lr(step,total):
    warm=min(46,max(1,total//4))
    return 1e-3*min((step+1)/warm,1.)*(1. if step<warm else .1+.45*(1+math.cos(math.pi*min(1.,(step-warm)/max(1,total-warm-1)))))


def layout(tree,mesh,*,stacked=True):
    from jax.sharding import NamedSharding,PartitionSpec as P
    def one(x):
        spec=[None]*x.ndim
        axes=[i for i,n in enumerate(x.shape) if n%mesh.size==0 and n>=mesh.size and (not stacked or i!=0)]
        if axes:spec[max(axes,key=lambda i:x.shape[i])]='model'
        return NamedSharding(mesh,P(*spec))
    return jax.tree.map(one,tree)


def initialize_host(directory,progress):
    """Only pinned original Qwen tensors. No historical training-state reader."""
    from flax import traverse_util
    from extent.config import HybridConfig,Mamba3Config,MLAConfig
    from extent.qwen_source import QWEN3_1_7B_BASE as source,teacher_config_from_spec
    from extent.weight_mapping import QwenCheckpointReader,teacher_qwen_mappings,expected_qwen_shape
    from extent.clean_transplant_170 import initialize
    reader=QwenCheckpointReader(directory)
    tc=teacher_config_from_spec(source,param_dtype='bfloat16',compute_dtype='bfloat16',remat_policy='none')
    c=HybridConfig(vocab_size=source.vocab_size,hidden_size=source.hidden_size,intermediate_size=source.intermediate_size,
        num_layers=28,attention_layer_indices=(6,13,20,27),rms_norm_eps=source.rms_norm_eps,
        tie_word_embeddings=True,param_dtype='float32',compute_dtype='bfloat16',
        mamba=Mamba3Config(d_state=128,expand=1.,head_dim=128,groups=8,mimo_rank=4),mla=MLAConfig(implementation='qwen3_gqa'))
    flat={}
    for entry in teacher_qwen_mappings(tc):
        value=reader.read(entry.source)
        if value.shape!=expected_qwen_shape(entry,tc):raise ValueError('original tensor shape mismatch')
        flat[tuple(entry.target.split('/'))]=value.T if entry.transform=='transpose' else value
    tree=traverse_util.unflatten_dict(flat)
    bank=[]
    for i in range(27):
        progress('initialize_QKVO_layer_'+str(i))
        arrays={f'model.layers.{i}.self_attn.{name}.weight':reader.read(f'model.layers.{i}.self_attn.{name}.weight')
                for name in ('q_proj','k_proj','v_proj','o_proj','q_norm','k_norm')}
        q=initialize(arrays,tc,c.mamba,i)
        # Original feature widths, exact own-head identity; initializer never fits/crops.
        qk=q['in_proj']['kernel'];d=source.hidden_size;n=source.head_dim
        assert np.array_equal(q['out_proj']['kernel'],arrays[f'model.layers.{i}.self_attn.o_proj.weight'].T)
        assert np.array_equal(qk[:,d:2*d],np.repeat(arrays[f'model.layers.{i}.self_attn.v_proj.weight'].T.reshape(d,8,n),2,1).reshape(d,d))
        b=qk[:,2*d:2*d+4*8*n].reshape(d,4,8,n)
        cc=qk[:,2*d+4*8*n:2*d+8*8*n].reshape(d,4,8,n)
        original_k=np.asarray(arrays[f'model.layers.{i}.self_attn.k_proj.weight'],np.float32).T.reshape(d,8,n)
        original_q=np.asarray(arrays[f'model.layers.{i}.self_attn.q_proj.weight'],np.float32).T.reshape(d,16,n)
        assert np.array_equal(b,np.broadcast_to(original_k[:,None],b.shape))
        for head in range(16):assert np.array_equal(cc[:,head%4,head//2],original_q[:,head])
        for target,name in [('b_norm','k_norm'),('c_norm','q_norm')]:
            assert np.array_equal(q[target]['scale'],arrays[f'model.layers.{i}.self_attn.{name}.weight'])
        bank.append(q)
    stack=lambda rows:jax.tree.map(lambda *values:np.stack(values),*rows)
    return c,tc,stack(bank),stack([tree[f'layers_{i}'] for i in range(27)]),tree['layers_27'],tree['embed_tokens']['embedding'],tree['norm']
