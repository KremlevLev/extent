from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from extent.config import Mamba3Config
from extent.clean_transplant_170 import initialize,geometry
from extent.mamba3_transplant import mamba3_projection_slices,_head_group_projection


def source_and_arrays():
    s=SimpleNamespace(hidden_size=16,num_attention_heads=4,num_key_value_heads=2,head_dim=4)
    c=Mamba3Config(head_dim=4,d_state=4,groups=2,expand=1.,mimo_rank=4)
    rng=np.random.default_rng(170)
    arrays={f'model.layers.0.self_attn.{name}.weight':rng.normal(size=shape).astype(np.float32)
        for name,shape in [('q_proj',(16,16)),('k_proj',(8,16)),('v_proj',(8,16)),('o_proj',(16,16)),('q_norm',(4,)),('k_norm',(4,))]}
    return s,c,arrays


def test_query_kv_output_and_norm_ownership_no_pooling_or_crop():
    s,c,a=source_and_arrays();p=initialize(a,s,c,0);slices=mamba3_projection_slices(16,c)
    kernel=p['in_proj']['kernel'];q=a['model.layers.0.self_attn.q_proj.weight'].T.reshape(16,4,4)
    k=a['model.layers.0.self_attn.k_proj.weight'].T.reshape(16,2,4)
    cc=kernel[:,slices['c']].reshape(16,4,2,4);b=kernel[:,slices['b']].reshape(16,4,2,4)
    for head in range(4):
        np.testing.assert_array_equal(cc[:,head%4,head//2],q[:,head])
        for rank in range(4):np.testing.assert_array_equal(b[:,rank,head//2],k[:,head//2])
    np.testing.assert_array_equal(kernel[:,slices['x']],np.repeat(a['model.layers.0.self_attn.v_proj.weight'].T.reshape(16,2,4),2,axis=1).reshape(16,16))
    np.testing.assert_array_equal(p['out_proj']['kernel'],a['model.layers.0.self_attn.o_proj.weight'].T)
    np.testing.assert_array_equal(p['b_norm']['scale'],a['model.layers.0.self_attn.k_norm.weight'])
    for name in ('dt','a','trap','angle'):assert not np.any(kernel[:,slices[name]])
    np.testing.assert_allclose(p['mimo_x'].sum(axis=1),1,atol=2e-7)
    assert np.all((p['mimo_o']==0)|(p['mimo_o']==1)) and np.all(p['mimo_o'].sum(axis=1)==1)
    assert not np.allclose(p['mimo_x'][:,0],p['mimo_x'][:,1])
    assert np.isfinite(p['dt_bias']).all()


def test_legacy_geometry_refused_instead_of_silent_projection_resize():
    s,c,_=source_and_arrays()
    with pytest.raises(ValueError,match='no crop/pooling'):geometry(s,replace(c,groups=1))
    with pytest.raises(ValueError,match='no crop/pooling'):geometry(s,replace(c,d_state=2))
    with pytest.raises(ValueError,match='multiple'):geometry(s,replace(c,mimo_rank=3))


def test_old_head_averaging_has_nullspace_even_when_mimo_lift_is_exact():
    # Mathematical counterexample, not a measurement of actual Qwen weight loss.
    a=np.arange(8,dtype=np.float32).reshape(2,4)
    kernel=np.stack([a,-a],axis=1).reshape(2,8)
    old=_head_group_projection(kernel,2,4,4,4,copy_channels=True)
    assert np.linalg.norm(kernel)>0 and np.linalg.norm(old)==0


def test_clean_initial_scan_matches_independent_numpy_no_rotation_reference():
    import jax.numpy as jnp
    from extent.clean_transplant_170 import forward
    s,c,a=source_and_arrays();p=initialize(a,s,c,0)
    cfg=SimpleNamespace(hidden_size=16,mamba=c,compute_dtype='float32',rms_norm_eps=1e-6)
    inputs=np.random.default_rng(13).normal(size=(2,7,16)).astype(np.float32)
    slices=mamba3_projection_slices(16,c);parts={k:(inputs@p['in_proj']['kernel'])[...,sl] for k,sl in slices.items()}
    x=parts['x'].reshape(2,7,4,4);z=parts['z'].reshape(x.shape)
    def bc(name,scale,bias):
        v=parts[name].reshape(2,7,4,2,4);v=v/np.sqrt(np.mean(v*v,axis=-1,keepdims=True)+1e-6)*p[scale]['scale']
        return np.repeat(v,2,axis=3)+p[bias].transpose(1,0,2)[None,None]
    b=bc('b','b_norm','b_bias');cc=bc('c','c_norm','c_bias')
    dt=np.logaddexp(0,p['dt_bias']);alpha=np.exp(-dt);state=np.zeros((2,4,4,4),np.float32)
    previous_b=np.zeros((2,4,4,4),np.float32);previous_v=np.zeros_like(previous_b);outputs=[]
    for t in range(7):
        value=x[:,t,None]*p['mimo_x'].transpose(1,0,2)[None]
        current=np.einsum('brhp,brhn->bhpn',value,b[:,t])
        previous=np.einsum('brhp,brhn->bhpn',previous_v,previous_b)
        state=alpha[None,:,None,None]*state+.5*dt[None,:,None,None]*(current+alpha[None,:,None,None]*previous)
        read=np.einsum('bhpn,brhn->brhp',state,cc[:,t])+value
        read=read/np.sqrt(np.mean(read*read,axis=-1,keepdims=True)+1e-5)
        gate=z[:,t,None]/(1+np.exp(-z[:,t,None]))
        outputs.append(np.sum(read*gate*p['mimo_o'].transpose(1,0,2)[None],axis=1).reshape(2,16)@p['out_proj']['kernel'])
        previous_b=b[:,t];previous_v=value
    np.testing.assert_allclose(np.asarray(forward(jnp.asarray(inputs),p,cfg)),np.stack(outputs,axis=1),atol=3e-5,rtol=3e-5)


def test_actual_clean_block_alignment_has_finite_live_dynamics_gradients():
    import jax
    import jax.numpy as jnp
    from extent import tiny_config,HybridForCausalLM
    from extent.model import _attention_source_config
    from extent.plateau_forensics import norm,mixer,mlp
    from extent.clean_transplant_170 import alignment_loss
    original=tiny_config();cfg=replace(original,mla=replace(original.mla,implementation='qwen3_gqa'),
        param_dtype='float32',compute_dtype='float32',remat_policy='none')
    tc=_attention_source_config(cfg);cfg=replace(cfg,mamba=replace(cfg.mamba,head_dim=tc.head_dim,d_state=tc.head_dim,
        groups=tc.num_key_value_heads,expand=1.,mimo_rank=4))
    teacher_cfg=replace(cfg,attention_layer_indices=tuple(range(cfg.num_layers)))
    tokens=jnp.arange(8,dtype=jnp.int32)[None]
    tree=HybridForCausalLM(teacher_cfg).init(jax.random.key(170),tokens)['params'];layer=tree['layers_0']
    arrays={}
    for name in ('q_proj','k_proj','v_proj','o_proj'):
        arrays[f'model.layers.0.self_attn.{name}.weight']=np.asarray(layer['self_attn'][name]['kernel']).T
    for name in ('q_norm','k_norm'):arrays[f'model.layers.0.self_attn.{name}.weight']=np.asarray(layer['self_attn'][name]['scale'])
    p=initialize(arrays,tc,cfg.mamba,0);h=tree['embed_tokens']['embedding'][tokens]
    middle=h+mixer(norm(h,layer['input_layernorm'],cfg),layer,cfg,0,tc)
    target=middle+mlp(norm(middle,layer['post_attention_layernorm'],cfg),layer,cfg)
    before=jax.tree.map(np.array,layer)
    value,g=jax.jit(jax.value_and_grad(lambda p:alignment_loss(p,layer,cfg,h,target)))(p)
    assert np.isfinite(float(value)) and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g))
    slices=mamba3_projection_slices(cfg.hidden_size,cfg.mamba)
    assert np.linalg.norm(np.asarray(g['in_proj']['kernel'])[:,slices['dt']])>0
    assert np.linalg.norm(np.asarray(g['in_proj']['kernel'])[:,slices['angle']])>0
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(layer)):np.testing.assert_array_equal(a,b)
