from types import SimpleNamespace
import json,os,sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from extent.config import Mamba3Config
from extent.clean_transplant_170 import initialize,forward
from extent.clean_tpu_171 import mamba_forward,relative_loss,teacher_block,attention,make_grad,optimizer
from extent.qwen3_teacher import Qwen3TeacherConfig,Qwen3GQAAttention


def fixture():
    tc=Qwen3TeacherConfig(vocab_size=37,hidden_size=16,intermediate_size=32,num_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=4,param_dtype='float32',compute_dtype='float32')
    m=Mamba3Config(head_dim=4,d_state=4,groups=2,expand=1.,mimo_rank=4)
    c=SimpleNamespace(hidden_size=16,mamba=m,compute_dtype='float32',rms_norm_eps=1e-6)
    rng=np.random.default_rng(171)
    arrays={f'model.layers.0.self_attn.{name}.weight':rng.normal(0,.1,shape).astype(np.float32)
        for name,shape in [('q_proj',(16,16)),('k_proj',(8,16)),('v_proj',(8,16)),('o_proj',(16,16)),('q_norm',(4,)),('k_norm',(4,))]}
    q=initialize(arrays,tc,m,0)
    q['in_proj']['kernel'][:,-13:]=rng.normal(0,.01,(16,13))
    h=jnp.asarray(rng.normal(0,.2,(2,128,16)),jnp.float32)
    return c,tc,q,h


def test_chunked_values_gradients_nonzero_rotation_and_cross_chunk_carry():
    c,tc,q,h=fixture()
    actual=mamba_forward(h,q,c);expected=forward(h,q,c)
    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-4)
    f=lambda p:jnp.mean(mamba_forward(h,p,c)**2)
    g=lambda p:jnp.mean(forward(h,p,c)**2)
    a=jax.grad(f)(q);b=jax.grad(g)(q)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-3)
    assert float(jnp.linalg.norm(a['in_proj']['kernel'][:,-1]))>0


def test_teacher_attention_matches_existing_qwen_reference_and_causality():
    c,tc,q,h=fixture()
    module=Qwen3GQAAttention(tc)
    p=module.init(jax.random.key(1),h,jnp.arange(128)[None])['params']
    expected=module.apply({'params':p},h,jnp.arange(128)[None])
    actual=attention(h,p,tc)
    np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-4)
    changed=h.at[:,64:].add(3)
    np.testing.assert_allclose(attention(changed,p,tc)[:,:64],actual[:,:64],atol=1e-6)


def test_relative_loss_microbatch_partition():
    x=jnp.arange(3*64*4,dtype=jnp.float32).reshape(3,64,4)/100
    y=x*.7+.2
    full=jax.grad(lambda w:relative_loss(w*x,y))(jnp.asarray(.8))
    split=jax.grad(lambda w:(relative_loss(w*x[:2],y[:2])*2+relative_loss(w*x[2:],y[2:]))/3)(jnp.asarray(.8))
    np.testing.assert_allclose(full,split,atol=1e-6)


def test_all_layer_scan_gradients_and_one_optimizer_clock():
    from extent.layers.common import SwiGLU
    c,tc,q,h=fixture();h=h[:1,:64]
    ap=Qwen3GQAAttention(tc).init(jax.random.key(2),h,jnp.arange(64)[None])['params']
    mp=SwiGLU(16,32,jnp.float32,jnp.float32).init(jax.random.key(3),h)['params']
    p=dict(self_attn=ap,mlp=mp,input_layernorm={'scale':jnp.ones(16)},post_attention_layernorm={'scale':jnp.ones(16)})
    stack=lambda value:jax.tree.map(lambda x:jnp.stack([x,x]),value)
    bank=stack(q);teacher=stack(p);emb=jax.random.normal(jax.random.key(4),(37,16))
    tokens=jnp.arange(64)[None]%37
    before=jax.tree.map(np.array,teacher)
    loss,g,layers=jax.jit(make_grad(c,tc))(bank,teacher,emb,tokens)
    assert np.isfinite(float(loss)) and layers.shape==(2,)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(g))
    tx=optimizer();state=tx.init(bank);updates,state=tx.update(g,state,bank)
    assert int(state[1].count)==1
    for x,y in zip(jax.tree.leaves(before),jax.tree.leaves(teacher)):np.testing.assert_array_equal(x,y)


def test_supervisor_environment_and_timeout_stack_capture(tmp_path):
    import engine_tpu171 as runner
    env=runner.environment(tmp_path,'tpu,cpu')
    assert all(env[k]=='0' for k in ('USE_TORCH','USE_TF','USE_FLAX'))
    command=[sys.executable,'-c','import time;time.sleep(10)']
    row=runner.wait_child(command,tmp_path,tmp_path/'log',.1)
    assert row['status']=='phase_timeout' and row['returncode']!=0


def test_sharding_layout_does_not_partition_layer_axis():
    from jax.sharding import Mesh
    from extent.clean_tpu_171 import layout
    mesh=Mesh(np.array(jax.devices()[:1]),('model',))
    q={'x':jnp.ones((27,16,32))}
    spec=layout(q,mesh)['x'].spec
    assert spec[0] is None


def test_eight_virtual_devices_real_sharded_gradient_and_update(tmp_path):
    import subprocess
    code='''
import numpy as np,jax,jax.numpy as jnp,optax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from tests.test_clean_tpu_171 import fixture
from extent.clean_tpu_171 import layout,make_grad,optimizer
from extent.qwen3_teacher import Qwen3GQAAttention
from extent.layers.common import SwiGLU
c,tc,q,h=fixture();h=h[:1,:64]
ap=Qwen3GQAAttention(tc).init(jax.random.key(1),h,jnp.arange(64)[None])['params']
mp=SwiGLU(16,32,jnp.float32,jnp.float32).init(jax.random.key(2),h)['params']
p=dict(self_attn=ap,mlp=mp,input_layernorm={'scale':jnp.ones(16)},post_attention_layernorm={'scale':jnp.ones(16)})
stack=lambda t:jax.tree.map(lambda x:jnp.stack([x,x]),t)
q=stack(q);p=stack(p);mesh=Mesh(np.array(jax.devices()),('model',));assert mesh.size==8
ql=layout(q,mesh);pl=layout(p,mesh);rep=NamedSharding(mesh,P())
q=jax.device_put(q,ql);p=jax.device_put(p,pl)
emb=jax.device_put(jnp.ones((40,16)),layout(jnp.ones((40,16)),mesh,stacked=False))
tokens=jax.device_put(jnp.arange(64)[None]%37,rep)
loss,g,layers=jax.jit(make_grad(c,tc),out_shardings=(rep,ql,rep))(q,p,emb,tokens)
assert np.isfinite(float(loss))
tx=optimizer();s=tx.init(q)
def step(q,s,g):
 u,s=tx.update(g,s,q);return optax.apply_updates(q,jax.tree.map(lambda x:-1e-3*x,u)),s
q,s=jax.jit(step,out_shardings=(ql,layout(s,mesh,stacked=False)))(q,s,g)
assert int(s[1].count)==1
assert all(len(x.addressable_shards)==8 for x in jax.tree.leaves(q))
print('eight_device_clean_gradient_update=PASS')
'''
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=8')
    result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr


def test_checkpoint_restores_own_clock_and_rejects_changed_source(tmp_path):
    from extent.recovery_durability import RecoveryCheckpointStore
    q={'w':jnp.ones((2,4))};tx=optimizer();state=tx.init(q)
    payload=dict(parameters=q,optimizer=state,cursor=np.asarray(16384,np.int64))
    contract=dict(source='original-Qwen',context=1024,effective_batch=16)
    store=RecoveryCheckpointStore(tmp_path,'exp171-test')
    store.save('ALIGN',contract,payload,1)
    saved,meta=store.restore('ALIGN',contract,payload)
    assert meta['step']==1 and int(saved['cursor'])==16384
    with pytest.raises(ValueError,match='contract mismatch'):
        store.restore('ALIGN',dict(contract,source='legacy'),payload)
