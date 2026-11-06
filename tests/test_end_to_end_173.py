from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from extent.end_to_end_173 import head_loss,make_grad,hidden,RETAINED
from extent.config import Mamba3Config
from extent.qwen3_teacher import Qwen3TeacherConfig


def test_tiled_exact_ce_kl_values_gradients_and_teacher_stop():
    rng=np.random.default_rng(173)
    s=jnp.asarray(rng.normal(size=(2,9,8)),jnp.float32)
    t=s*.7+.1;emb=jnp.asarray(rng.normal(size=(19,8)),jnp.float32)
    labels=jnp.asarray(rng.integers(0,19,size=(2,9)))
    def dense(s):
        lp=jax.nn.log_softmax(s@emb.T,-1);lt=jax.nn.log_softmax(t@emb.T,-1)
        ce=-jnp.take_along_axis(lp,labels[...,None],-1).mean()
        kl=jnp.sum(jnp.exp(lt)*(lt-lp),-1).mean()
        return .5*(ce+kl)
    value,metrics=head_loss(s,t,emb,labels,tile=4)
    np.testing.assert_allclose(value,dense(s),rtol=1e-6)
    np.testing.assert_allclose(jax.grad(lambda s:head_loss(s,t,emb,labels,tile=4)[0])(s),jax.grad(dense)(s),atol=1e-7)
    assert np.all(np.asarray(jax.grad(lambda t:head_loss(s,t,emb,labels,tile=4)[0])(t))==0)


def test_composed_gradient_crosses_layers_but_not_retained_mixers():
    from extent.clean_transplant_170 import initialize
    from extent.clean_tpu_171 import teacher_block,rms
    rng=np.random.default_rng(9);d=8;n=4;h=2;g=1
    tc=Qwen3TeacherConfig(vocab_size=19,hidden_size=d,intermediate_size=12,num_layers=28,
        num_attention_heads=h,num_key_value_heads=g,head_dim=n,param_dtype='float32',compute_dtype='bfloat16')
    c=SimpleNamespace(hidden_size=d,rms_norm_eps=1e-6,mamba=Mamba3Config(head_dim=n,d_state=n,groups=g,expand=1.,mimo_rank=4))
    def matrix(a,b):return jnp.asarray(rng.normal(0,.08,(a,b)),jnp.float32)
    attn={name:dict(kernel=matrix(d,width)) for name,width in [('q_proj',d),('k_proj',n),('v_proj',n),('o_proj',d)]}
    attn.update(q_norm=dict(scale=jnp.ones(n)),k_norm=dict(scale=jnp.ones(n)))
    p=dict(self_attn=attn,input_layernorm=dict(scale=jnp.ones(d)),post_attention_layernorm=dict(scale=jnp.ones(d)),
        mlp=dict(gate_proj=dict(kernel=matrix(d,12)),up_proj=dict(kernel=matrix(d,12)),down_proj=dict(kernel=matrix(12,d))))
    arrays={f'model.layers.0.self_attn.{k}.weight':np.array(v['kernel']).T if 'kernel' in v else np.array(v['scale']) for k,v in attn.items()}
    q0=initialize(arrays,tc,c.mamba,0)
    q=jax.tree.map(lambda x:jnp.stack([x]*27),q0);teacher=jax.tree.map(lambda x:jnp.stack([x]*27),p)
    emb=matrix(19,d).astype(jnp.bfloat16);norm=dict(scale=jnp.ones(d));tokens=jnp.arange(64)[None]%19
    import os
    if os.environ.get('EXP173_VIRTUAL8'):
        from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
        from extent.clean_tpu_171 import layout
        mesh=Mesh(np.array(jax.devices()),('model',));assert mesh.size==8
        q=jax.device_put(q,layout(q,mesh));teacher=jax.device_put(teacher,layout(teacher,mesh))
        emb=jax.device_put(emb,layout(emb,mesh,stacked=False))
        p=jax.device_put(p,layout(p,mesh,stacked=False));norm=jax.device_put(norm,layout(norm,mesh,stacked=False))
        rep=NamedSharding(mesh,P())
        fn=jax.jit(make_grad(c,tc),out_shardings=(rep,layout(q,mesh),rep))
    else:fn=jax.jit(make_grad(c,tc))
    value,grad,metrics=fn(q,teacher,emb,p,norm,tokens)
    assert np.isfinite(value) and metrics.shape==(2,)
    norms=np.array([sum(float(jnp.sum(x[i]**2)) for x in jax.tree.leaves(grad)) for i in range(27)])
    assert np.all(norms[[i for i in RETAINED if i<27]]==0)
    assert np.all(norms[[i for i in range(27) if i not in RETAINED]]>0)
    # Composed student receives its own prefix, not teacher-prefix local targets.
    a=hidden(q,teacher,p,emb,norm,tokens,c,tc)
    q2=jax.tree.map(lambda x:x.at[0].multiply(0),q)
    b=hidden(q2,teacher,p,emb,norm,tokens,c,tc)
    assert float(jnp.linalg.norm(a.astype(jnp.float32)-b.astype(jnp.float32)))>0


def test_eight_virtual_chips_composed_gradient():
    import os,subprocess,sys
    from pathlib import Path
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=8',EXP173_VIRTUAL8='1')
    file=str(Path(__file__).resolve())
    code=f'import runpy; runpy.run_path({file!r})["test_composed_gradient_crosses_layers_but_not_retained_mixers"]()'
    row=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=180)
    assert row.returncode==0,row.stdout+row.stderr


def test_stage2_cli_is_separate_and_default_source_root():
    import engine_tpu173
    a=engine_tpu173.parser().parse_args([])
    assert a.work_dir=='/dev/shm/extent173' and a.total_updates==12208
    assert engine_tpu173.PREFIX=='experiments/exp173-gqa7-end-to-end-v1'
