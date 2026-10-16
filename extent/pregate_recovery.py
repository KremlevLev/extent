"""EXP129 isolated pre-gate headwise RMSNorm intervention.

Mathematical placement audited against arianraje's Apache-2.0 HF implementation
at b946069890837bbe9b5dd75e2d118a7e98d2d193, modeling_qwen3_mamba3.py.
Independent JAX implementation derived from OUR reference recurrence below;
no external Python executed. This is a standard upstream normalization choice,
not a claim that the normalization itself is new. Legacy kernels stay untouched.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from extent.layers.mamba3 import (ScanState, _rotate_mimo_halves,
    _rotate_mimo_sequence_halves, _trapezoidal_dual_mask)
from extent.layers.common import dtype_from_name
from extent.plateau_forensics import project_mamba, norm, mixer, mlp, plain
from extent.t4_scale_recovery import Engine as ScaleEngine, initialize as base_initialize, corrected_layer, log_rms
from extent.stable_gradient_clip import stable_clip_by_global_norm

TARGET_STEPS=32768
RAMP_STEPS=2048
EVAL_STEPS=(0,128,512,1024,2048,4096,8192,16384,24576,32768)


def arm_for(experiment,worker):
    if experiment!=129 or worker not in (0,1):raise ValueError('EXP129 workers0/1 required')
    return dict(name=('R32-CONTROL','R32-PREGATE-RMS')[worker],lora=True,
        scales=False,alpha=0.,penalty=0.,pregate=bool(worker),norm_eps=1e-5,
        norm_lr=1e-4,mlp=False)


def normalize_readout(value,weight,strength,eps=1e-5):
    # Last axis P only: no time, batch, rank or head aggregation. Shared H,P
    # weights across ranks, exactly as the competitor's fused-kernel contract.
    x=value.astype(jnp.float32)
    scale=jnp.maximum(jnp.max(jnp.abs(x),axis=-1,keepdims=True),math.sqrt(eps))
    logden=jnp.log(scale)+.5*jnp.log(jnp.mean((x/scale)**2,axis=-1,keepdims=True)+eps/scale/scale)
    # Geometric identity -> RMSNorm; full chain rule, no detached coefficient.
    return x*jnp.exp(-strength*logden)*(1.+strength*(weight-1.))


def pregate_scan(
    x: jax.Array,
    z: jax.Array,
    b: jax.Array,
    c: jax.Array,
    dt: jax.Array,
    decay: jax.Array,
    trap: jax.Array,
    angle_step: jax.Array,
    mimo_x: jax.Array,
    mimo_z: jax.Array,
    mimo_o: jax.Array,
    skip: jax.Array,
    rotary_pairs: int,
    weight: jax.Array,
    strength: jax.Array,
    eps: float = 1e-5,
    details: bool = False,
) -> jax.Array:
    """Scan one sequence.

    Shapes: x/z=[B,L,H,P], b/c=[B,L,R,H,N], scalar terms=[B,L,H].
    The recurrent state is float32 for stability; returned activations match x.
    """
    batch, _, heads, head_dim = x.shape
    rank, state_dim = b.shape[2], b.shape[-1]
    initial = ScanState(
        ssm=jnp.zeros((batch, heads, head_dim, state_dim), jnp.float32),
        previous_k=jnp.zeros((batch, rank, heads, state_dim), jnp.float32),
        previous_v=jnp.zeros((batch, heads, head_dim), jnp.float32),
        angle=jnp.zeros((batch, heads, rotary_pairs), jnp.float32),
    )
    mimo_x_rank_first = jnp.swapaxes(mimo_x, 0, 1).astype(jnp.float32)
    mimo_z_rank_first = jnp.swapaxes(mimo_z, 0, 1).astype(jnp.float32)
    mimo_o_rank_first = jnp.swapaxes(mimo_o, 0, 1).astype(jnp.float32)

    time_major = tuple(jnp.swapaxes(t, 0, 1) for t in (x, z, b, c, dt, decay, trap, angle_step))

    def step(state: ScanState, inputs: tuple[jax.Array, ...]):
        x_t, z_t, b_t, c_t, dt_t, decay_t, trap_t, angle_t = inputs
        dt_t = dt_t.astype(jnp.float32)
        alpha = jnp.exp(decay_t.astype(jnp.float32) * dt_t)
        gamma = jax.nn.sigmoid(trap_t.astype(jnp.float32)) * dt_t
        beta = (1.0 - jax.nn.sigmoid(trap_t.astype(jnp.float32))) * dt_t * alpha

        angle_increment = jnp.pi * angle_t[:, None, :].astype(jnp.float32)
        new_angle = state.angle + angle_increment * dt_t[..., None]
        b_rot = _rotate_mimo_halves(b_t.astype(jnp.float32), new_angle, rotary_pairs)
        c_rot = _rotate_mimo_halves(c_t.astype(jnp.float32), new_angle, rotary_pairs)

        # Rank-specific value streams; the MIMO rank is reduced only after readout.
        value_now = x_t[:, None].astype(jnp.float32) * mimo_x_rank_first[None]
        value_prev = state.previous_v[:, None] * mimo_x_rank_first[None]
        injection = jnp.einsum("brhp,brhn->bhpn", gamma[:, None, :, None] * value_now, b_rot)
        previous = jnp.einsum("brhp,brhn->bhpn", beta[:, None, :, None] * value_prev, state.previous_k)
        ssm = alpha[:, :, None, None] * state.ssm + injection + previous
        y_rank = jnp.einsum("bhpn,brhn->brhp", ssm, c_rot)
        y_rank += skip[None, None, :, None].astype(jnp.float32) * value_now
        before=y_rank
        y_rank = normalize_readout(y_rank, weight[None, None], strength, eps)
        gated = y_rank * jax.nn.silu(
            z_t[:, None].astype(jnp.float32) * mimo_z_rank_first[None]
        )
        y = jnp.sum(gated * mimo_o_rank_first[None], axis=1)
        next_state = ScanState(ssm, b_rot, x_t.astype(jnp.float32), new_angle)
        if details:
            def head_rms(v):
                scale=jnp.maximum(jnp.max(jnp.abs(v),-1),1e-30)
                return scale*jnp.sqrt(jnp.mean((v/scale[...,None])**2,-1))
            gate=jax.nn.silu(z_t[:,None].astype(jnp.float32)*mimo_z_rank_first[None])
            metrics=jnp.stack((jnp.mean(head_rms(before)),jnp.max(head_rms(before)),
                jnp.mean(head_rms(y_rank)),jnp.max(head_rms(y_rank)),
                jnp.mean(head_rms(gate)),jnp.max(head_rms(gate))))
            return next_state,(y.astype(x_t.dtype),metrics)
        return next_state, y.astype(x_t.dtype)

    _, output = jax.lax.scan(step, initial, time_major)
    if details:
        output,metrics=output
        names=('readout_pre_rms_mean','readout_pre_rms_max','readout_post_rms_mean',
            'readout_post_rms_max','gate_rms_mean','gate_rms_max')
        summary={name:(jnp.max(metrics[:,i]) if name.endswith('_max') else jnp.mean(metrics[:,i])) for i,name in enumerate(names)}
        return jnp.swapaxes(output,0,1),summary
    return jnp.swapaxes(output, 0, 1)


def pregate_dual(
    x: jax.Array,
    z: jax.Array,
    b: jax.Array,
    c: jax.Array,
    dt: jax.Array,
    decay: jax.Array,
    trap: jax.Array,
    angle_step: jax.Array,
    mimo_x: jax.Array,
    mimo_z: jax.Array,
    mimo_o: jax.Array,
    skip: jax.Array,
    rotary_pairs: int,
    weight: jax.Array,
    strength: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Quadratic SSD-dual execution exactly equivalent to the scan.

    This is a training and analysis path, not an inference kernel.  Unlike a
    generic linear-attention bridge it retains every canonical Mamba-3 MIMO
    degree of freedom, including complex rotation, heavy-tail decay,
    trapezoidal injection, rank-specific value/gate/readout factors and skip.
    Switching these parameters back to :func:`mamba3_reference_scan` therefore
    introduces no parameter conversion or dimensional shock.
    """
    x32 = x.astype(jnp.float32)
    z32 = z.astype(jnp.float32)
    b32 = b.astype(jnp.float32)
    c32 = c.astype(jnp.float32)
    dt32 = dt.astype(jnp.float32)
    alpha = jnp.exp(decay.astype(jnp.float32) * dt32)
    trap_gate = jax.nn.sigmoid(trap.astype(jnp.float32))
    gamma = trap_gate * dt32
    beta = (1.0 - trap_gate) * dt32 * alpha

    increments = jnp.pi * angle_step[:, :, None, :].astype(jnp.float32)
    angle = jnp.cumsum(increments * dt32[..., None], axis=1)
    b_rot = _rotate_mimo_sequence_halves(b32, angle, rotary_pairs)
    c_rot = _rotate_mimo_sequence_halves(c32, angle, rotary_pairs)
    mask = _trapezoidal_dual_mask(alpha, gamma, beta)

    # Sum the R rank-specific input injections exactly as the recurrent scan
    # does before storing them in its shared [H, P, N] state.
    value_by_rank = x32[:, :, None] * jnp.swapaxes(
        mimo_x.astype(jnp.float32), 0, 1
    )[None, None]
    injections = jnp.einsum("btrhp,btrhn->bthpn", value_by_rank, b_rot)
    states = jnp.einsum("bhts,bshpn->bthpn", mask, injections)
    y_rank = jnp.einsum("bthpn,btrhn->btrhp", states, c_rot)
    y_rank += (
        skip[None, None, None, :, None].astype(jnp.float32) * value_by_rank
    )
    y_rank = normalize_readout(y_rank, weight[None, None, None], strength, eps)
    gate_by_rank = z32[:, :, None] * jnp.swapaxes(
        mimo_z.astype(jnp.float32), 0, 1
    )[None, None]
    output_weight = jnp.swapaxes(mimo_o.astype(jnp.float32), 0, 1)[None, None]
    output = jnp.sum(y_rank * jax.nn.silu(gate_by_rank) * output_weight, axis=2)
    return output.astype(x.dtype)



def normalized_mixer(inputs,p,cfg,weight,strength,*,execution='scan',details=False):
    values,_=project_mamba(inputs,p,cfg)
    c=cfg.mamba
    execute=pregate_scan if execution=='scan' else pregate_dual
    kw={'details':True} if details else {}
    if details and execution!='scan':raise ValueError('profiles require scan')
    y=execute(*values,p['mimo_x'],p['mimo_z'],p['mimo_o'],p['D'],
        int(c.d_state*c.rope_fraction)//2,weight,strength,**kw)
    if details:y,metrics=y
    features=y.reshape(inputs.shape[0],inputs.shape[1],-1)
    out=nn.Dense(cfg.hidden_size,use_bias=False,dtype=dtype_from_name(cfg.compute_dtype),
        param_dtype=dtype_from_name(cfg.param_dtype)).apply({'params':p['out_proj']},features)
    return (out,metrics) if details else out


def initialize(patient,spec):
    result=base_initialize(patient,spec)
    if spec['pregate']:
        c=patient.cfg.mamba;heads=int(patient.cfg.hidden_size*c.expand)//c.head_dim
        for i in patient.cfg.mamba_layer_indices:
            result[str(i)]['pregate_weight']=jnp.ones((heads,c.head_dim),jnp.float32)
    return result


def optimizer(q,spec):
    labels=jax.tree.map_with_path(lambda path,x:'norm' if any(getattr(k,'key',None)=='pregate_weight' for k in path) else 'mamba',q)
    def transformation(peak):
        return optax.chain(optax.scale_by_adam(b1=.9,b2=.999,eps=1e-8),
            optax.scale_by_schedule(lambda n:-peak*jnp.minimum((n+1)/256.,1.)))
    return optax.chain(stable_clip_by_global_norm(1.),optax.multi_transform(
        dict(mamba=transformation(3e-4),norm=transformation(spec['norm_lr'])),labels))


def block_forward(cfg,index,p,q,h,columns,spec,strength,*,details=False):
    p=corrected_layer(p,q,columns)
    n=norm(h,p['input_layernorm'],cfg)
    internal={}
    if details and index in cfg.mamba_layer_indices:
        c=cfg.mamba;heads=int(cfg.hidden_size*c.expand)//c.head_dim
        weight=q.get('pregate_weight',jnp.ones((heads,c.head_dim),jnp.float32))
        mixed,internal=normalized_mixer(n,p['mamba'],cfg,weight,strength if spec['pregate'] else 0.,details=True)
    elif index in cfg.mamba_layer_indices and spec['pregate']:
        # Exactly the established source executable at phase0 (also used for
        # background/source controls). Modified path only after intervention.
        mixed=jax.lax.cond(strength==0.,lambda _:mixer(n,p,cfg,index),
            lambda _:normalized_mixer(n,p['mamba'],cfg,q['pregate_weight'],strength),None)
    else:mixed=mixer(n,p,cfg,index)
    middle=h+mixed
    contribution=mlp(norm(middle,p['post_attention_layernorm'],cfg),p,cfg)
    out=middle+contribution
    penalty=jax.lax.stop_gradient(jnp.maximum(log_rms(mixed)-log_rms(h),0.)**2)
    if not details:return out,penalty
    metrics=dict(input_rms=jnp.exp(log_rms(h)),raw_mixer_rms=jnp.exp(log_rms(mixed)),
        scaled_mixer_rms=jnp.exp(log_rms(mixed)),output_rms=jnp.exp(log_rms(out)),
        mlp_rms=jnp.exp(log_rms(contribution)),output_max=jnp.max(jnp.abs(out.astype(jnp.float32))),
        mlp_addition_absorbed_fraction=jnp.sum((out==middle)&(contribution!=0))/jnp.maximum(jnp.sum(contribution!=0),1),
        mixer_addition_absorbed_fraction=jnp.sum((middle==h)&(mixed!=0))/jnp.maximum(jnp.sum(mixed!=0),1))
    metrics.update(internal)
    if 'pregate_weight' in q:
        metrics.update(norm_weight_min=jnp.min(q['pregate_weight']),norm_weight_max=jnp.max(q['pregate_weight']),
            norm_weight_rms=jnp.exp(log_rms(q['pregate_weight'])))
    return out,penalty,metrics


class Engine(ScaleEngine):
    def __init__(self,patient,spec):
        super().__init__(patient,spec)
        self.set_step(0)
    def set_step(self,step):
        self.phase=self.p.placed(np.asarray(min(max(int(step),0)/RAMP_STEPS,1.),np.float32))
    def kernels(self,layer,q):
        attention=layer in self.cfg.attention_layer_indices
        key=(attention,'lora' in q,'pregate_weight' in q)
        if key not in self.forward_kernels:
            index=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
            def fn(p,q,h,phase):return block_forward(self.cfg,index,p,q,h,self.p.columns,self.spec,phase)
            self.forward_kernels[key]=jax.jit(fn)
            def backward(p,q,h,dy,phase):
                _,pullback=jax.vjp(lambda q,h:fn(p,q,h,phase),q,h)
                return pullback((dy,jnp.array(0.,jnp.float32)))
            self.backward_kernels[key]=jax.jit(backward)
        return (lambda p,q,h:self.forward_kernels[key](p,q,h,self.phase),
                lambda p,q,h,dy:self.backward_kernels[key](p,q,h,dy,self.phase))
    def inspect(self,q,tokens):
        h=self.p.embed(tokens,self.cfg.compute_dtype);rows=[]
        for layer in range(self.cfg.num_layers):
            self.p.check();current=q[str(layer)]
            attention=layer in self.cfg.attention_layer_indices
            key=(attention,'lora' in current,'pregate_weight' in current)
            if key not in self.profile_kernels:
                index=self.cfg.attention_layer_indices[0] if attention else self.cfg.mamba_layer_indices[0]
                self.profile_kernels[key]=jax.jit(lambda p,q,h,phase,index=index:block_forward(
                    self.cfg,index,p,q,h,self.p.columns,self.spec,phase,details=True))
            parameters=self.p.placed(self.parameters(layer));forward,_=self.kernels(layer,current)
            ordinary,_=forward(parameters,current,h)
            diagnostic,_,metrics=self.profile_kernels[key](parameters,current,h,self.phase)
            error=float(jnp.max(jnp.abs(ordinary.astype(jnp.float32)-diagnostic.astype(jnp.float32)))/jnp.maximum(jnp.max(jnp.abs(ordinary)),1e-30))
            metrics=plain(jax.block_until_ready(metrics))
            rows.append(dict(layer=layer,relative_output_discrepancy=error if math.isfinite(error) else str(error),
                accepted=math.isfinite(error) and error<=.02,**metrics))
            h=ordinary
        return dict(layers=rows,ramp_strength=float(self.phase),relative_output_tolerance=.02,
            interpretation='One A window; isolated pre-gate normalization; no artificial residual bound or recurrence-trace claim')
