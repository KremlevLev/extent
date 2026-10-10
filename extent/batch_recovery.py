"""Matched token-clock CE recovery with sequential gradient accumulation."""
import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from extent.decoder_recovery import forward_parameters
from extent.full_model_distillation import causal_cross_entropy
from extent.layer_clip_recovery import clip_gradients, AGC_RATIO, MIN_WEIGHT_NORM
from extent.stable_gradient_clip import scaled_norm_parts
from extent.recovery_subspace import coordinates_finite

TOTAL_STEPS = 65536  # consumed context256 windows, NOT optimizer updates
WARMUP = 1024       # same token-clock schedule for batch1 and batch4


def schedule(windows):
    windows = jnp.asarray(windows,jnp.float32)
    fraction = jnp.clip((windows-WARMUP)/(TOTAL_STEPS-WARMUP),0,1)
    return jnp.where(windows<WARMUP,1e-5*windows/WARMUP,
                     1e-6+9e-6*(1+jnp.cos(jnp.pi*fraction))/2)


def optimizer(parameters, adaptive=False, accumulation=1):
    if accumulation not in (1,4):
        raise ValueError("registered accumulation is1 or4")
    mask=traverse_util.unflatten_dict({path:x.ndim>=2 and not path[-1].endswith("bias")
        for path,x in traverse_util.flatten_dict(parameters).items()})
    def clip(updates,state,params):
        return clip_gradients(updates,params,adaptive)[0],state
    return optax.chain(optax.GradientTransformation(lambda _:optax.EmptyState(),clip),
        optax.scale_by_adam(b1=.9,b2=.95,eps=1e-8),
        optax.add_decayed_weights(.1,mask=mask),
        optax.scale_by_schedule(lambda n:-schedule(n*accumulation)))


def averaged_gradient(loss,parameters,tokens):
    """Mean UNCLIPPED gradients on the SAME weights, then one Adam update.

    Microbatches have shape1x256. Divide before adding to reduce overflow risk;
    finite guards cover every microbatch, not just a possibly cancelling mean.
    """
    count=tokens.shape[0]
    def micro(carry,t):
        gradient_sum,finite=carry
        (value,forward_finite),gradient=jax.value_and_grad(loss,has_aux=True)(parameters,t)
        _,_,log_norm=scaled_norm_parts(gradient)
        finite &= forward_finite & jnp.isfinite(value) & coordinates_finite(gradient)
        gradient_sum=jax.tree.map(lambda a,g:a+g/count,gradient_sum,gradient)
        return (gradient_sum,finite),(value,log_norm)
    (mean_gradient,finite),(losses,logs)=jax.lax.scan(micro,
        (jax.tree.map(jnp.zeros_like,parameters),jnp.array(True)),tokens)
    _,_,mean_log=scaled_norm_parts(mean_gradient)
    log_mean_micro_norm=jax.scipy.special.logsumexp(logs*jnp.log(10.))/jnp.log(10.)-jnp.log10(float(count))
    diagnostics=dict(micro_loss_std=jnp.std(losses),micro_max_log10_grad_norm=jnp.max(logs),
        log10_mean_to_mean_micro_norm=jnp.where(jnp.isneginf(mean_log) & jnp.isneginf(log_mean_micro_norm),0.,mean_log-log_mean_micro_norm))
    return jnp.mean(losses),mean_gradient,finite,diagnostics


def make_step(model,tx,arm,columns,head_dim,layouts=None,*,adaptive=False,accumulation=1):
    def step(parameters,state,fixed,tokens):
        if tokens.shape[0]!=accumulation:
            raise ValueError("microbatch count differs from optimizer contract")
        def loss(p,t):
            logits=model.apply({"params":forward_parameters(arm,p,fixed,columns,head_dim)},t)
            return causal_cross_entropy(logits,t),jnp.all(jnp.isfinite(logits))
        value,gradients,micro_finite,noise=averaged_gradient(loss,parameters,tokens)
        updates,proposed_state=tx.update(gradients,state,parameters)
        proposed=optax.apply_updates(parameters,updates)
        _,diagnostics=clip_gradients(gradients,parameters,adaptive)
        for name in diagnostics:
            _,_,update_log=scaled_norm_parts(updates[name])
            diagnostics[name]["log10_update_norm"]=update_log
            diagnostics[name]["log10_relative_update"]=update_log-jnp.maximum(
                diagnostics[name]["log10_weight_norm"],jnp.log10(MIN_WEIGHT_NORM))
        maximum,scaled,logarithm=scaled_norm_parts(gradients)
        finite=(micro_finite & coordinates_finite(gradients) & coordinates_finite(proposed)
                & coordinates_finite(proposed_state))
        health=dict(loss=value,finite=finite,grad_max_abs=maximum,scaled_norm=scaled,
            log10_grad_norm=logarithm,naive_norm_finite=jnp.isfinite(optax.global_norm(gradients)),
            layers=diagnostics,update_max_abs=jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in jax.tree.leaves(updates)])),
            **noise)
        return proposed,proposed_state,health
    if layouts is None:return jax.jit(step)
    p,s,f,b,replicated=layouts
    return jax.jit(step,in_shardings=(p,s,f,b),out_shardings=(p,s,replicated))
