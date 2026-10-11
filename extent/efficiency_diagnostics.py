"""Read-only CE, component, BF16 and rollback diagnostics; no optimizer mutation."""
from dataclasses import replace
import gc,time
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from extent.full_model_distillation import causal_cross_entropy
from extent.stable_gradient_clip import scaled_norm_parts
from extent.sharding import named_sharding_tree
from extent.model import HybridDecoderLayer,HybridForCausalLM
from extent.qwen3_teacher import Qwen3DecoderLayer
from scripts.m3q_paper_decoder_recovery_campaign import evaluate

def stats(x):
    maximum,_,lognorm=scaled_norm_parts({'x':x})
    return dict(log10_norm=lognorm,max_abs=maximum,rms=jnp.where(maximum==0,0.,jnp.power(10.,lognorm)/jnp.sqrt(float(x.size))),
        zero_fraction=jnp.mean(x==0),finite=jnp.all(jnp.isfinite(x)))

def relative(a,b):
    # Float64 is not available on TPU; normalized cosine avoids huge products.
    a,b=a.astype(jnp.float32),b.astype(jnp.float32)
    ma,_,la=scaled_norm_parts({'x':a}); mb,_,lb=scaled_norm_parts({'x':b})
    u=a/jnp.maximum(ma,1e-30);v=b/jnp.maximum(mb,1e-30)
    me,_,le=scaled_norm_parts({'x':a-b})
    return dict(relative_l2=jnp.where(me==0,0.,jnp.power(10.,le-lb)),cosine=jnp.sum(u*v)/jnp.maximum(jnp.sqrt(jnp.sum(u*u)*jnp.sum(v*v)),1e-30))

def component_arrays(parameters,config):
    flat=traverse_util.flatten_dict(parameters); result={}
    inner=int(config.hidden_size*config.mamba.expand);heads=inner//config.mamba.head_dim
    sizes=(inner,inner,config.mamba.mimo_rank*config.mamba.groups*config.mamba.d_state,
        config.mamba.mimo_rank*config.mamba.groups*config.mamba.d_state,heads,heads,heads,int(config.mamba.d_state*config.mamba.rope_fraction)//2)
    for path,x in flat.items():
        if path[0] in ('embed_tokens','lm_head'):continue
        name='/'.join(path)
        if path[-3:]==('mamba','in_proj','kernel'):
            offset=0
            if x.shape[-1]!=sum(sizes):raise ValueError('packed component schema mismatch')
            for part,size in zip(('gate','value','B','C','dt','decay','trapezoid','angle'),sizes):
                if size:result[name+'/'+part]=x[...,offset:offset+size]
                offset+=size
        else:result[name]=x
    return result

def json_safe(tree):
    def scalar(x):
        value=np.asarray(x).item()
        return value if not isinstance(value,float) or np.isfinite(value) else str(value)
    return jax.tree.map(scalar,jax.device_get(tree))

def checked_call(function,*args):
    executable=function.lower(*args).compile()
    memory=executable.memory_analysis();device=jax.devices()[0].memory_stats() or {}
    limit=device.get('bytes_limit',device.get('bytes_reservable_limit'))
    if memory is not None and limit is not None:
        extra=memory.output_size_in_bytes+memory.temp_size_in_bytes-memory.alias_size_in_bytes
        if memory.argument_size_in_bytes+extra>limit or extra>limit-device.get('bytes_in_use',0):
            raise MemoryError('diagnostic compiler memory exceeds HBM')
    return executable(*args)

def update_metrics(before,after,effective_before,effective_after,config):
    """Actual accepted-step deltas; this function never calls the optimizer."""
    b=traverse_util.flatten_dict(before); a=traverse_util.flatten_dict(after)
    coordinates={'/'.join(k):dict(update=stats(a[k]-b[k]),log10_relative_update=
        scaled_norm_parts({'x':a[k]-b[k]})[2]-jnp.maximum(scaled_norm_parts({'x':b[k]})[2],-30.)) for k in b}
    eb=component_arrays(effective_before,config);ea=component_arrays(effective_after,config)
    kernels={k:dict(update=stats(ea[k].astype(jnp.float32)-eb[k].astype(jnp.float32)),
        bf16_changed_fraction=jnp.mean(ea[k]!=eb[k])) for k in eb}
    return dict(trainable=coordinates,effective=kernels)

class Probes:
    def __init__(self,model,teacher,config,source,mesh,batch):
        self.model,self.teacher,self.config,self.source,self.mesh,self.batch=model,teacher,config,source,mesh,batch
        def capture(module,p,t):
            logits,states=module.apply({'params':p},t,return_hidden_states=True)
            return logits,(p['embed_tokens']['embedding'][t],)+states
        self.capture=jax.jit(lambda p,t:capture(model,p,t))
        self.teacher_capture=jax.jit(lambda p,t:capture(teacher,p,t))
        def probe(p,start,t,h,ht,hs,logits):
            def objective(weights):
                # Some MIMO factors use their stored dtype directly. Explicit
                # BF16 casts preserve the exact training forward, not just Dense.
                effective=jax.tree.map(lambda x:x.astype(jnp.bfloat16),weights)
                return causal_cross_entropy(model.apply({'params':effective},t),t)
            # Differentiate FP32 effective weights; forward still casts to BF16.
            fp=jax.tree.map(lambda x:x.astype(jnp.float32),p)
            gradient_loss,grad=jax.value_and_grad(objective)(fp)
            layer={str(j):dict(hidden=stats(h[j+1]),contribution=stats(h[j+1]-h[j]),
                contribution_error=relative(h[j+1]-h[j],ht[j+1]-ht[j]),
                initial_contribution_error=relative(hs[j+1]-hs[j],ht[j+1]-ht[j]),input_drift=relative(h[j],ht[j])) for j in range(config.num_layers)}
            g=component_arrays(grad,config); c=component_arrays(p,config);base=component_arrays(start,config)
            components={k:dict(gradient=stats(g[k]),effective_displacement=stats(c[k].astype(jnp.float32)-base[k].astype(jnp.float32)),
                bf16_changed_fraction=jnp.mean(c[k]!=base[k])) for k in c}
            labels=t[:,1:];scores=jax.nn.log_softmax(logits[:,:-1].astype(jnp.float32),axis=-1)
            losses=-jnp.take_along_axis(scores,labels[...,None],axis=-1)[...,0]
            positions={f'{a}-{b-1}':jnp.mean(losses[:,a:b]) for a,b in ((0,64),(64,128),(128,192),(192,255))}
            return dict(loss=causal_cross_entropy(logits,t),gradient_forward_nll=gradient_loss,
                layers=layer,components=components,positions=positions)
        self.probe=jax.jit(probe)
        self.local={}
        self.fp32=None
    def run(self,current,start,teacher_host,panel,*,rollback=False,fp32=False):
        begun=time.monotonic(); tp=jax.device_put(teacher_host,named_sharding_tree(teacher_host,self.mesh))
        rows=[];inputs=[]
        for window in panel:
            t=jax.device_put(window[None],self.batch)
            logits,h=checked_call(self.capture,current,t);_,hs=checked_call(self.capture,start,t);_,ht=checked_call(self.teacher_capture,tp,t)
            metrics=checked_call(self.probe,current,start,t,h,ht,hs,logits); rows.append(json_safe(metrics))
            # One packed input pair per panel window, no logits persisted.
            inputs.append((h,ht))
        report=dict(windows=rows,canonical_nll=evaluate(self.model,current,panel,self.batch),
            math_note='Captured-hidden/gradient executables may round differently in BF16; canonical_nll uses the training evaluation path. Start/current hidden captures use the same forward function.')
        # Local blocks on identical teacher/hybrid inputs separate input shift.
        local_results=[]
        for h,ht in inputs:
            local_row={}
            for j in range(self.config.num_layers):
                key=int(j in self.config.attention_layer_indices)
                if key not in self.local:
                    student=HybridDecoderLayer(self.config,j); original=Qwen3DecoderLayer(self.source)
                    def call(p,q,x,y,student=student,original=original):
                        positions=jnp.arange(x.shape[1])[None]
                        a=student.apply({'params':p},x,positions,None)-x
                        b=student.apply({'params':p},y,positions,None)-y
                        c=original.apply({'params':q},x,positions,None)-x
                        d=original.apply({'params':q},y,positions,None)-y
                        return dict(teacher_input=relative(a,c),hybrid_input=relative(b,d))
                    self.local[key]=jax.jit(call)
                local_row[str(j)]=json_safe(self.local[key](current[f'layers_{j}'],tp[f'layers_{j}'],ht[j],h[j]))
            local_results.append(local_row)
        report['matched_input']=local_results
        if rollback:
            ranking=sorted(range(self.config.num_layers),key=lambda j:-np.mean([r['layers'][str(j)]['contribution_error']['relative_l2']-r['layers'][str(j)]['initial_contribution_error']['relative_l2'] for r in rows]))[:2]
            report['rollback']={}
            baseline=evaluate(self.model,current,panel,self.batch)['nll']
            identity=dict(current);identity['layers_0']=jax.tree.map(lambda x:x,current['layers_0'])
            report['rollback_identity_nll_error']=abs(evaluate(self.model,identity,panel,self.batch)['nll']-baseline)
            if report['rollback_identity_nll_error']>1e-6:raise ValueError('rollback identity control failed')
            for j in sorted(set(ranking+[min(j+1,self.config.num_layers-1) for j in ranking])):
                altered=dict(current);altered[f'layers_{j}']=start[f'layers_{j}']
                report['rollback'][str(j)]=dict(selected=j in ranking,nll_change=evaluate(self.model,altered,panel,self.batch)['nll']-baseline)
        if fp32:
            if self.fp32 is None:self.fp32=HybridForCausalLM(replace(self.config,compute_dtype='float32',param_dtype='float32'))
            # Equal values, including tied readout: avoid its param_dtype BF16 cast.
            values=None
            try:
                with jax.default_matmul_precision('highest'):
                    values=jax.tree.map(lambda x:x.astype(jnp.float32),current)
                    call=jax.jit(lambda p,t:causal_cross_entropy(self.fp32.apply({'params':p},t),t))
                    losses=[float(checked_call(call,values,jax.device_put(w[None],self.batch))) for w in panel[:2]]
                    if not np.all(np.isfinite(losses)):raise FloatingPointError('nonfinite FP32 diagnostic')
                    report['fp32_forward']=dict(nll=float(np.mean(losses)),windows=losses)
            except Exception as e:
                if isinstance(e,MemoryError) or 'RESOURCE_EXHAUSTED' in str(e) or 'out of memory' in str(e).lower():
                    report['fp32_forward']=dict(unavailable='memory feasibility',error_type=type(e).__name__)
                else:raise
            finally:
                del values;gc.collect()
        else:report['fp32_forward']='not requested; no FP32 stability conclusion'
        report['seconds']=time.monotonic()-begun
        del tp,inputs;gc.collect()
        return report
