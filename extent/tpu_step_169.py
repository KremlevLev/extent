"""Resident compiled TPU proposals, audited against the layer-streamed chain rule."""
import os,time,gc
import numpy as np
from extent.method_sweep_163 import MethodReference


class TPUReference(MethodReference):
    def teacher_forward(self,base,tp,tokens):
        if self.beta:raise ValueError('EXP169 reference registers hidden_beta0')
        h=self.embed(base['embed_tokens']['embedding'],tokens)
        for i in range(self.cfg.num_layers):
            self.check();h=self.teacher(tp[f'layers_{i}'],h)
        return h,{}


class Step:
    def __init__(self,cfg,tc,columns,spec,tx):
        import jax
        from extent.method_execution_163 import compiled_step
        self.fn=jax.jit(compiled_step(cfg,tc,columns,spec,tx));self.return_gradients=False
    def __call__(self,*args):
        import jax
        result=jax.block_until_ready(self.fn(*args))
        return result if self.return_gradients else result[:3]


def plan_kernel(cfg,tc,columns,spec,tx,q,state,base,teacher,tokens,cursor,check,record):
    import jax
    from extent.device_execution import compare_proposals
    engine=Step(cfg,tc,columns,spec,tx) if os.environ['EXTENT169_ENGINE']=='compiled' else TPUReference(cfg,tc,columns,spec,tx)
    engine.check=check;samples=[];started=time.perf_counter();warm_seconds=0.
    for index in range(5):
        check();begun=time.perf_counter();out=engine(q,state,base,teacher,tokens,cursor)
        if not bool(out[2]['finite']):raise FloatingPointError('nonfinite TPU probe')
        if index>=2:samples.append(time.perf_counter()-begun)
        else:warm_seconds+=time.perf_counter()-begun
        del out
    batch=tokens.shape[0];result=dict(finite=True,accepted_updates=0,selected_batch=batch,
        seconds_per_update=float(np.median(samples)),input_tokens_per_second=tokens.size/float(np.median(samples)),
        next_token_targets_per_second=batch*(tokens.shape[1]-1)/float(np.median(samples)),
        probe_input_presentations=5*tokens.size,elapsed_seconds=time.perf_counter()-started,
        compile_and_warmup_seconds=warm_seconds,measurement_seconds=sum(samples),
        execution=os.environ['EXTENT169_ENGINE'])
    if os.environ.get('EXTENT169_PROBE')!='1':
        engine.return_gradients=True;actual=engine(q,state,base,teacher,tokens,cursor)
        actual=jax.tree.map(lambda v:np.asarray(jax.device_get(v)),actual);gc.collect()
        reference=TPUReference(cfg,tc,columns,spec,tx,return_gradients=True);reference.check=check
        expected=reference(q,state,base,teacher,tokens,cursor)
        result['gradient_audit']=compare_proposals(expected,actual)
        del expected,actual,reference;gc.collect();engine.return_gradients=False
        if not result['gradient_audit']['passed']:raise ValueError('TPU compiled/reference gradient audit failed')
    if isinstance(engine,Step):
        try:
            analysis=engine.fn.lower(q,state,base,teacher,tokens,cursor).compile().memory_analysis()
            result['compiled_memory']=None if analysis is None else {k:getattr(analysis,k,None) for k in
                ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
        except (NotImplementedError,AttributeError) as exc:result['memory_analysis_unavailable']=type(exc).__name__
    result['devices']=[]
    for d in jax.devices():
        if d.platform!='tpu':continue
        try:memory=d.memory_stats()
        except NotImplementedError:memory=None
        result['devices'].append(dict(id=d.id,memory=memory))
    record(result);return engine,result


def gradient_stats(a,b):
    import jax
    result={}
    for key in a:
        aa=[np.asarray(jax.device_get(v),dtype=np.float64) for v in jax.tree.leaves(a[key])]
        bb=[np.asarray(jax.device_get(v),dtype=np.float64) for v in jax.tree.leaves(b[key])]
        na=sum(float(np.sum(v*v)) for v in aa);nb=sum(float(np.sum(v*v)) for v in bb)
        dot=sum(float(np.sum(x*y)) for x,y in zip(aa,bb))
        result[str(key)]=dict(ce_norm=na**.5,kl_norm=nb**.5,cosine=dot/max((na*nb)**.5,1e-300))
    return result


def diagnostics(cfg,tc,columns,spec,tx,payload,base,teacher,tokens,step,check,validation_tokens=None):
    """Unaccepted same-source CE/KL pair; bounded to one sequence, not a training trial."""
    import jax
    import jax.numpy as jnp
    q=payload['parameters'];state=payload['optimizer'];gradients=[];rows=[]
    for alpha in (0.,1.):
        check();engine=TPUReference(cfg,tc,columns,dict(spec,alpha=alpha),tx,return_gradients=True);engine.check=check
        out=engine(q,state,base,teacher,tokens[:1],jnp.asarray(step,jnp.float32))
        if not bool(out[2]['finite']):raise FloatingPointError('nonfinite CE/KL diagnostic')
        gradients.append(jax.tree.map(lambda v:np.asarray(jax.device_get(v)),out[3]))
        updates={}
        for layer in q:
            before=jax.tree.leaves(q[layer]);after=jax.tree.leaves(out[0][layer])
            norm=change=rounded=count=0
            for a,b in zip(before,after):
                ah=np.asarray(jax.device_get(a),dtype=np.float64);bh=np.asarray(jax.device_get(b),dtype=np.float64)
                norm+=float(np.sum(ah*ah));change+=float(np.sum((bh-ah)**2));count+=a.size
                rounded+=int(np.sum(np.asarray(jax.device_get(a.astype(jnp.bfloat16)))==np.asarray(jax.device_get(b.astype(jnp.bfloat16)))))
            updates[layer]=dict(relative_update_norm=(change/max(norm,1e-300))**.5,bf16_coordinate_unchanged_fraction=rounded/max(count,1))
        rows.append(dict(alpha=alpha,ce=float(out[2]['ce']),kl=float(out[2]['kl']),loss=float(out[2]['loss']),updates_by_layer=updates,
            log10_gradient_norm=float(out[2]['log10_grad_norm']),global_clip_factor=10.**min(0.,-float(out[2]['log10_grad_norm'])),
            bf16_scope='coordinate cast only; not a full-forward equivalence test'))
        del out,engine;gc.collect()
    # head_tile alpha weights KL; alpha0 is CE, alpha1 is KL.
    layers=gradient_stats(gradients[0],gradients[1])
    grouped=[]
    for tree in gradients:
        groups={'mixer':[],'mlp':[],'norm':[]}
        for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
            names=[getattr(k,'key',None) for k in path]
            label='norm' if any(k in names for k in ('pregate_weight','input_layernorm','post_attention_layernorm')) else 'mlp' if any(k in names for k in ('dense_mlp','mlp','mlp_lora')) else 'mixer'
            groups[label].append(value)
        grouped.append(groups)
    groups=gradient_stats(*grouped);del gradients,grouped
    profile=layer_profile(cfg,tc,columns,spec,q,base,teacher,tokens[:1],check)
    return dict(scope='one shared fresh source window; unaccepted proposals, no causal long-run superiority claim',
        objective_pairs=rows,ce_kl_by_layer=layers,ce_kl_by_group=groups,accepted_updates=0,
        layer_profile=profile,
        validation_layer_profile=layer_profile(cfg,tc,columns,spec,q,base,teacher,validation_tokens[:1],check) if validation_tokens is not None else None,
        hypotheses=['CE/KL gradient conflict','weak per-layer learning signal'],
        not_measured=['long-run alternative loss benefit','full activation causal attribution','FP32-vs-BF16 trajectory'])


def shard_bytes(tree):
    """Physical addressable bytes by chip; replication counted on each chip."""
    import jax
    result={}
    for value in jax.tree.leaves(tree):
        for shard in value.addressable_shards:
            key=str(shard.device.id);result[key]=result.get(key,0)+shard.data.size*shard.data.dtype.itemsize
    return result


def layer_profile(cfg,tc,columns,spec,q,base,teacher,tokens,check):
    """Teacher-prefix and hybrid-prefix block discrepancies on identical tokens."""
    import jax
    import jax.numpy as jnp
    from extent.method_sweep_163 import block_forward
    from extent.plateau_forensics import mixer,mlp,norm
    # Use the same primitives as StreamedStep.teacher, with original Qwen config.
    def teacher_block(p,h):
        h=h+mixer(norm(h,p['input_layernorm'],cfg),p,cfg,0,tc)
        return h+mlp(norm(h,p['post_attention_layernorm'],cfg),p,cfg)
    teach=jax.jit(teacher_block);blocks={}
    h=base['embed_tokens']['embedding'][tokens].astype(getattr(jnp,cfg.compute_dtype));th=h;rows=[]
    def stats(x):
        v=np.asarray(jax.device_get(x),dtype=np.float64)
        return dict(rms=float(np.sqrt(np.mean(v*v))),max_abs=float(np.max(np.abs(v))))
    def distance(a,b):
        v=np.asarray(jax.device_get(a),dtype=np.float64);w=np.asarray(jax.device_get(b),dtype=np.float64)
        return float(np.sqrt(np.sum((v-w)**2)/max(np.sum(w*w),1e-300)))
    for i in range(cfg.num_layers):
        check();key=i in cfg.attention_layer_indices
        if key not in blocks:
            blocks[key]=jax.jit(lambda p,r,x,index=i:block_forward(cfg,index,p,r,x,columns,spec,jnp.asarray(1.,jnp.float32))[0])
        f=blocks[key];p=base[f'layers_{i}'];r=q[str(i)]
        teacher_on_student=teach(teacher[f'layers_{i}'],h);next_th=teach(teacher[f'layers_{i}'],th)
        student_on_teacher=f(p,r,th);next_h=f(p,r,h)
        rows.append(dict(layer=i,input=stats(h),output=stats(next_h),residual=stats(next_h.astype(jnp.float32)-h.astype(jnp.float32)),
            teacher_prefix_relative_error=distance(student_on_teacher,next_th),hybrid_prefix_relative_error=distance(next_h,teacher_on_student),
            accumulated_relative_error=distance(next_h,next_th)))
        h,th=next_h,next_th
    return dict(layers=rows,scope='one shared fresh training window; observational discrepancies, not causal attribution')
