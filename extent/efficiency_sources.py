"""Pinned original ONPOLICY or native random Mamba, no recovered EXP098."""
import gc,json,time,hashlib
from dataclasses import replace
from pathlib import Path
import jax
import numpy as np
from flax import traverse_util
from extent import HybridForCausalLM
from extent.campaign_checkpoint import CampaignCheckpointStore
from extent.initialization import abstract_parameter_tree,initialize_sharded_parameters
from extent.recovery_subspace import initialize_corrections
from extent.sharding import named_sharding_tree,batch_sharding
from scripts import m3q_subspace_engine as engine
from scripts.m3q_paper_decoder_recovery_campaign import evaluate

ROOT=Path(__file__).resolve().parents[1]

class ReadOnlySourceStore(CampaignCheckpointStore):
    def sync(self,slot):return False

def tree_digest(tree):
    h=hashlib.sha256()
    for path,array in sorted(traverse_util.flatten_dict(tree).items()):
        value=np.asarray(array)
        h.update(json.dumps([path,value.shape,str(value.dtype)]).encode())
        h.update(value.tobytes())
    return h.hexdigest()

def load_sources(config,mesh,args,hub,tokens,result,persist,deadline,*,experiment):
    pin=json.loads((ROOT/'results/EXP-111-113-source-pin.json').read_text())
    seq=engine.configured_contract(config); order=tuple(seq['replacement_order'])
    source=engine.teacher_config_from_spec(engine.QWEN3_1_7B_BASE,param_dtype='bfloat16',compute_dtype='bfloat16',remat_policy='full')
    teacher=engine.Qwen3ForCausalLM(source)
    batch=batch_sharding(mesh); init_tokens=jax.device_put(np.zeros((1,1),np.int32),batch)
    initial=initialize_sharded_parameters(teacher,jax.random.key(940),init_tokens,mesh)
    tp,_=engine.stream_teacher_qwen_weights(initial.params,engine._ensure_checkpoint(Path(args.qwen_cache_dir)),source)
    del initial
    if not result.get('original_qwen'):
        result['original_qwen']={name:evaluate(teacher,tp,tokens[name],batch) for name in ('locked_test','pg19_test','reproduction')};persist()
    model=HybridForCausalLM(config); abstract=abstract_parameter_tree(model); layout=named_sharding_tree(abstract,mesh)
    pinned=replace(hub,repo_id=pin['repo_id'],repo_type='dataset',revision=pin['revision'])
    stores=(ReadOnlySourceStore(Path(args.state_dir)/'source069',f"{engine.EXP069_PREFIX}/checkpoints",pinned),
        ReadOnlySourceStore(Path(args.state_dir)/'source072',f"{engine.EXP072['HF_PREFIX']}/checkpoints",pinned))
    sources={}
    for seed in (123,456):
        if time.monotonic()>=deadline:raise TimeoutError('source loading deadline')
        if experiment==113:
            random=initialize_sharded_parameters(model,jax.random.key(seed),init_tokens,mesh)
            endpoints={j:random.params[f'layers_{j}']['mamba'] for j in order}
        else:
            endpoints={}
            for depth,layer in enumerate(order,1):
                if time.monotonic()>=deadline:raise TimeoutError('ONPOLICY binary preflight deadline')
                prep=stores[0].metadata(f'prep/seed-{seed}/layer-{layer}',dict(seq['source_contract'],seed=seed,layer=layer,kind='prepared_mamba'))
                if prep is None:raise FileNotFoundError('missing pinned EXP069 metadata')
                contract=engine.sequential.endpoint_contract(seq,seed,'ONPOLICY',layer,depth,prep['checkpoint_sha256'])
                saved=stores[1].restore(f'seed-{seed}/ONPOLICY/layer-{layer}',contract)
                if saved is None:raise FileNotFoundError('missing pinned EXP072 binary')
                payload,metadata=saved
                if metadata['checkpoint_sha256']!=pin['seeds'][str(seed)][str(layer)]:raise ValueError('pinned ONPOLICY SHA mismatch')
                endpoints[layer]=payload['params']
        base=engine.compose_parameters(tp,endpoints,order,abstract,layout)
        c=initialize_corrections(base,order,'INOUT-LORA',seed=seed,rank=32,head_dim=config.mamba.head_dim)
        host=jax.device_get(base)
        result.setdefault('source_models',{})[str(seed)]=dict(sha256=tree_digest(host),
            mamba_sha256=tree_digest({f'layers_{j}':host[f'layers_{j}']['mamba'] for j in order}),
            kind='native_random_mamba_only' if experiment==113 else 'original_072_ONPOLICY')
        sources[seed]=(host,jax.device_get(c));persist()
        del base,c,endpoints
        if experiment==113:del random
        gc.collect()
    teacher_host=jax.device_get(tp); del tp; gc.collect()
    return model,order,sources,(teacher,teacher_host,source)
