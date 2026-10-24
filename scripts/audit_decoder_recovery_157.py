"""Shape-only full decoder resource estimate; no model tensors allocated."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from extent.config import load_config
from extent import HybridForCausalLM
from extent.sharding import create_v5e_mesh,named_sharding_tree
from scripts.m3q_peer_tpu_campaign import coordinate_layout
from scripts.m3q_paper_decoder_recovery_campaign import memory_estimate
from extent.decoder_recovery_157 import ROOT

def main():
    cfg,_=load_config(ROOT/'config/hybrid_1_7b_gqa_v5e8.yaml')
    shape=jax.eval_shape(lambda t:HybridForCausalLM(cfg).init(jax.random.key(157),t)['params'],jax.ShapeDtypeStruct((1,8),jnp.int32))
    q={str(i):{'dense_layer':jax.tree.map(lambda v:jax.ShapeDtypeStruct(v.shape,jnp.float32),shape[f'layers_{i}'])} for i in range(cfg.num_layers)}
    for i in cfg.mamba_layer_indices:q[str(i)]['pregate_weight']=jax.ShapeDtypeStruct((int(cfg.hidden_size*cfg.mamba.expand)//cfg.mamba.head_dim,cfg.mamba.head_dim),jnp.float32)
    mesh=create_v5e_mesh();layout=coordinate_layout(q,mesh)
    report=memory_estimate(q,shape,layout,named_sharding_tree(shape,mesh))
    report.update(devices=len(jax.devices()),scope='shape-only; no actual full model compile or accelerator allocation',
        checkpoint_payload_bytes=16*report['trainable_count'],local_previous_plus_next_checkpoint_bytes=32*report['trainable_count'])
    report['other_arm_memory_and_evaluation_workspace_not_included']=True
    path=ROOT/'results/EXP-157-shape-audit.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report
if __name__=='__main__':main()
