import jax
import jax.numpy as jnp

from singularity import HybridForCausalLM, tiny_config
from singularity.initialization import abstract_parameter_tree, optimizer_state_layout
from singularity.optimizer import create_lion
from singularity.preflight import build_preflight_report
from singularity.sharding import create_v5e_mesh, named_sharding_tree


def test_shape_only_preflight_counts_and_memory():
    abstract = abstract_parameter_tree(HybridForCausalLM(tiny_config()), sequence_length=1)
    report = build_preflight_report(abstract, {"data": 1, "fsdp": 1, "tensor": 2})
    exact_count = sum(leaf.size for leaf in jax.tree.leaves(abstract))
    assert report.parameter_count == exact_count
    assert report.tensor_count == len(jax.tree.leaves(abstract))
    assert report.partitioned_tensor_count > 0
    assert report.weight_bytes_per_device < report.global_weight_bytes
    assert report.training_bytes_per_device == report.weight_bytes_per_device * 3


def test_lion_state_layout_matches_parameter_layout():
    abstract = abstract_parameter_tree(HybridForCausalLM(tiny_config()), sequence_length=1)
    mesh = create_v5e_mesh()
    param_layout = named_sharding_tree(abstract, mesh)
    tx = create_lion(total_steps=4, warmup_steps=1)
    abstract_state, state_layout = optimizer_state_layout(tx, abstract, param_layout, mesh)

    lion_index = next(i for i, item in enumerate(abstract_state) if hasattr(item, "mu"))
    param = param_layout["layers_0"]["mamba"]["in_proj"]["kernel"]
    momentum = state_layout[lion_index].mu["layers_0"]["mamba"]["in_proj"]["kernel"]
    momentum_shape = abstract_state[lion_index].mu["layers_0"]["mamba"]["in_proj"]["kernel"]
    assert momentum.spec == param.spec
    assert momentum_shape.dtype == jnp.bfloat16
