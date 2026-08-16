import jax

from singularity import HybridForCausalLM, tiny_config
from singularity.initialization import abstract_parameter_tree
from singularity.preflight import build_preflight_report


def test_shape_only_preflight_counts_and_memory():
    abstract = abstract_parameter_tree(HybridForCausalLM(tiny_config()), sequence_length=1)
    report = build_preflight_report(abstract, {"data": 1, "fsdp": 1, "tensor": 2})
    exact_count = sum(leaf.size for leaf in jax.tree.leaves(abstract))
    assert report.parameter_count == exact_count
    assert report.tensor_count == len(jax.tree.leaves(abstract))
    assert report.partitioned_tensor_count > 0
    assert report.weight_bytes_per_device < report.global_weight_bytes
    assert report.training_bytes_per_device == report.weight_bytes_per_device * 3
