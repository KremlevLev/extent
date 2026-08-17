import jax
import jax.numpy as jnp
import numpy as np

from singularity.layers.mla import MultiHeadLatentAttention, apply_partial_rope
from singularity.mla_conversion import (
    convert_qwen3_gqa_to_mla_joint_svd,
    factorize_qwen3_joint_kv,
    partial_rope_indices,
    qwen3_mla_conversion_config,
)
from singularity.qwen3_parity import layer_mapping_entries
from singularity.qwen3_teacher import tiny_qwen3_teacher_config
from singularity.weight_mapping import expected_qwen_shape


def _tiny_arrays():
    source = tiny_qwen3_teacher_config()
    rng = np.random.default_rng(4)
    arrays = {}
    for entry in layer_mapping_entries(source, 0):
        shape = expected_qwen_shape(entry, source)
        arrays[entry.source] = (
            np.ones(shape, dtype=np.float32)
            if entry.source.endswith(("norm.weight", "layernorm.weight"))
            else rng.normal(0.0, 0.02, shape).astype(np.float32)
        )
    return source, arrays


def test_partial_rope_indices_keep_split_half_pairs():
    content, rope = partial_rope_indices(16, 8, "high")
    np.testing.assert_array_equal(rope, [0, 1, 2, 3, 8, 9, 10, 11])
    np.testing.assert_array_equal(content, [4, 5, 6, 7, 12, 13, 14, 15])
    x = jnp.arange(8, dtype=jnp.float32).reshape(1, 1, 1, 8)
    at_zero = apply_partial_rope(x, jnp.array([[0]]), 1_000_000.0, 16, "high")
    np.testing.assert_array_equal(np.asarray(at_zero), np.asarray(x))


def test_joint_svd_grouped_and_shared_rope_are_runnable():
    source, arrays = _tiny_arrays()
    x = jnp.ones((1, 3, source.hidden_size), dtype=jnp.float32)
    positions = jnp.arange(3, dtype=jnp.int32)[None, :]
    for grouped in (True, False):
        config = qwen3_mla_conversion_config(
            source, kv_lora_rank=16, rope_dim=8, grouped_rope=grouped
        )
        params, report = convert_qwen3_gqa_to_mla_joint_svd(
            arrays, source, config, 0
        )
        output = MultiHeadLatentAttention(
            source.hidden_size,
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        ).apply({"params": params}, x, positions)
        assert output.shape == x.shape
        assert np.all(np.isfinite(np.asarray(output)))
        if grouped:
            assert report.rope_aggregation_relative_l2 == 0.0
            assert report.target_cache_elements_per_token == 32
        else:
            assert report.rope_aggregation_relative_l2 > 0.0
            assert report.target_cache_elements_per_token == 24


def test_one_factorization_supports_monotonic_rank_sweep():
    source, arrays = _tiny_arrays()
    factors = factorize_qwen3_joint_kv(
        arrays, source, 0, rope_dim=8, max_rank=32, seed=3
    )
    errors = []
    for rank in (8, 16, 32):
        config = qwen3_mla_conversion_config(
            source, kv_lora_rank=rank, rope_dim=8, grouped_rope=True
        )
        _, report = convert_qwen3_gqa_to_mla_joint_svd(
            arrays, source, config, 0, factors=factors
        )
        errors.append(report.joint_reconstruction_relative_l2)
    assert errors[0] >= errors[1] >= errors[2]
