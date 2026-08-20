import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from extent.qwen3_teacher import (
    Qwen3ForCausalLM,
    apply_qwen3_rope,
    tiny_qwen3_teacher_config,
)
from extent.weight_mapping import (
    expected_qwen_shape,
    teacher_qwen_mappings,
    validate_teacher_mapping_plan,
)


def test_tiny_qwen3_teacher_shapes_dtypes_and_jit():
    config = tiny_qwen3_teacher_config()
    model = Qwen3ForCausalLM(config)
    tokens = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)
    params = model.init(jax.random.key(0), tokens)["params"]
    logits = jax.jit(model.apply)({"params": params}, tokens)

    assert logits.shape == (2, 4, config.vocab_size)
    assert logits.dtype == jnp.float32
    assert params["layers_0"]["self_attn"]["q_norm"]["scale"].shape == (config.head_dim,)
    assert params["layers_0"]["self_attn"]["k_norm"]["scale"].shape == (config.head_dim,)
    assert all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(params))


def test_qwen3_rope_uses_split_half_rotation():
    tensor = jnp.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=jnp.float32)
    positions = jnp.array([[1]], dtype=jnp.int32)
    actual = apply_qwen3_rope(tensor, positions, theta=100.0)
    angles = np.array([1.0, 0.1], dtype=np.float32)
    cos = np.tile(np.cos(angles), 2)
    sin = np.tile(np.sin(angles), 2)
    rotated = np.array([-3.0, -4.0, 1.0, 2.0], dtype=np.float32)
    expected = np.array([1.0, 2.0, 3.0, 4.0]) * cos + rotated * sin
    np.testing.assert_allclose(np.asarray(actual[0, 0, 0]), expected, rtol=1e-6)


def test_teacher_is_causal():
    config = tiny_qwen3_teacher_config()
    model = Qwen3ForCausalLM(config)
    first = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    second = jnp.array([[1, 2, 99, 100]], dtype=jnp.int32)
    params = model.init(jax.random.key(1), first)["params"]
    first_logits = model.apply({"params": params}, first)
    second_logits = model.apply({"params": params}, second)
    np.testing.assert_allclose(
        np.asarray(first_logits[:, :2]), np.asarray(second_logits[:, :2]), rtol=0, atol=0
    )


def test_complete_teacher_mapping_is_bijective():
    config = tiny_qwen3_teacher_config()
    model = Qwen3ForCausalLM(config)
    params = jax.eval_shape(
        model.init, jax.random.key(2), jax.ShapeDtypeStruct((1, 1), jnp.int32)
    )["params"]
    entries = teacher_qwen_mappings(config)
    weight_map = {entry.source: "unused.safetensors" for entry in entries}
    report = validate_teacher_mapping_plan(config, params, weight_map)
    expected_parameters = sum(
        int(np.prod(expected_qwen_shape(entry, config))) for entry in entries
    )

    assert report.tensor_count == len(traverse_util.flatten_dict(params))
    assert report.parameter_count == expected_parameters
