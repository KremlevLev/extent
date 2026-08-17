import jax.numpy as jnp
import numpy as np

from singularity.layers.common import RMSNorm
from singularity.layers.rorope_bkv import Qwen3RoRoPEBKVAttention
from singularity.qwen3_parity import layer_mapping_entries
from singularity.qwen3_teacher import tiny_qwen3_teacher_config
from singularity.rorope import rorope_attend
from singularity.rorope_bkv_conversion import map_qwen3_to_rorope_bkv
from singularity.weight_mapping import expected_qwen_shape


def _arrays(source):
    rng = np.random.default_rng(12)
    arrays = {}
    for entry in layer_mapping_entries(source, 0):
        shape = expected_qwen_shape(entry, source)
        arrays[entry.source] = (
            np.ones(shape, dtype=np.float32)
            if entry.source.endswith(("norm.weight", "layernorm.weight"))
            else rng.normal(0.0, 0.02, shape).astype(np.float32)
        )
    return arrays


def test_deployable_rorope_bkv_full_rank_matches_uncompressed_path():
    source = tiny_qwen3_teacher_config()
    arrays = _arrays(source)
    rng = np.random.default_rng(13)
    calibration = jnp.asarray(
        rng.normal(size=(1, 64, source.hidden_size)).astype(np.float32)
    )
    joint_width = (2 * source.num_key_value_heads - 1) * source.head_dim
    params, report = map_qwen3_to_rorope_bkv(
        arrays, source, 0, calibration, latent_rank=joint_width, svd_seed=3
    )
    inputs = jnp.asarray(rng.normal(size=(1, 5, source.hidden_size)).astype(np.float32))
    positions = jnp.arange(5, dtype=jnp.int32)[None, :]
    module = Qwen3RoRoPEBKVAttention(
        source, latent_rank=joint_width, dtype=jnp.float32, param_dtype=jnp.float32
    )
    output, cache = module.apply(
        {"params": params}, inputs, positions, return_cache=True
    )

    prefix = "model.layers.0.self_attn"
    kernel = lambda name: jnp.asarray(arrays[f"{prefix}.{name}.weight"].T)
    query = (inputs @ kernel("q_proj")).reshape(
        1, 5, source.num_attention_heads, source.head_dim
    )
    key = (inputs @ kernel("k_proj")).reshape(
        1, 5, source.num_key_value_heads, source.head_dim
    )
    value = (inputs @ kernel("v_proj")).reshape(
        1, 5, source.num_key_value_heads, source.head_dim
    )
    query = RMSNorm(source.head_dim, source.rms_norm_eps, jnp.float32).apply(
        {"params": {"scale": jnp.asarray(arrays[f"{prefix}.q_norm.weight"])}}, query
    )
    key = RMSNorm(source.head_dim, source.rms_norm_eps, jnp.float32).apply(
        {"params": {"scale": jnp.asarray(arrays[f"{prefix}.k_norm.weight"])}}, key
    )
    mapping = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    attended = rorope_attend(
        query,
        key,
        value,
        positions,
        params["rorope_rotations"],
        mapping,
        1,
        source.rope_theta,
    )
    expected = attended.reshape(1, 5, source.hidden_size) @ kernel("o_proj")

    np.testing.assert_allclose(output, expected, atol=3e-4, rtol=3e-4)
    assert cache["kv_latent"].shape == (1, 5, joint_width)
    assert cache["k_rope"].shape == (1, 5, source.head_dim)
    assert report.target_cache_elements_per_token == 64
    assert np.all(np.isfinite(np.asarray(output)))
