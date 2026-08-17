import numpy as np
import pytest
from dataclasses import replace

from singularity.qwen3_parity import (
    jax_attention_params,
    jax_layer_params,
    layer_mapping_entries,
    parity_metrics,
    required_layer_shards,
    torch_layer_state,
)
from singularity.qwen3_teacher import Qwen3DecoderLayer, tiny_qwen3_teacher_config
from singularity.weight_mapping import expected_qwen_shape


def test_one_teacher_layer_has_all_eleven_qwen_tensors():
    config = tiny_qwen3_teacher_config()
    entries = layer_mapping_entries(config, 1)
    assert len(entries) == 11
    weight_map = {
        entry.source: ("first.safetensors" if index < 5 else "second.safetensors")
        for index, entry in enumerate(entries)
    }
    assert required_layer_shards(weight_map, config, 1) == (
        "first.safetensors",
        "second.safetensors",
    )


def test_parity_metrics_and_frozen_gate():
    reference = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    candidate = reference + np.array([[1e-4, 0.0, -1e-4]], dtype=np.float32)
    metrics = parity_metrics(reference, candidate)
    assert metrics.max_abs == pytest.approx(1e-4, rel=1e-3)
    assert metrics.cosine_similarity > 0.999999
    assert metrics.passes(max_abs_tolerance=5e-3, relative_l2_tolerance=5e-4)
    assert not metrics.passes(max_abs_tolerance=1e-5, relative_l2_tolerance=5e-4)


def test_tiny_jax_layer_matches_transformers_qwen3():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import jax
    import jax.numpy as jnp
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer as TorchQwen3DecoderLayer,
        Qwen3RotaryEmbedding,
    )

    config = tiny_qwen3_teacher_config()
    rng = np.random.default_rng(7)
    arrays = {}
    for entry in layer_mapping_entries(config, 0):
        shape = expected_qwen_shape(entry, config)
        arrays[entry.source] = (
            np.ones(shape, dtype=np.float32)
            if entry.source.endswith("norm.weight") or entry.source.endswith("layernorm.weight")
            else rng.normal(0.0, 0.02, shape).astype(np.float32)
        )

    torch_config = Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=False,
        attention_dropout=0.0,
    )
    torch_config._attn_implementation = "eager"
    with torch.device("meta"):
        torch_layer = TorchQwen3DecoderLayer(torch_config, layer_idx=0)
    torch_layer.load_state_dict(torch_layer_state(arrays, config, 0), strict=True, assign=True)
    torch_layer.eval()

    hidden = rng.normal(size=(1, 4, config.hidden_size)).astype(np.float32)
    torch_hidden = torch.from_numpy(hidden)
    positions = torch.arange(4, dtype=torch.long)[None, :]
    rotary = Qwen3RotaryEmbedding(torch_config)
    position_embeddings = rotary(torch_hidden, positions)
    causal = torch.full((1, 1, 4, 4), torch.finfo(torch.float32).min)
    causal = torch.triu(causal, diagonal=1)
    with torch.inference_mode():
        torch_output = torch_layer(
            torch_hidden,
            attention_mask=causal,
            position_ids=positions,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    if isinstance(torch_output, tuple):
        torch_output = torch_output[0]

    fp32_config = replace(config, param_dtype="float32", compute_dtype="float32")
    jax_output = Qwen3DecoderLayer(fp32_config).apply(
        {"params": jax_layer_params(arrays, config, 0)},
        jnp.asarray(hidden),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.ones((1, 4), dtype=jnp.bool_),
    )
    metrics = parity_metrics(torch_output.numpy(), np.asarray(jax_output))
    assert metrics.max_abs < 5e-5
    assert metrics.relative_l2 < 5e-5


def test_attention_only_loader_excludes_mlp_tensors():
    config = tiny_qwen3_teacher_config()
    rng = np.random.default_rng(9)
    arrays = {
        entry.source: rng.normal(
            size=expected_qwen_shape(entry, config)
        ).astype(np.float32)
        for entry in layer_mapping_entries(config, 0)
    }
    params = jax_attention_params(arrays, config, 0)
    assert set(params) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
    }
