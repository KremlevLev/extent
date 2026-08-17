import json

import jax
import numpy as np
import pytest
from safetensors.numpy import save_file

from singularity import HybridForCausalLM, tiny_config
from singularity.qwen_source import (
    QwenSourceSpec,
    validate_source_marker,
    validate_source_metadata,
    write_source_marker,
)
from singularity.weight_mapping import (
    QwenCheckpointReader,
    direct_qwen_mappings,
    expected_qwen_shape,
    stream_direct_qwen_weights,
    validate_direct_mapping_plan,
    validate_local_direct_shapes,
)


def test_source_metadata_is_immutable():
    spec = QwenSourceSpec(
        repo_id="test/qwen",
        revision="abc",
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        num_hidden_layers=3,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        tensor_count=2,
        shard_count=1,
        total_size_bytes=12,
    )
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "num_hidden_layers": 3,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
    }
    index = {
        "metadata": {"total_size": 12},
        "weight_map": {"a": "model.safetensors", "b": "model.safetensors"},
    }
    assert validate_source_metadata(config, index, spec).tensor_count == 2
    with pytest.raises(ValueError, match="does not match pinned source"):
        validate_source_metadata({**config, "hidden_size": 65}, index, spec)


def test_source_marker_pins_repo_and_revision(tmp_path):
    spec = QwenSourceSpec(
        "test/qwen", "abc", "Qwen2ForCausalLM", "qwen2", 1, 8, 16, 2, 1, 32, 1, 1, 10
    )
    write_source_marker(tmp_path, spec)
    validate_source_marker(tmp_path, spec)
    marker = tmp_path / ".singularity_source.json"
    marker.write_text(json.dumps({"repo_id": "test/qwen", "revision": "wrong"}))
    with pytest.raises(ValueError, match="source marker mismatch"):
        validate_source_marker(tmp_path, spec)


def test_stream_direct_weights_validates_shapes_and_preserves_mixers(tmp_path):
    config = tiny_config()
    model = HybridForCausalLM(config)
    tokens = np.zeros((1, 1), np.int32)
    params = model.init(jax.random.key(0), tokens)["params"]
    entries = direct_qwen_mappings(config)
    source_tensors = {}
    for index, entry in enumerate(entries):
        shape = expected_qwen_shape(entry, config)
        source_tensors[entry.source] = np.full(shape, index + 1, np.float32)

    shard_name = "model-00001-of-00001.safetensors"
    save_file(source_tensors, tmp_path / shard_name)
    weight_map = {name: shard_name for name in source_tensors}
    total_size = sum(value.nbytes for value in source_tensors.values())
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    reader = QwenCheckpointReader(tmp_path)
    report = validate_direct_mapping_plan(config, params, reader.weight_map)
    validate_local_direct_shapes(reader, config)
    mixer_before = np.asarray(params["layers_0"]["mamba"]["in_proj"]["kernel"])
    loaded, streamed_report = stream_direct_qwen_weights(params, reader, config)

    assert streamed_report == report
    gate_entry = next(entry for entry in entries if entry.target == "layers_0/mlp/gate_proj/kernel")
    expected_gate = source_tensors[gate_entry.source].T.astype(np.asarray(
        loaded["layers_0"]["mlp"]["gate_proj"]["kernel"]
    ).dtype)
    np.testing.assert_array_equal(
        np.asarray(loaded["layers_0"]["mlp"]["gate_proj"]["kernel"]), expected_gate
    )
    np.testing.assert_array_equal(
        np.asarray(loaded["layers_0"]["mamba"]["in_proj"]["kernel"]), mixer_before
    )
