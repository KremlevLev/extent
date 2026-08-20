from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.qwen_activation_cache import (
    _create_data_parallel_norm_runner,
    _safe_prune_shards,
)
from extent.layers.common import RMSNorm
from extent.teacher_activation_cache import (
    activation_window_layout,
    array_artifact,
    atomic_save_array,
    checkpoint_shard_last_use,
    load_activation_cache,
    run_host_data_parallel,
    run_host_microbatches,
)


def test_activation_window_layout_is_disjoint_and_exhaustive():
    layout = activation_window_layout(8, 80, 4)
    assert layout.calibration == slice(0, 8)
    assert layout.training == slice(8, 88)
    assert layout.evaluation == slice(88, 92)
    assert layout.total_windows == 92
    with pytest.raises(ValueError, match="positive"):
        activation_window_layout(8, 0, 4)


def test_host_microbatch_runner_preserves_order_and_fp32_output():
    inputs = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    output = run_host_microbatches(
        lambda params, batch: batch * params,
        2.0,
        inputs,
        2,
        input_dtype=jnp.float32,
    )
    np.testing.assert_array_equal(output, inputs * 2)
    assert output.dtype == np.float32


def test_data_parallel_host_batches_pad_and_restore_order():
    inputs = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    output = run_host_data_parallel(
        lambda params, batch: batch + params,
        3.0,
        inputs,
        per_device_windows=2,
        device_count=2,
        input_dtype=jnp.float32,
    )
    np.testing.assert_array_equal(output, inputs + 3)


def test_pmapped_norm_runner_executes_on_available_device():
    devices = jax.devices()[:1]
    module = RMSNorm(4, eps=1e-6, param_dtype=jnp.float32)
    params = {"scale": jnp.ones((4,), jnp.float32)}
    inputs = np.ones((3, 2, 4), np.float32)
    runner = _create_data_parallel_norm_runner(module, devices)
    output = run_host_data_parallel(
        runner,
        params,
        inputs,
        per_device_windows=2,
        device_count=1,
        input_dtype=jnp.float32,
    )
    assert output.shape == inputs.shape
    assert np.all(np.isfinite(output))


def test_checkpoint_shard_last_use_tracks_shared_shards():
    weight_map = {
        "model.embed_tokens.weight": "a.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "a.safetensors",
        "model.layers.0.mlp.up_proj.weight": "b.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "b.safetensors",
    }
    assert checkpoint_shard_last_use(weight_map, 1) == {
        "a.safetensors": 0,
        "b.safetensors": 1,
    }


def test_atomic_array_artifact_has_shape_dtype_and_hash(tmp_path):
    path = tmp_path / "cache.npy"
    atomic_save_array(path, np.ones((2, 3), np.float32), np.dtype(np.float16))
    metadata = array_artifact(path)
    assert metadata["shape"] == [2, 3]
    assert metadata["dtype"] == "float16"
    assert metadata["bytes"] > 0
    assert len(metadata["sha256"]) == 64


def test_shard_pruning_is_exact_and_rejects_parent_traversal(tmp_path):
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    shard = model_dir / "model-00001.safetensors"
    shard.write_bytes(b"fixture")
    removed = _safe_prune_shards(
        model_dir, {shard.name: 3, "future.safetensors": 4}, 3
    )
    assert removed == [shard.name]
    assert not shard.exists()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"keep")
    with pytest.raises(ValueError, match="outside checkpoint cache"):
        _safe_prune_shards(model_dir, {"../outside.safetensors": 3}, 3)
    assert outside.exists()


def test_portable_cache_loader_resolves_manifest_sibling_and_checks_hash(tmp_path):
    import json

    artifacts = {}
    shapes = {
        "token_ids": (2, 3),
        "residual_input": (2, 3, 4),
        "normalized_input": (2, 3, 4),
        "attention_target": (2, 3, 4),
    }
    for index, (name, shape) in enumerate(shapes.items()):
        path = tmp_path / f"{name}.npy"
        dtype = np.int32 if name == "token_ids" else np.float32
        atomic_save_array(path, np.full(shape, index, dtype=dtype), np.dtype(dtype))
        artifacts[name] = array_artifact(path)
        artifacts[name]["path"] = f"/unavailable/kaggle/path/{path.name}"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"passed": True, "artifacts": artifacts}), encoding="utf-8"
    )
    _, arrays, paths = load_activation_cache(manifest)
    assert arrays["attention_target"].shape == (2, 3, 4)
    assert paths["residual_input"] == (tmp_path / "residual_input.npy").resolve()
    artifacts["attention_target"]["sha256"] = "0" * 64
    manifest.write_text(
        json.dumps({"passed": True, "artifacts": artifacts}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_activation_cache(manifest)
