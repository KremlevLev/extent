from __future__ import annotations

import json

from scripts.qwen_tpu_campaign import (
    _context_cache_arguments,
    _prune_completed_depth_arrays,
    aggregate_context_transfer,
)


def test_context_cache_keeps_fixed_token_budget(tmp_path):
    arguments, manifest, artifact_dir = _context_cache_arguments(
        layer=18,
        sequence_length=64,
        qwen_cache_dir="weights",
        dataset_cache_dir="dataset",
        output_dir=tmp_path,
        compute_dtype="bfloat16",
        storage_dtype="float16",
        per_device_windows=1,
    )
    assert manifest.name == "exp047-layer18-seq64-manifest.json"
    assert artifact_dir.name == "exp047-layer18-seq64-cache"
    assert arguments[arguments.index("--evaluation-windows") + 1] == "128"
    assert arguments[arguments.index("--sequence-length") + 1] == "64"
    assert "--prune-consumed-shards" in arguments


def test_context_aggregate_applies_cell_count_best_count_and_regret_gate():
    results = {}
    for layer in (0, 18):
        for length in (64, 128, 256):
            results[f"layer{layer}-seq{length}"] = {
                "target_layer": layer,
                "sequence_length": length,
                "passed": True,
                "aggregate": {
                    "mean_contribution_minus_mixer_nll": -0.01,
                    "mean_contribution_minus_joint_nll": -0.005,
                },
            }
    aggregate = aggregate_context_transfer(results)
    assert aggregate["contribution_best_cells"] == 6
    assert aggregate["maximum_contribution_regret_nll"] == -0.005
    assert aggregate["scientific_gate_passed"]
    results["layer18-seq256"]["aggregate"][
        "mean_contribution_minus_joint_nll"
    ] = 0.021
    assert not aggregate_context_transfer(results)["scientific_gate_passed"]


def test_depth_array_pruning_preserves_manifests_and_requires_result(tmp_path):
    cache = tmp_path / "exp045-layer0-train-cache"
    cache.mkdir()
    array = cache / "values.npy"
    manifest = cache / "manifest.json"
    array.write_bytes(b"regenerable")
    manifest.write_text("{}", encoding="utf-8")
    assert _prune_completed_depth_arrays(tmp_path) == []
    (tmp_path / "exp045-depth-objective.json").write_text(
        json.dumps({"passed": True, "protocol": "exp045-layer0-layer18"}),
        encoding="utf-8",
    )
    removed = _prune_completed_depth_arrays(tmp_path)
    assert removed == [str(array.resolve())]
    assert not array.exists()
    assert manifest.exists()
