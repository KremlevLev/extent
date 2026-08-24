from __future__ import annotations

import numpy as np

from scripts.qwen_depth_objective_run import (
    RAM_CACHE_MINIMUM_FREE_BYTES,
    _cache_arguments,
    resolve_qwen_cache_dir,
    stage_saved_output,
)
from scripts.qwen_streamed_multiseed_end_to_end import (
    aggregate_depth_objectives,
    bootstrap_depth_objective_deltas,
    branch_names,
)


def test_depth_objective_branch_order_includes_paired_counterfactual_arm():
    assert branch_names((123,), 1024, include_contribution=True) == (
        "ORIGINAL-CACHED-QWEN",
        "SEED-123-MIXER-ONLY-STEP1024",
        "SEED-123-JOINT-STEP1024",
        "SEED-123-CONTRIBUTION-STEP1024",
    )


def test_depth_objective_aggregate_uses_absolute_paired_nll_deltas():
    seed_metrics = {
        str(seed): {
            "mixer_only": {"mean_nll": 5.0 + offset},
            "joint": {"mean_nll": 4.8 + offset},
            "contribution": {"mean_nll": 4.6 + offset},
        }
        for seed, offset in ((1, 0.0), (2, 0.1), (3, -0.1))
    }
    result = aggregate_depth_objectives(4.0, seed_metrics)
    np.testing.assert_allclose(result["mean_contribution_minus_mixer_nll"], -0.4)
    np.testing.assert_allclose(result["mean_contribution_minus_joint_nll"], -0.2)
    assert result["contribution_wins_vs_mixer"] == 3
    assert result["contribution_wins_vs_joint"] == 3
    assert result["all_finite"]


def test_depth_objective_bootstrap_is_paired_and_deterministic():
    windows = {
        str(seed): {
            "mixer_only": [2.0, 2.1, 1.9, 2.2],
            "joint": [1.8, 1.9, 1.7, 2.0],
            "contribution": [1.5, 1.6, 1.4, 1.7],
        }
        for seed in (123, 456, 789)
    }
    first = bootstrap_depth_objective_deltas(windows, samples=100, seed=7)
    second = bootstrap_depth_objective_deltas(windows, samples=100, seed=7)
    assert first == second
    assert first["contribution_minus_mixer_95ci"][1] < 0
    assert first["contribution_minus_joint_95ci"][1] < 0


def test_exp045_cache_protocol_is_frozen(tmp_path):
    arguments, manifest, artifact_dir = _cache_arguments(
        layer=18,
        evaluation_only=True,
        qwen_cache_dir="weights",
        dataset_cache_dir="dataset",
        output_dir=tmp_path,
        compute_dtype="bfloat16",
        storage_dtype="float16",
        per_device_windows=1,
    )
    assert manifest.name == "exp045-layer18-validation-manifest.json"
    assert artifact_dir.name == "exp045-layer18-validation-cache"
    assert arguments[arguments.index("--evaluation-windows") + 1] == "256"
    assert arguments[arguments.index("--sequence-length") + 1] == "32"
    assert "--evaluation-only" in arguments
    assert "--prune-consumed-shards" in arguments


def test_exp048_cache_uses_requested_unique_training_window_budget(tmp_path):
    arguments, manifest, artifact_dir = _cache_arguments(
        layer=18,
        evaluation_only=False,
        qwen_cache_dir="weights",
        dataset_cache_dir="dataset",
        output_dir=tmp_path,
        compute_dtype="bfloat16",
        storage_dtype="float16",
        per_device_windows=1,
        experiment="exp048-long",
        training_windows=4096,
    )
    assert manifest.name == "exp048-long-layer18-train-manifest.json"
    assert artifact_dir.name == "exp048-long-layer18-train-cache"
    assert arguments[arguments.index("--training-windows") + 1] == "4096"


def test_exp045_qwen_cache_falls_back_when_ramdisk_is_too_small(
    tmp_path, monkeypatch
):
    usage = type(
        "Usage", (), {"free": RAM_CACHE_MINIMUM_FREE_BYTES - 1}
    )()
    monkeypatch.setattr(
        "scripts.qwen_depth_objective_run.shutil.disk_usage",
        lambda path: usage,
    )
    path, storage = resolve_qwen_cache_dir(
        "disk-cache", "auto", ram_root=tmp_path
    )
    assert path == "disk-cache"
    assert storage == "disk"


def test_exp045_qwen_cache_uses_large_ramdisk(tmp_path, monkeypatch):
    usage = type(
        "Usage", (), {"free": RAM_CACHE_MINIMUM_FREE_BYTES}
    )()
    monkeypatch.setattr(
        "scripts.qwen_depth_objective_run.shutil.disk_usage",
        lambda path: usage,
    )
    path, storage = resolve_qwen_cache_dir(
        "disk-cache", "auto", ram_root=tmp_path
    )
    assert path == str(tmp_path / "extent-qwen3-exp045-weights")
    assert storage == "ram"


def test_exp045_saved_output_requires_stage_manifest(tmp_path):
    source = tmp_path / "saved"
    source.mkdir()
    with np.testing.assert_raises_regex(ValueError, "does not contain EXP-045"):
        stage_saved_output(source, tmp_path / "working")
