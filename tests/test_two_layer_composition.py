from __future__ import annotations

import numpy as np
import pytest

from scripts.qwen_two_layer_composition import (
    _validate_cache_pair,
    composition_branch_names,
)
from scripts.qwen_two_layer_composition_run import _cache_arguments


def _manifest(layer: int, *, evaluation: bool) -> dict:
    return {
        "source": "Qwen/Qwen3-14B@pinned",
        "dataset": "wikitext@pinned",
        "target_layer": layer,
        "sequence_length": 32,
        "dataset_split": "validation" if evaluation else "train",
        "token_offset": 0,
        "token_range": [0, 64],
        "window_layout": {
            "calibration": [0, 0] if evaluation else [0, 1],
            "training": [0, 0] if evaluation else [1, 1],
            "evaluation": [0, 2] if evaluation else [1, 2],
            "total_windows": 2,
        },
        "evaluation_only": evaluation,
    }


def test_composition_branch_order_is_frozen():
    assert composition_branch_names((123, 456, 789)) == (
        "ORIGINAL-CACHED-QWEN",
        "SEED-123-LAYER0-ONLY",
        "SEED-123-LAYER18-ONLY",
        "SEED-123-LAYER0-LAYER18",
        "SEED-456-LAYER0-ONLY",
        "SEED-456-LAYER18-ONLY",
        "SEED-456-LAYER0-LAYER18",
        "SEED-789-LAYER0-ONLY",
        "SEED-789-LAYER18-ONLY",
        "SEED-789-LAYER0-LAYER18",
    )


def test_composition_cache_pair_requires_matching_tokens_and_roles():
    layer0 = _manifest(0, evaluation=True)
    layer18 = _manifest(18, evaluation=True)
    arrays = {"token_ids": np.asarray([[1, 2], [3, 4]])}
    _validate_cache_pair(layer0, layer18, arrays, arrays, evaluation=True)
    bad = {"token_ids": np.asarray([[1, 2], [3, 5]])}
    with pytest.raises(ValueError, match="different token IDs"):
        _validate_cache_pair(layer0, layer18, arrays, bad, evaluation=True)
    with pytest.raises(ValueError, match="evaluation_only"):
        _validate_cache_pair(layer0, layer18, arrays, arrays, evaluation=False)


def test_one_shot_cache_protocol_is_frozen(tmp_path):
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
    assert manifest.name == "exp044-layer18-validation-manifest.json"
    assert artifact_dir.name == "exp044-layer18-validation-cache"
    assert manifest.parent == artifact_dir
    assert arguments[arguments.index("--target-layer") + 1] == "18"
    assert arguments[arguments.index("--evaluation-windows") + 1] == "256"
    assert "--evaluation-only" in arguments
    assert "--data-parallel" in arguments
