from __future__ import annotations

import json

import pytest

from extent.experiment_stage import update_stage_manifest


def test_stage_manifest_accumulates_atomic_progress(tmp_path):
    path = tmp_path / "stage.json"
    update_stage_manifest(
        path,
        experiment="exp044",
        stage="cache-layer0",
        status="completed",
        details={"manifest": "layer0.json"},
    )
    result = update_stage_manifest(
        path,
        experiment="exp044",
        stage="training-layer0",
        status="completed",
        details={"sha256": "abc"},
    )
    assert set(result["stages"]) == {"cache-layer0", "training-layer0"}
    assert json.loads(path.read_text())["current_stage"] == "training-layer0"
    assert result["status"] == "running"
    assert not path.with_name("stage.json.tmp").exists()
    with pytest.raises(ValueError, match="another experiment"):
        update_stage_manifest(
            path,
            experiment="exp045",
            stage="cache",
            status="running",
        )
