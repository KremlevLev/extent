from __future__ import annotations

import json

import pytest

from extent.hf_checkpoint_sync import (
    normalize_hub_path,
    upload_checkpoint_to_hub,
    validate_checkpoint_directory,
)


def _checkpoint(tmp_path):
    payload = tmp_path / "training_state.msgpack"
    payload.write_bytes(b"verified-state")
    import hashlib

    metadata = {
        "step": 12,
        "checkpoint_file": payload.name,
        "checkpoint_bytes": payload.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(metadata))
    return metadata


def test_checkpoint_validation_and_hub_path_gate(tmp_path):
    metadata = _checkpoint(tmp_path)
    restored, files = validate_checkpoint_directory(tmp_path)
    assert restored == metadata
    assert set(files) == {"checkpoint.json", "training_state.msgpack"}
    assert normalize_hub_path("experiments/exp036/step-1024") == "experiments/exp036/step-1024"
    with pytest.raises(ValueError, match="relative"):
        normalize_hub_path("../escape")


def test_push_is_one_commit_with_only_verified_files(tmp_path):
    _checkpoint(tmp_path)

    class FakeApi:
        def __init__(self):
            self.created = None
            self.commit = None

        def create_repo(self, **kwargs):
            self.created = kwargs

        def create_commit(self, **kwargs):
            self.commit = kwargs
            return type("Info", (), {"oid": "abc123"})()

    api = FakeApi()
    info = upload_checkpoint_to_hub(
        tmp_path,
        repo_id="owner/private-checkpoints",
        path_in_repo="exp036/step-1024",
        token="secret",
        api=api,
    )
    assert info.oid == "abc123"
    assert api.created["private"] is True
    assert len(api.commit["operations"]) == 2
    assert {
        operation.path_in_repo for operation in api.commit["operations"]
    } == {
        "exp036/step-1024/checkpoint.json",
        "exp036/step-1024/training_state.msgpack",
    }
