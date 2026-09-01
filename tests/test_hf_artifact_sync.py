from __future__ import annotations

from pathlib import Path

from extent.hf_artifact_sync import (
    HubArtifactConfig,
    artifact_config_from_env,
    restore_artifact,
    upload_artifact,
)


def test_artifact_env_config_is_opt_in(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("EXTENT_HF_CHECKPOINT_REPO", raising=False)
    assert artifact_config_from_env(include_kaggle_secrets=False) is None
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("EXTENT_HF_CHECKPOINT_REPO", "owner/extent")
    monkeypatch.setenv("EXTENT_HF_REPO_TYPE", "dataset")
    assert artifact_config_from_env(include_kaggle_secrets=False) == HubArtifactConfig(
        "owner/extent", "dataset", "secret"
    )


def test_upload_and_restore_use_dataset_namespace(tmp_path):
    source = tmp_path / "result.json"
    source.write_text('{"passed": true}', encoding="utf-8")
    config = HubArtifactConfig("owner/extent", "dataset", "secret")

    class FakeApi:
        kwargs = None

        def upload_file(self, **kwargs):
            self.kwargs = kwargs
            return "uploaded"

    api = FakeApi()
    assert upload_artifact(
        source,
        "experiments/exp067/layer-0.json",
        config,
        commit_message="layer 0",
        api=api,
    ) == "uploaded"
    assert api.kwargs["repo_type"] == "dataset"

    remote = tmp_path / "remote.json"
    remote.write_text('{"restored": true}', encoding="utf-8")

    def fake_download(**kwargs):
        assert kwargs["repo_type"] == "dataset"
        return str(remote)

    destination = tmp_path / "restored" / "result.json"
    assert restore_artifact(
        destination,
        "experiments/exp067/layer-0.json",
        config,
        download=fake_download,
    )
    assert destination.read_text(encoding="utf-8") == '{"restored": true}'
