"""Small, resumable scientific-artifact synchronization through HF Hub."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from extent.hf_checkpoint_sync import normalize_hub_path, validate_repo_type


@dataclass(frozen=True)
class HubArtifactConfig:
    repo_id: str
    repo_type: str
    token: str
    revision: str = "main"


def artifact_config_from_env(*, include_kaggle_secrets: bool = True) -> HubArtifactConfig | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    repo_id = os.environ.get("EXTENT_HF_CHECKPOINT_REPO", "").strip()
    repo_type = os.environ.get("EXTENT_HF_REPO_TYPE", "dataset").strip()
    if (not token or not repo_id) and include_kaggle_secrets:
        try:
            from kaggle_secrets import UserSecretsClient

            secrets = UserSecretsClient()
            token = token or secrets.get_secret("HF_TOKEN").strip()
            repo_id = repo_id or secrets.get_secret(
                "EXTENT_HF_CHECKPOINT_REPO"
            ).strip()
            if "EXTENT_HF_REPO_TYPE" not in os.environ:
                repo_type = secrets.get_secret("EXTENT_HF_REPO_TYPE").strip()
        except Exception:  # Optional environment integration must never block a run.
            return None
    if not token or not repo_id:
        return None
    return HubArtifactConfig(repo_id, validate_repo_type(repo_type), token)


def upload_artifact(
    local_path: str | Path,
    path_in_repo: str,
    config: HubArtifactConfig,
    *,
    commit_message: str,
    api: Any | None = None,
):
    from huggingface_hub import HfApi

    source = Path(local_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = normalize_hub_path(path_in_repo)
    api = HfApi(token=config.token) if api is None else api
    return api.upload_file(
        path_or_fileobj=source,
        path_in_repo=destination,
        repo_id=config.repo_id,
        repo_type=config.repo_type,
        revision=config.revision,
        commit_message=commit_message,
        token=config.token,
    )


def restore_artifact(
    local_path: str | Path,
    path_in_repo: str,
    config: HubArtifactConfig,
    *,
    download: Any | None = None,
) -> bool:
    """Atomically restore one artifact; return False when it does not exist."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    remote = normalize_hub_path(path_in_repo)
    download = hf_hub_download if download is None else download
    try:
        with tempfile.TemporaryDirectory(
            prefix="extent-hf-artifact-", dir=destination.parent
        ) as temporary:
            fetched = Path(
                download(
                    repo_id=config.repo_id,
                    repo_type=config.repo_type,
                    revision=config.revision,
                    filename=remote,
                    local_dir=temporary,
                    token=config.token,
                )
            )
            staging = destination.with_suffix(destination.suffix + ".download")
            shutil.copyfile(fetched, staging)
            staging.replace(destination)
    except EntryNotFoundError:
        return False
    return True
