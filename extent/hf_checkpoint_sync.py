from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_hub_path(path_in_repo: str) -> str:
    path = PurePosixPath(path_in_repo)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("path-in-repo must be a non-empty relative Hub path")
    return path.as_posix().rstrip("/")


def validate_checkpoint_directory(directory: str | Path) -> tuple[dict, dict[str, Path]]:
    root = Path(directory)
    metadata_path = root / "checkpoint.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload_path = root / metadata["checkpoint_file"]
    if not payload_path.exists():
        raise FileNotFoundError(f"checkpoint payload is missing: {payload_path}")
    if payload_path.stat().st_size != int(metadata["checkpoint_bytes"]):
        raise ValueError("checkpoint payload byte-size mismatch")
    if _sha256(payload_path) != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint payload SHA-256 mismatch")
    files = {
        "checkpoint.json": metadata_path,
        metadata["checkpoint_file"]: payload_path,
    }
    metrics = root / "metrics.json"
    if metrics.exists():
        files["metrics.json"] = metrics
    return metadata, files


def upload_checkpoint_to_hub(
    directory: str | Path,
    *,
    repo_id: str,
    path_in_repo: str,
    token: str | bool | None = None,
    revision: str = "main",
    private: bool = True,
    commit_message: str | None = None,
    api: Any | None = None,
):
    """Upload a verified checkpoint as one atomic Hugging Face Hub commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    metadata, files = validate_checkpoint_directory(directory)
    prefix = normalize_hub_path(path_in_repo)
    api = HfApi(token=token) if api is None else api
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{prefix}/{name}", path_or_fileobj=path
        )
        for name, path in files.items()
    ]
    message = commit_message or f"checkpoint step {int(metadata['step'])}"
    return api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        operations=operations,
        commit_message=message,
        token=token,
    )


def download_checkpoint_from_hub(
    directory: str | Path,
    *,
    repo_id: str,
    path_in_repo: str,
    token: str | bool | None = None,
    revision: str = "main",
) -> dict:
    """Download, verify, and atomically install a Hub checkpoint locally."""
    from huggingface_hub import hf_hub_download

    root = Path(directory)
    root.parent.mkdir(parents=True, exist_ok=True)
    prefix = normalize_hub_path(path_in_repo)
    with tempfile.TemporaryDirectory(
        prefix="extent-hf-checkpoint-", dir=root.parent
    ) as temporary:
        temporary_root = Path(temporary)
        metadata_download = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=f"{prefix}/checkpoint.json",
                local_dir=temporary_root,
                token=token,
            )
        )
        metadata = json.loads(metadata_download.read_text(encoding="utf-8"))
        payload_name = metadata["checkpoint_file"]
        payload_download = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=f"{prefix}/{payload_name}",
                local_dir=temporary_root,
                token=token,
            )
        )
        if payload_download.stat().st_size != int(metadata["checkpoint_bytes"]):
            raise ValueError("downloaded checkpoint byte-size mismatch")
        if _sha256(payload_download) != metadata["checkpoint_sha256"]:
            raise ValueError("downloaded checkpoint SHA-256 mismatch")

        root.mkdir(parents=True, exist_ok=True)
        temporary_payload = root / f"{payload_name}.download"
        shutil.copyfile(payload_download, temporary_payload)
        temporary_payload.replace(root / payload_name)
        temporary_metadata = root / "checkpoint.download.json"
        shutil.copyfile(metadata_download, temporary_metadata)
        temporary_metadata.replace(root / "checkpoint.json")
    validate_checkpoint_directory(root)
    return metadata
