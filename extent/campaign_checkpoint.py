"""Verified local/HF state slots for single-host resumable campaigns."""

from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

from flax import serialization
import jax
import numpy as np

from extent.hf_checkpoint_sync import (
    _sha256, download_checkpoint_from_hub,
    validate_checkpoint_directory,
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    staging.replace(path)


class CampaignCheckpointStore:
    """Slot metadata is committed last; previous local payload survives a failed save.

    A slot holds model parameters, optional optimizer state, and the matching
    data cursor/metrics. Checkpoints use host memory for serialization and should
    live in a sufficiently large RAM filesystem on Kaggle.
    """

    def __init__(self, root: Path, prefix: str, hub=None):
        self.root, self.prefix, self.hub = Path(root), prefix, hub
        self.events: list[dict] = []
        self.checked_remote: set[str] = set()

    def metadata(self, slot: str, contract: dict) -> dict | None:
        directory = self.root / slot
        if not (directory / "checkpoint.json").exists() and self.hub is not None:
            if slot not in self.checked_remote:
                from huggingface_hub.errors import EntryNotFoundError

                try:
                    downloaded = download_checkpoint_from_hub(
                        directory, repo_id=self.hub.repo_id,
                        repo_type=self.hub.repo_type, token=self.hub.token,
                        revision=self.hub.revision,
                        path_in_repo=f"{self.prefix}/{slot}",
                    )
                    (directory / "synced.sha256").write_text(downloaded["checkpoint_sha256"], encoding="utf-8")
                    self.events.append({"operation": "restore", "slot": slot, "passed": True})
                except EntryNotFoundError:
                    pass
                # Auth/network errors propagate: never silently repeat expensive work.
                self.checked_remote.add(slot)
        if not (directory / "checkpoint.json").exists():
            return None
        metadata, _ = validate_checkpoint_directory(directory)
        if metadata.get("contract") != contract:
            raise ValueError(f"checkpoint contract mismatch: {slot}")
        marker = directory / "synced.sha256"
        if self.hub and (not marker.exists() or marker.read_text(encoding="utf-8") != metadata["checkpoint_sha256"]):
            self.sync(slot)
        return metadata

    def restore(self, slot: str, contract: dict, template=None):
        metadata = self.metadata(slot, contract)
        if metadata is None:
            return None
        payload = (self.root / slot / metadata["checkpoint_file"]).read_bytes()
        state = (
            serialization.msgpack_restore(payload) if template is None
            else serialization.from_bytes(template, payload)
        )
        return state, metadata

    def save(self, slot: str, state, *, contract: dict, step: int, metrics: dict) -> dict:
        directory = self.root / slot
        directory.mkdir(parents=True, exist_ok=True)
        old = None
        if (directory / "checkpoint.json").exists():
            old = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        payload = directory / f"state-{uuid.uuid4().hex}.msgpack"
        staging = payload.with_suffix(".tmp")
        staging.write_bytes(serialization.to_bytes(jax.device_get(state)))
        staging.replace(payload)
        metadata = {
            "format_version": 1, "step": int(step), "contract": contract,
            "checkpoint_file": payload.name,
            "checkpoint_bytes": payload.stat().st_size,
            "checkpoint_sha256": _sha256(payload), "metrics": metrics,
        }
        write_json_atomic(directory / "checkpoint.json", metadata)
        # Only the previous file explicitly referenced by this slot is removed.
        if old:
            previous = directory / old["checkpoint_file"]
            if previous.parent.resolve() != directory.resolve():
                raise ValueError("invalid previous checkpoint path")
            if previous != payload and previous.is_file():
                previous.unlink()
        self.sync(slot)
        return metadata

    def sync(self, slot: str) -> bool:
        if self.hub is None:
            return False
        # Stable Hub filename prevents accumulating a new live multi-GB file at
        # every step. Local payloads remain generation-named for crash consistency.
        from huggingface_hub import CommitOperationAdd, HfApi

        directory = self.root / slot
        metadata, _ = validate_checkpoint_directory(directory)
        remote_meta = dict(metadata, checkpoint_file="state.msgpack")
        remote = f"{self.prefix}/{slot}"
        api = HfApi(token=self.hub.token)
        for attempt in range(3):
            try:
                api.create_commit(
                    repo_id=self.hub.repo_id, repo_type=self.hub.repo_type,
                    revision=self.hub.revision,
                    commit_message=f"{slot} step {metadata['step']}",
                    operations=[
                        CommitOperationAdd(
                            path_in_repo=f"{remote}/state.msgpack",
                            path_or_fileobj=directory / metadata["checkpoint_file"],
                        ),
                        CommitOperationAdd(
                            path_in_repo=f"{remote}/checkpoint.json",
                            path_or_fileobj=json.dumps(remote_meta, allow_nan=False).encode(),
                        ),
                    ],
                )
                self.events.append({"operation": "upload", "slot": slot, "passed": True,
                                    "step": metadata["step"]})
                (directory / "synced.sha256").write_text(metadata["checkpoint_sha256"], encoding="utf-8")
                print(f"hf_checkpoint=PASS slot={slot} step={metadata['step']}")
                return True
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    self.events.append({"operation": "upload", "slot": slot, "passed": False,
                                        "step": metadata['step'], "error_type": type(exc).__name__})
                    print(f"hf_checkpoint=FAILED slot={slot}; local checkpoint retained")
        return False
