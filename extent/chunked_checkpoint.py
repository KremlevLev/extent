"""Bounded host staging and atomic manifests for large FP32 training states."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import uuid
import jax
import numpy as np
from flax import serialization
from extent.campaign_checkpoint import write_json_atomic
from extent.hf_artifact_sync import restore_artifact, upload_artifacts_together


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def leaves(tree, path=()):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from leaves(value, path + (key,))
    else:
        yield path, tree


class ChunkedCheckpointStore:
    def __init__(self, root, prefix, hub=None, chunk_bytes=128 << 20):
        self.root, self.prefix, self.hub = Path(root), prefix, hub
        self.chunk_bytes = chunk_bytes
        self.next_sync = 0.0
        self.sync_errors = {}

    def directory(self, slot):
        parts = Path(slot).parts
        if Path(slot).is_absolute() or ".." in parts:
            raise ValueError("unsafe checkpoint slot")
        return self.root / slot

    def metadata(self, slot, contract):
        directory = self.directory(slot)
        pointer = directory / "checkpoint.json"
        if not pointer.exists() and self.hub is not None:
            restore_artifact(pointer, f"{self.prefix}/{slot}/checkpoint.json", self.hub)
        if not pointer.exists():
            return None
        meta = json.loads(pointer.read_text(encoding="utf-8"))
        if meta["contract"] != contract:
            raise ValueError("chunked checkpoint contract mismatch")
        return meta

    def save(self, slot, contract, payload, step):
        directory = self.directory(slot)
        directory.mkdir(parents=True, exist_ok=True)
        tree = serialization.to_state_dict(payload)
        size = sum(int(x.size * x.dtype.itemsize) for _, x in leaves(tree))
        # Keep previous generation until the new complete manifest is published.
        if shutil.disk_usage(directory).free < size + (256 << 20):
            raise OSError("insufficient checkpoint disk space; previous generation retained")
        generation = f"step-{step}-{uuid.uuid4().hex}"
        target = directory / generation
        target.mkdir()
        rows = []
        for number, (path, array) in enumerate(leaves(tree)):
            row = dict(path=list(path), shape=list(array.shape), dtype=str(array.dtype), chunks=[])
            elements = max(1, self.chunk_bytes // array.dtype.itemsize)
            flat = array.reshape(-1)
            for offset in range(0, int(flat.size), elements):
                name = f"{generation}/{number:05d}-{offset:012d}.npy"
                file = directory / name
                # Gather only this chunk, never all parameters/moments at once.
                np.save(file, np.asarray(jax.device_get(flat[offset:offset + elements])), allow_pickle=False)
                row["chunks"].append(dict(file=name, sha256=digest(file), bytes=file.stat().st_size))
            rows.append(row)
        meta = dict(format=1, generation=generation, contract=contract, step=int(step), leaves=rows,
                    payload_bytes=sum(c["bytes"] for r in rows for c in r["chunks"]))
        write_json_atomic(directory / "checkpoint.json", meta)
        # Atomic pointer already names a complete local generation. Old remote
        # manifests still reference immutable remote chunks, so removing obsolete
        # local chunks does not invalidate either local or remote recovery.
        for old in directory.iterdir():
            if old.is_dir() and old.name.startswith("step-") and old.name != generation:
                resolved = old.resolve()
                if resolved.parent != directory.resolve() or old.is_symlink():
                    raise ValueError("unsafe obsolete generation")
                shutil.rmtree(resolved)
        return meta

    def restore(self, slot, contract, template, layouts=None):
        meta = self.metadata(slot, contract)
        if meta is None:
            return None
        expected = dict(leaves(serialization.to_state_dict(template)))
        rows = {tuple(row["path"]): row for row in meta["leaves"]}
        if rows.keys() != expected.keys():
            raise ValueError("checkpoint tree differs from template")
        # Retain empty optimizer subtrees (clip/decay EmptyState); leaf-only
        # reconstruction otherwise drops tuple entries and cannot resume AdamW.
        tree = serialization.to_state_dict(template)
        layout_dict = dict(leaves(serialization.to_state_dict(layouts))) if layouts is not None else None
        for path, reference in expected.items():
            row = rows[path]
            if list(reference.shape) != row["shape"] or str(reference.dtype) != row["dtype"]:
                raise ValueError("checkpoint shape/dtype mismatch")
            array = np.empty(int(reference.size), dtype=str(reference.dtype))
            cursor = 0
            for chunk in row["chunks"]:
                relative = Path(chunk["file"])
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != meta["generation"]:
                    raise ValueError("unsafe checkpoint chunk path")
                file = self.directory(slot) / relative
                if not file.exists() and self.hub is not None:
                    restore_artifact(file, f"{self.prefix}/{slot}/{chunk['file']}", self.hub)
                if digest(file) != chunk["sha256"]:
                    raise ValueError("checkpoint chunk SHA mismatch")
                piece = np.load(file, allow_pickle=False)
                if piece.dtype != array.dtype or cursor + piece.size > array.size:
                    raise ValueError("invalid checkpoint chunk")
                array[cursor:cursor + piece.size] = piece
                cursor += piece.size
            if cursor != array.size:
                raise ValueError("truncated checkpoint leaf")
            array = array.reshape(reference.shape)
            value = jax.device_put(array, layout_dict[path]) if layout_dict is not None else array
            parent = tree
            for key in path[:-1]:
                parent = parent.setdefault(key, {})
            parent[path[-1]] = value
        return serialization.from_state_dict(template, tree), meta

    def sync(self, slot, *, force=False, deadline=None):
        if self.hub is None:
            return False
        if not force and time.monotonic() < self.next_sync:
            return False
        directory = self.directory(slot)
        meta = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        receipt = directory / f"{meta['generation']}.uploaded.json"
        uploaded = set(json.loads(receipt.read_text())["files"]) if receipt.exists() else set()
        started = time.monotonic()
        try:
            groups, group, group_bytes = [], [], 0
            for row in meta["leaves"]:
                for chunk in row["chunks"]:
                    name = chunk["file"]
                    if name in uploaded:
                        continue
                    if digest(directory / name) != chunk["sha256"]:
                        raise ValueError("local upload chunk SHA mismatch")
                    if group and (len(group) >= 32 or group_bytes + chunk["bytes"] > (512 << 20)):
                        groups.append(group)
                        group, group_bytes = [], 0
                    group.append(name)
                    group_bytes += chunk["bytes"]
            if group:
                groups.append(group)
            for group in groups:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                upload_artifacts_together([(directory / name, f"{self.prefix}/{slot}/{name}") for name in group],
                                          self.hub, commit_message="EXP104 immutable checkpoint chunks")
                uploaded.update(group)
                write_json_atomic(receipt, {"files": sorted(uploaded)})
            # Remote cursor advances only after every immutable chunk exists.
            if deadline is not None and time.monotonic() >= deadline:
                return False
            upload_artifacts_together([(directory / "checkpoint.json", f"{self.prefix}/{slot}/checkpoint.json")],
                                      self.hub, commit_message="EXP104 durable checkpoint manifest")
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            self.sync_errors[slot] = dict(error_type=type(exc).__name__, http_status=status,
                                          retry_after_seconds=3600 if status == 429 else 120)
            if status in (401, 403) or isinstance(exc, ValueError):
                raise
            self.next_sync = time.monotonic() + (3600 if status == 429 else 120)
            return False
        write_json_atomic(directory / "synced.json", dict(generation=meta["generation"], step=meta["step"],
                          upload_seconds=time.monotonic() - started, payload_bytes=meta["payload_bytes"]))
        self.sync_errors.pop(slot, None)
        return True
