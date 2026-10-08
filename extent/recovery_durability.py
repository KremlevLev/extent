"""Large-state upload batching and independent remote round-trip probes."""
import json
import time
import uuid
import numpy as np
from extent.chunked_checkpoint import ChunkedCheckpointStore, digest
from extent.campaign_checkpoint import write_json_atomic
from extent.hf_artifact_sync import restore_artifact, upload_artifacts_together


class RecoveryCheckpointStore(ChunkedCheckpointStore):
    def sync_until(self, slot, deadline):
        while time.monotonic() < deadline:
            if self.sync(slot, deadline=deadline):
                return True
            wait = max(1., self.next_sync-time.monotonic())
            if wait >= deadline-time.monotonic():
                return False
            time.sleep(min(wait,60))
        return False

    def sync(self, slot, *, force=False, deadline=None):
        if self.hub is None or time.monotonic() < self.next_sync:
            return False
        directory = self.directory(slot)
        meta = json.loads((directory / "checkpoint.json").read_text())
        receipt = directory / f"{meta['generation']}.uploaded.json"
        uploaded = set(json.loads(receipt.read_text())["files"]) if receipt.exists() else set()
        started = time.monotonic()
        try:
            groups, group, size = [], [], 0
            for row in meta["leaves"]:
                for chunk in row["chunks"]:
                    name = chunk["file"]
                    if name in uploaded:
                        continue
                    if digest(directory / name) != chunk["sha256"]:
                        raise ValueError("local upload chunk SHA mismatch")
                    if group and (len(group) >= 128 or size + chunk["bytes"] > (2 << 30)):
                        groups.append(group)
                        group, size = [], 0
                    group.append(name)
                    size += chunk["bytes"]
            if group:
                groups.append(group)
            for group in groups:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                upload_artifacts_together([(directory / name, f"{self.prefix}/{slot}/{name}") for name in group],
                    self.hub, commit_message="Extent recovery checkpoint chunks")
                uploaded.update(group)
                write_json_atomic(receipt, dict(files=sorted(uploaded)))
            if deadline is not None and time.monotonic() >= deadline:
                return False
            upload_artifacts_together([(directory / "checkpoint.json", f"{self.prefix}/{slot}/checkpoint.json")],
                self.hub, commit_message="Extent recovery durable manifest")
            # Read the published manifest and one representative large chunk
            # into a separate directory, without consulting local generations.
            probe = self.root / "remote-verification" / uuid.uuid4().hex
            pointer = probe / "checkpoint.json"
            if not restore_artifact(pointer, f"{self.prefix}/{slot}/checkpoint.json", self.hub):
                raise RuntimeError("remote checkpoint manifest unavailable")
            remote = json.loads(pointer.read_text())
            if remote != meta:
                raise ValueError("remote checkpoint manifest differs")
            chunk = max((c for r in meta["leaves"] for c in r["chunks"]), key=lambda c:c["bytes"])
            sample = probe / "sample.npy"
            if not restore_artifact(sample, f"{self.prefix}/{slot}/{chunk['file']}", self.hub):
                raise RuntimeError("remote checkpoint sample unavailable")
            if digest(sample) != chunk["sha256"]:
                raise ValueError("remote checkpoint sample SHA mismatch")
            sample.unlink()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            delay = 3600 if status == 429 else 120
            self.sync_errors[slot] = dict(error_type=type(exc).__name__, http_status=status,
                                          retry_after_seconds=delay)
            if status in (401, 403) or isinstance(exc, ValueError):
                raise
            self.next_sync = time.monotonic() + delay
            return False
        write_json_atomic(directory / "synced.json", dict(generation=meta["generation"], step=meta["step"],
            upload_seconds=time.monotonic()-started, payload_bytes=meta["payload_bytes"],
            remote_manifest_and_sample_verified=True))
        self.sync_errors.pop(slot, None)
        return True


def cloud_roundtrip(root, prefix, hub, deadline):
    """Before model allocation: complete small binary save/upload/fresh restore."""
    store = RecoveryCheckpointStore(root / "probe-upload", prefix + "/probe", hub, chunk_bytes=4096)
    slot = uuid.uuid4().hex
    contract = dict(kind="remote-roundtrip-v1", nonce=slot)
    payload = dict(parameters=np.arange(1024, dtype=np.float32), optimizer=dict(count=np.asarray(7, np.int32)))
    store.save(slot, contract, payload, 7)
    if not store.sync_until(slot, min(deadline,time.monotonic()+900)):
        raise RuntimeError("HF binary roundtrip upload failed; training not started")
    reader = RecoveryCheckpointStore(root / "probe-download", prefix + "/probe", hub)
    restored = reader.restore(slot, contract, payload)
    if restored is None or restored[1]["step"] != 7:
        raise RuntimeError("HF binary roundtrip restore failed")
    for name in payload:
        import jax
        for actual, expected in zip(jax.tree.leaves(restored[0][name]), jax.tree.leaves(payload[name])):
            np.testing.assert_array_equal(actual, expected)
    return dict(verified=True, scope="complete small binary payload; large checkpoints verify manifest and sample")
