"""Owned tmpfs workspace, SSD headroom and immutable upload snapshots."""
import json
import os
from pathlib import Path
import shutil
import uuid

from extent.broad_corpus import digest_file, write_json

RAM_BASE = Path("/dev/shm")

def host_available():
    return int(next(row.split()[1] for row in Path("/proc/meminfo").read_text().splitlines()
                    if row.startswith("MemAvailable:"))) * 1024


def disk_guard(work_dir, minimum_gib=30):
    free = shutil.disk_usage(work_dir).free
    if free < minimum_gib * 2**30:
        raise OSError("SSD free space fell below the reserved headroom")
    return free


def create_ram_workspace(work_dir, minimum_available_gib=160):
    work_dir = Path(work_dir).resolve()
    pointer = work_dir / "ram-workspace.json"
    if pointer.exists():
        root = Path(json.loads(pointer.read_text())["path"])
        verify_owned(root)
        if host_available() < 80 * 2**30:
            raise MemoryError("Need80GiB available host RAM for restart")
        return root
    available = host_available()
    base = RAM_BASE.resolve()
    if available < minimum_available_gib * 2**30 or shutil.disk_usage(base).free < 120 * 2**30:
        raise MemoryError("Need160GiB available host RAM and120GiB free tmpfs")
    root = base / ("extent176-" + uuid.uuid4().hex)
    root.mkdir(mode=0o700)
    write_json(root / "owner.json", dict(uid=os.getuid(), path=str(root), experiment=176))
    write_json(pointer, dict(path=str(root)))
    return root


def verify_owned(root):
    root = Path(root)
    if root.is_symlink() or root.resolve().parent != RAM_BASE.resolve() or not root.name.startswith("extent176-"):
        raise ValueError("Unknown RAM workspace")
    marker = json.loads((root / "owner.json").read_text())
    if marker != dict(uid=os.getuid(), path=str(root.resolve()), experiment=176):
        raise ValueError("RAM workspace ownership differs")
    if root.stat().st_uid != os.getuid():
        raise ValueError("RAM workspace belongs to another user")
    return root.resolve()


def check_ram_growth(root, maximum_gib=120):
    root = verify_owned(root)
    if host_available() < 48 * 2**30:
        raise MemoryError("Host RAM headroom fell below48GiB")
    # Count hard-linked checkpoint snapshots only once.
    seen, used = set(), 0
    for file in root.rglob("*"):
        if file.is_symlink():
            raise ValueError("Symlink in owned workspace")
        if file.is_file():
            try:
                stat = file.stat()
            except FileNotFoundError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                used += stat.st_size
                seen.add(key)
    if used > maximum_gib * 2**30:
        raise OSError("Owned RAM files exceed configured budget")
    return used


def upload_snapshot(root):
    from filelock import FileLock
    root = verify_owned(root)
    target = root / "upload"
    if target.is_symlink():
        raise ValueError("Unsafe upload staging")
    target.mkdir(exist_ok=True)
    with FileLock(str(root / "current.lock")):
        metadata = json.loads((root / "CURRENT.json").read_text())
        current = root / "CURRENT.pt"
        if digest_file(current) != metadata["sha256"]:
            raise ValueError("Local CURRENT descriptor mismatch")
        snapshot = target / "CURRENT.pt"
        snapshot.unlink(missing_ok=True)
        os.link(current, snapshot)
        write_json(target / "CURRENT.json", metadata)
    return target, metadata


def cleanup_after_verified_sync(root, work_dir, remote_sha):
    from filelock import FileLock
    root = verify_owned(root)
    with FileLock(str(root / "current.lock")):
        local = json.loads((root / "CURRENT.json").read_text())
        if local["sha256"] != remote_sha or digest_file(root / "CURRENT.pt") != remote_sha:
            raise ValueError("Latest local CURRENT has not been verified remotely")
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("Symlink refused during cleanup")
    # Caller must have observed that its GPU child exited.
    shutil.rmtree(root)
    (Path(work_dir) / "ram-workspace.json").unlink(missing_ok=True)
