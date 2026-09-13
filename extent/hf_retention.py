"""Classify large Hugging Face campaign checkpoints for safe retention."""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath


DEFAULT_PROTECTED_PREFIXES = (
    "experiments/exp069-allocation/",
    "experiments/exp072-full-depth-sequential-confirmation-v2/",
)


def normalize_prefix(value: str) -> str:
    value = value.strip().strip("/")
    if not value:
        raise ValueError("retention prefix cannot be empty")
    return value + "/"


def is_checkpoint_payload(path: str) -> bool:
    return PurePosixPath(path).name == "state.msgpack"


def is_protected(path: str, prefixes=DEFAULT_PROTECTED_PREFIXES) -> bool:
    normalized = path.strip("/")
    return any(normalized.startswith(normalize_prefix(prefix)) for prefix in prefixes)


def classify_repo_files(files, protected_prefixes=DEFAULT_PROTECTED_PREFIXES):
    """Return byte/count totals without downloading repository contents."""
    rows = []
    known_paths = set()
    for item in files:
        path = str(getattr(item, "path", "")).strip("/")
        if not path:
            continue
        size = int(getattr(item, "size", 0) or 0)
        known_paths.add(path)
        rows.append((path, size))

    candidates, protected = [], []
    by_experiment = defaultdict(lambda: {"files": 0, "bytes": 0})
    for path, size in rows:
        if not is_checkpoint_payload(path):
            continue
        if is_protected(path, protected_prefixes):
            protected.append({"path": path, "bytes": size})
            continue
        candidates.append({"path": path, "bytes": size})
        parts = PurePosixPath(path).parts
        experiment = "/".join(parts[:2]) if len(parts) >= 2 else path
        by_experiment[experiment]["files"] += 1
        by_experiment[experiment]["bytes"] += size

    review_paths = []
    for row in candidates:
        review_paths.append(row["path"])
        metadata = str(PurePosixPath(row["path"]).with_name("checkpoint.json"))
        if metadata in known_paths:
            review_paths.append(metadata)

    return {
        "repository_files": len(rows),
        "repository_bytes": sum(size for _, size in rows),
        "checkpoint_payload_files": len(candidates) + len(protected),
        "reviewable_payload_files": len(candidates),
        "reviewable_payload_bytes": sum(row["bytes"] for row in candidates),
        "protected_payload_files": len(protected),
        "protected_payload_bytes": sum(row["bytes"] for row in protected),
        "protected_prefixes": [normalize_prefix(value) for value in protected_prefixes],
        "review_paths": sorted(set(review_paths)),
        "reviewable_by_experiment": {
            key: value for key, value in sorted(by_experiment.items())
        },
    }
