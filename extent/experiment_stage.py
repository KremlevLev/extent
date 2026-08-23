from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def update_stage_manifest(
    path: str | Path,
    *,
    experiment: str,
    stage: str,
    status: str,
    details: dict | None = None,
) -> dict:
    """Atomically record durable progress without treating it as model quality."""
    if status not in {"running", "completed", "failed"}:
        raise ValueError("invalid experiment stage status")
    target = Path(path)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("experiment") != experiment:
            raise ValueError("stage manifest belongs to another experiment")
    else:
        payload = {
            "format_version": 1,
            "experiment": experiment,
            "stages": {},
        }
    record = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        record["details"] = details
    payload["stages"][stage] = record
    payload["current_stage"] = stage
    if stage == "final" and status == "completed":
        payload["status"] = "completed"
    elif status == "failed":
        payload["status"] = "failed"
    else:
        payload["status"] = "running"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)
    return payload
