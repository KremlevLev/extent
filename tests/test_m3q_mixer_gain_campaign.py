import json
from types import SimpleNamespace

import jax.numpy as jnp
import optax
import pytest

from extent.campaign_checkpoint import write_json_atomic

from scripts import m3q_mixer_gain_campaign as campaign


def test_gate_needs_both_seeds_and_unchanged_start_improvement():
    result = {"branches": {
        "123": {
            "GLOBAL": {"complete": True, "locked_test_nll": 8.0, "start_test_nll": 9.0},
            "PER-LAYER": {"complete": True, "locked_test_nll": 7.9},
        },
        "456": {
            "GLOBAL": {"complete": True, "locked_test_nll": 8.0, "start_test_nll": 9.0},
            "PER-LAYER": {"complete": True, "locked_test_nll": 7.9},
        },
    }}
    assert campaign.aggregate(result)["scientific_gate_passed"]
    result["branches"]["456"]["PER-LAYER"]["locked_test_nll"] = 7.97
    assert not campaign.aggregate(result)["scientific_gate_passed"]


def test_tiny_optimizer_state_survives_json_roundtrip():
    raw = jnp.zeros((3,), dtype=jnp.float32)
    tx = optax.adam(0.01)
    state = tx.init(raw)
    _, state = tx.update(jnp.ones_like(raw), state, raw)
    saved = campaign._optimizer_record(state)
    restored = campaign._restore_optimizer(tx, raw, saved)
    assert campaign._optimizer_record(restored) == saved


def test_rate_limit_recognition_does_not_mask_other_errors():
    assert campaign._is_hub_rate_limit(
        SimpleNamespace(response=SimpleNamespace(status_code=429))
    )
    assert not campaign._is_hub_rate_limit(
        SimpleNamespace(response=SimpleNamespace(status_code=403))
    )


def test_sync_only_uploads_local_result_without_tpu(tmp_path, monkeypatch):
    path = tmp_path / "extent-m3q-mixer-gains.json"
    write_json_atomic(path, {
        "status": "failed", "branches": {}, "remote_sync_pending": True,
        "remote_sync_error": "429",
    })
    monkeypatch.setattr(campaign, "artifact_config_from_env", lambda: object())
    uploads = []
    monkeypatch.setattr(campaign, "_upload_result_pair",
                        lambda *args: uploads.append(args))
    result = campaign.main(["--output-dir", str(tmp_path), "--sync-only"])
    assert len(uploads) == 1
    assert "remote_sync_pending" not in result


def test_sync_only_keeps_pending_flag_if_upload_fails(tmp_path, monkeypatch):
    path = tmp_path / "extent-m3q-mixer-gains.json"
    write_json_atomic(path, {
        "status": "failed", "branches": {}, "remote_sync_pending": True,
        "remote_sync_error": "429",
    })
    monkeypatch.setattr(campaign, "artifact_config_from_env", lambda: object())

    def fail(*args):
        raise RuntimeError("still limited")

    monkeypatch.setattr(campaign, "_upload_result_pair", fail)
    with pytest.raises(RuntimeError, match="still limited"):
        campaign.main(["--output-dir", str(tmp_path), "--sync-only"])
    assert json.loads(path.read_text(encoding="utf-8"))["remote_sync_pending"]


def test_original_code_revision_can_resume_only_for_registered_hotfix():
    prior = next(iter(campaign.ALLOWED_PREVIOUS_REVISIONS))
    result = {"git_revision": prior}
    campaign._record_resume_revision(result, "new-revision")
    assert result["code_revisions"] == [prior, "new-revision"]
    with pytest.raises(ValueError, match="code revision changed"):
        campaign._record_resume_revision({"git_revision": "unexpected"}, "new-revision")
