import json
import pytest

from scripts import m3q_downstream_campaign as campaign
from scripts import m3q_dual_domain_trust_campaign as base
from scripts import m3q_trust_region_sweep_campaign as sweep


def test_scientific_wrapper_builds_real_runtime_config_and_restores_globals(monkeypatch):
    original = sweep.STEPS_PER_LAYER
    monkeypatch.setattr(base, "main", lambda argv: {
        "replication": base.replication_overrides(0),
        "steps": sweep.STEPS_PER_LAYER, "windows": sweep.TRAIN_WINDOWS,
    })
    result = campaign.run_science([])
    assert result["steps"] == 512
    assert result["windows"] == 64
    assert result["replication"]["GROUP_PROPOSAL_MODE"] == {
        "LOCAL-PAIR-MSE": "joint_segment", "DOWNSTREAM-KL": "downstream_kl",
    }
    assert sweep.STEPS_PER_LAYER == original
    contract = campaign.configured_contract()
    assert contract["downstream_training_extension"]["train_length"] == 64
    configs = campaign.validate_runtime_configs()
    assert len(configs) == 4
    assert configs[0]["ARM_MIN_RELATIVE_GAIN"] == {
        "LOCAL-PAIR-MSE": 0.001, "DOWNSTREAM-KL": 0.001,
    }


def mock_operations(monkeypatch):
    monkeypatch.setattr(campaign, "artifact_config_from_env", lambda: object())
    monkeypatch.setattr(campaign, "upload_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign.jax, "clear_caches", lambda: None)


def test_session_continues_directly_after_preflight(monkeypatch, tmp_path):
    mock_operations(monkeypatch)
    calls = []
    monkeypatch.setattr(campaign, "run_preflight", lambda argv: calls.append("preflight") or {"status": "completed"})
    monkeypatch.setattr(campaign, "run_science", lambda argv: calls.append("science") or {"status": "deadline_partial", "aggregate": {}})
    result = campaign.main(["--output-dir", str(tmp_path), "--no-telegram"])
    assert calls == ["preflight", "science"]
    assert result["status"] == "deadline_partial"


def test_session_saves_startup_failure_without_starting_science(monkeypatch, tmp_path):
    mock_operations(monkeypatch)
    def fail(argv):
        raise RuntimeError("injected startup failure")
    monkeypatch.setattr(campaign, "run_preflight", fail)
    monkeypatch.setattr(campaign, "run_science", lambda argv: pytest.fail("must not train"))
    with pytest.raises(RuntimeError, match="injected"):
        campaign.main(["--output-dir", str(tmp_path), "--no-telegram"])
    saved = json.loads((tmp_path / "extent-m3q-downstream-campaign.json").read_text())
    assert saved["status"] == "failed"
    assert saved["stage"] == "preflight"
