"""Small mocked-cloud workflow; numerical modules run on real local JAX CPU."""
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from extent import HybridForCausalLM, tiny_config
from extent.campaign_checkpoint import CampaignCheckpointStore
from extent.sharding import create_v5e_mesh
from scripts import m3q_subspace_engine as engine


@pytest.mark.parametrize("mode", ["normal", "interruption", "rate-limit", "diagnostics"])
def test_cloud_campaign_saves_binary_states_and_completed_rerun_skips_model(tmp_path, monkeypatch, mode):
    config = tiny_config()
    real_devices = jax.devices()
    mesh = create_v5e_mesh(real_devices)
    spec = engine.Campaign(994, "unit-workflow", (
        engine.Arm("OUT", "OUT-LORA", 2), engine.Arm("INOUT", "INOUT-LORA", 2)), "INOUT", "OUT")
    monkeypatch.setattr(engine, "HORIZONS", (1, 2, 3))
    monkeypatch.setattr(engine, "SEEDS", (123,))
    monkeypatch.setattr(engine, "LENGTH", 4)
    monkeypatch.setattr(engine, "TRAIN_WINDOWS", 3)
    monkeypatch.setattr(engine, "DATA", (("train", "train", 0, 3),
        ("validation", "validation", 0, 1), ("locked_test", "test", 0, 1)))
    monkeypatch.setattr(engine, "load_config", lambda _: (config, {}))
    monkeypatch.setattr(engine, "configured_contract", lambda _: {"replacement_order": [0, 2], "source_contract": {}})
    monkeypatch.setattr(engine, "artifact_config_from_env", lambda: object())
    monkeypatch.setattr(engine, "restore_artifact", lambda *args: False)
    monkeypatch.setattr(engine, "_safe_notify", lambda *args: None)
    monkeypatch.setattr(engine.exp091, "preflight_source_endpoints", lambda: None)
    monkeypatch.setattr(engine.jax, "devices", lambda: [SimpleNamespace(platform="tpu") for _ in range(8)])
    monkeypatch.setattr(engine, "create_v5e_mesh", lambda _: mesh)
    monkeypatch.setattr(engine, "load_wikitext2_tokens", lambda count, *args, **kwargs: np.arange(count, dtype=np.int32) % 128)
    monkeypatch.setattr(engine, "teacher_config_from_spec", lambda *args, **kwargs: config)
    monkeypatch.setattr(engine, "Qwen3ForCausalLM", HybridForCausalLM)
    monkeypatch.setattr(engine, "_ensure_checkpoint", lambda _: None)
    monkeypatch.setattr(engine, "stream_teacher_qwen_weights", lambda params, *args: (params, None))
    monkeypatch.setattr(engine, "compose_parameters", lambda teacher, *args: teacher)
    monkeypatch.setattr(engine.sequential, "endpoint_contract", lambda *args: {})

    class MockSourceStore(CampaignCheckpointStore):
        def metadata(self, slot, contract):
            if self.prefix.startswith(spec.prefix):
                if not (self.root / slot / "checkpoint.json").exists():
                    return None  # No pre-existing mocked remote adapter.
                return super().metadata(slot, contract)
            return {"checkpoint_sha256": "source-sha"}

        def restore(self, slot, contract, template=None):
            if self.prefix.startswith(spec.prefix):
                return super().restore(slot, contract, template)
            return {"params": {}}, {"checkpoint_sha256": "source-sha"}

    monkeypatch.setattr(engine, "CampaignCheckpointStore", MockSourceStore)
    uploads = []
    def upload(files, *args, **kwargs):
        uploads.append([destination for _, destination in files])
        if mode == "rate-limit" and len(uploads) == 1:
            error = RuntimeError("injected Hub 429")
            error.response = SimpleNamespace(status_code=429)
            raise error
        for source, _ in files:
            assert source.is_file()
    monkeypatch.setattr(engine, "upload_artifacts_together", upload)
    if mode == "interruption":
        make_step = engine.make_train_step
        crashed = []
        def interrupt_once(*args):
            real_step = make_step(*args)
            def step(*values):
                if int(values[-1]) == 2 and not crashed:
                    crashed.append(True)
                    raise RuntimeError("injected session interruption")
                return real_step(*values)
            return step
        monkeypatch.setattr(engine, "make_train_step", interrupt_once)
    args = ["--output-dir", str(tmp_path / "output"), "--state-dir", str(tmp_path / "state"), "--no-telegram"]
    options = {}
    if mode == "diagnostics":
        def factory(*values):
            actual = engine.make_train_step(*values)
            def run(*inputs):
                proposal = actual(*inputs)
                return (*proposal, {"forward_finite": np.bool_(True),
                    "gradients_finite": np.bool_(True), "norm_only_overflow": np.bool_(True),
                    "log10_grad_norm": np.float32(1.0), "naive_norm": np.float32(np.inf)})
            return run
        options = {"step_factory": factory, "contract_extra": {"test_diagnostics": True,
                   "evaluate_teacher_baseline": True}}
    if mode == "interruption":
        with pytest.raises(RuntimeError, match="injected session interruption"):
            engine.run_campaign(spec, args)
        partial = json.loads((tmp_path / "output" / f"{spec.stem}.json").read_text())
        assert partial["status"] == "failed"
        assert all(row["step"] == 1 for row in partial["branches"]["123"].values())
    result = engine.run_campaign(spec, args, **options)
    assert result["status"] == "completed"
    assert not result["remote_sync_pending"]
    assert all(row["step"] == 3 and row["complete"] for row in result["branches"]["123"].values())
    assert any(path.endswith("state.msgpack") for batch in uploads for path in batch)
    assert all(path.startswith(spec.prefix) for batch in uploads for path in batch)
    local = json.loads((tmp_path / "output" / f"{spec.stem}.json").read_text())
    assert "optimizer" not in local["branches"]["123"]["OUT"]
    if mode == "diagnostics":
        assert np.isfinite(local["original_teacher_test_nll"])
        assert local["branches"]["123"]["OUT"]["latest_health"]["naive_norm"] is None
        event = local["branches"]["123"]["OUT"]["first_norm_only_overflow"]
        assert event["step"] == 1
        meta = json.loads((tmp_path / "state" / "adapters" / event["checkpoint_slot"] / "checkpoint.json").read_text())
        assert meta["step"] == 0  # The exact state BEFORE the overflow-producing proposal.
    monkeypatch.setattr(engine, "initialize_sharded_parameters", lambda *args: (_ for _ in ()).throw(AssertionError("unexpected allocation")))
    assert engine.run_campaign(spec, args, **options)["status"] == "completed"
