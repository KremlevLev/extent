import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from extent.torch_continue_176 import ExactRMSNorm, destinations, save_current

@pytest.mark.parametrize("length", [1, 2048, 310771712, 151936 * 2048])
def test_precision_sample_indices_stay_inside_large_parameters(length):
    from extent.torch_continue_176 import sample_indices
    indices = sample_indices(length)
    assert indices.dtype == torch.int64
    assert int(indices.min()) == 0
    assert int(indices.max()) == length - 1
    assert bool((indices[1:] >= indices[:-1]).all())

def test_restart_excludes_journal_steps_not_in_restored_checkpoint():
    from engine_torch176 import history_at_checkpoint
    previous = dict(step=700, history=[dict(step=640), dict(step=650), dict(step=700)],
                    evaluations=[dict(step=640, nll=2.53), dict(step=700, nll=2.52)])
    restored = history_at_checkpoint(previous, 650)
    assert restored["excluded_updates"] == 50
    assert restored["history"] == [dict(step=640), dict(step=650)]
    assert restored["evaluations"] == [dict(step=640, nll=2.53)]


def test_diagnostic_only_contract_migration_rejects_training_changes():
    from engine_torch176 import compatible_contract
    expected = dict(lr=1e-6, code={"extent/torch_continue_176.py": "new", "other": "same"})
    old = dict(expected, code=dict(expected["code"]))
    old["code"]["extent/torch_continue_176.py"] = "624a9a35a5b4cac433d37b7ebf1e4178671f031683cfb770b81363b4619074da"
    assert compatible_contract(old, expected)
    assert not compatible_contract(dict(old, lr=1e-5), expected)
    old["code"]["other"] = "changed"
    assert not compatible_contract(old, expected)


def test_mapping_keeps_only_active_modules_and_transposes_kernels():
    names = list(destinations(["qwen", "0", "self_attn", "q_proj", "kernel"]))
    assert [row[1] for row in names] == [3, 7, 11, 15, 19, 23]
    assert all(row[2] for row in names)
    norms = list(destinations(["qwen", "0", "input_layernorm", "scale"]))
    assert len(norms) == 27 and not any(row[2] for row in norms)
    assert len(list(destinations(["mamba", "in_proj", "kernel"]))) == 21
    assert list(destinations(["qwen", "2"])) == [("qwen.embed_tokens.weight", None, False)]


def test_norm_preserves_bf16_boundary_with_fp32_master_weight():
    from types import SimpleNamespace
    norm = ExactRMSNorm(SimpleNamespace(weight=torch.nn.Parameter(torch.ones(8)), variance_epsilon=1e-6))
    assert norm(torch.ones(2, 8, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_host_bf16_storage_bits_distinguish_sub_ulp_changes():
    from extent.torch_continue_176 import host_precision_changes
    before = {"ones": torch.ones(8)}
    after = {"ones": before["ones"] + 1e-6}
    metrics = host_precision_changes(before, after)["ones"]
    assert metrics["fp32_changed_fraction"] == 1.
    assert metrics["bf16_changed_fraction"] == 0.


def test_atomic_current_descriptor_matches_saved_bytes(tmp_path):
    from extent.broad_corpus import digest_file
    payload = dict(step=512, cursor=92274688, contract={"experiment": 176}, model={"x": torch.ones(2)})
    file = save_current(tmp_path, payload)
    metadata = json.loads((tmp_path / "CURRENT.json").read_text())
    assert metadata["sha256"] == digest_file(file)
    assert metadata["step"] == 512
    assert not (tmp_path / "CURRENT.next.pt").exists()
    assert torch.load(file, weights_only=True)["cursor"] == 92274688


def virtual_bridge():
    import runpy
    import tempfile
    import jax
    import jax.numpy as jnp
    from transformers import Qwen3Config
    from extent.chunked_checkpoint import ChunkedCheckpointStore
    from extent.clean_tpu_171 import optimizer as jax_optimizer
    from extent.torch_continue_176 import FullStudent, convert_source

    fixture = runpy.run_path(str(Path(__file__).with_name("test_tpu174.py")))["virtual_full_fixture"]
    _, _, parameters, _, _, _, _, _ = fixture()
    state = jax_optimizer().init(parameters)
    adam = state[1]._replace(
        count=jnp.asarray(511, jnp.int32),
        mu=jax.tree.map(lambda p: jnp.full_like(p, .001), parameters),
        nu=jax.tree.map(lambda p: jnp.full_like(p, .01), parameters),
    )
    state = (state[0], adam)
    config = Qwen3Config(vocab_size=24, hidden_size=8, intermediate_size=16, num_hidden_layers=28,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                        rms_norm_eps=1e-6, tie_word_embeddings=True)
    with tempfile.TemporaryDirectory() as directory:
        store = ChunkedCheckpointStore(Path(directory), "test176")
        metadata = store.save("source", {}, dict(parameters=parameters, optimizer=state,
                                                cursor=np.asarray(92258304, np.int64)), 511)
        model = FullStudent(config, backend="reference")
        opt, cursor = convert_source(model, metadata, Path(directory) / "source")
        assert cursor == 92258304
        np.testing.assert_array_equal(model.qwen.embed_tokens.weight.detach(), np.asarray(parameters["qwen"][2]))
        np.testing.assert_array_equal(
            model.mixers["0"].in_weight.detach(), np.asarray(parameters["mamba"]["in_proj"]["kernel"][0]).T,
        )
        for parameter in model.parameters():
            assert int(opt.state[parameter]["step"]) == 511
            torch.testing.assert_close(opt.state[parameter]["exp_avg"], torch.full_like(parameter, .001))
            torch.testing.assert_close(opt.state[parameter]["exp_avg_sq"], torch.full_like(parameter, .01))
        selected = model.qwen.embed_tokens.weight
        previous = selected.detach().clone()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .02)
        opt.step()
        mu = .9 * .001 + .1 * .02
        nu = .95 * .01 + .05 * .02**2
        delta = 1e-6 * (mu / (1 - .9**512)) / (np.sqrt(nu / (1 - .95**512)) + 1e-8)
        torch.testing.assert_close(selected, previous - delta, rtol=0, atol=2e-9)
        assert int(opt.state[selected]["step"]) == 512
        # Activation recomputation must bind each concrete layer, not a changing loop index.
        opt.zero_grad(set_to_none=True)
        loss = model(torch.arange(64)[None] % 24)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.mixers["0"].in_weight.grad is not None
        assert model.qwen.layers[3].self_attn.q_proj.weight.grad is not None
        assert model.qwen.layers[0].mlp.up_proj.weight.grad is not None


def test_real_jax_checkpoint_bridge_and_adam_continuity():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=8")
    script = f'import runpy; runpy.run_path({str(Path(__file__).resolve())!r})["virtual_bridge"]()'
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def owned_workspace(tmp_path, monkeypatch):
    import extent.weekend_storage_176 as storage
    base = tmp_path / "ram"
    base.mkdir()
    root = base / "extent176-test"
    root.mkdir()
    uid = root.stat().st_uid
    monkeypatch.setattr(storage, "RAM_BASE", base)
    monkeypatch.setattr(storage, "host_available", lambda: 400 * 2**30)
    monkeypatch.setattr(storage.os, "getuid", lambda: uid, raising=False)
    (root / "owner.json").write_text(json.dumps(dict(uid=uid, path=str(root.resolve()), experiment=176)))
    return storage, root


def test_upload_hardlink_preserves_snapshot_while_current_advances(tmp_path, monkeypatch):
    storage, root = owned_workspace(tmp_path, monkeypatch)
    first = dict(step=512, cursor=92274688, contract={}, model={"x": torch.ones(2)})
    save_current(root, first)
    snapshot, meta = storage.upload_snapshot(root)
    save_current(root, dict(first, step=513, model={"x": torch.zeros(2)}))
    assert torch.load(snapshot / "CURRENT.pt", weights_only=True)["step"] == 512
    assert torch.load(root / "CURRENT.pt", weights_only=True)["step"] == 513
    with pytest.raises(ValueError):
        storage.cleanup_after_verified_sync(root, tmp_path, meta["sha256"])
    assert root.exists()


def test_cleanup_requires_matching_latest_remote_sha_and_owned_root(tmp_path, monkeypatch):
    storage, root = owned_workspace(tmp_path, monkeypatch)
    save_current(root, dict(step=512, cursor=92274688, contract={}, model={"x": torch.ones(2)}))
    meta = json.loads((root / "CURRENT.json").read_text())
    storage.cleanup_after_verified_sync(root, tmp_path, meta["sha256"])
    assert not root.exists()
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError):
        storage.cleanup_after_verified_sync(other, tmp_path, meta["sha256"])
    assert other.exists()


@pytest.mark.parametrize("status", ["deadline_partial", "fast_paused", "failed"])
def test_launcher_cleanup_only_after_gpu_exit_and_remote_verification(tmp_path, monkeypatch, status):
    import engine_torch176 as launch
    import engine_torch170
    import extent.hf_download_retry
    import extent.isolated_tpu_planner
    import extent.notifications
    storage, root = owned_workspace(tmp_path, monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    (work / "ram-workspace.json").write_text(json.dumps({"path": str(root)}))
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(storage, "create_ram_workspace", lambda work: root)
    monkeypatch.setattr(storage, "disk_guard", lambda *args: 80 * 2**30)
    monkeypatch.setattr(launch.signal, "signal", lambda *args: None)
    monkeypatch.setattr(extent.hf_download_retry, "install_download_retries", lambda: None)
    monkeypatch.setattr(extent.notifications, "send_telegram_message", lambda *args: None)

    def prepare(root, work, policy):
        save_current(root, dict(step=511, cursor=92258304, contract={}, model={"x": torch.ones(2)}))

    monkeypatch.setattr(launch, "prepare", prepare)
    monkeypatch.setattr(launch, "transport_probe", lambda *args: {"verified": True})
    def probe(command, result, log, timeout, **options):
        Path(log).write_text("probe log")
        Path(result).write_text("{}")
        return dict(status="measured", report={"input_tokens_per_second": 2000.})
    monkeypatch.setattr(extent.isolated_tpu_planner, "run_probe", probe)
    uploads = []

    class GPUChild:
        def __init__(self, command, **kwargs):
            target = Path(command[command.index("--result") + 1])
            save_current(root, dict(step=512, cursor=92274688, contract={}, model={"x": torch.zeros(2)}))
            target.write_text(json.dumps(dict(
                status=status, local_checkpoint_step=512,
                branches={"FULL-CE": dict(step=512)},
            )))
            self.returncode = 1 if status == "failed" else 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(launch.subprocess, "Popen", GPUChild)
    def sync(snapshot, report, policy, **options):
        from extent.broad_corpus import digest_file
        uploads.append(options["checkpoint_step"])
        return dict(sha256=digest_file(snapshot / "CURRENT.pt"), commit="verified-test")
    monkeypatch.setattr(engine_torch170, "sync_remote", sync)
    command = ["--work-dir", str(work), "--max-wall-hours", "2"]
    if status == "failed":
        with pytest.raises(RuntimeError):
            launch.main(command)
    else:
        launch.main(command)
    if status == "deadline_partial":
        assert uploads == [512]
        assert not root.exists()
        result = json.loads((work / "results/runtime-torch176.json").read_text())
        assert result["hf_verified"] is True and result["ram_cleanup_verified"] is True
    else:
        assert uploads == []
        assert (root / "CURRENT.pt").exists()


def test_status_labels_new_validation_and_effective_speed():
    from engine_status import format_report
    report = dict(contract=dict(experiment=176, format="torch-full-ce-v1"), status="running",
        branches={"FULL-CE": dict(step=600, source_step=511, accepted_input_tokens=1458176,
            history=[dict(ce=2.5, loss=2.5, input_tokens_per_second=1800.)],
            last_validation_nll=2.539, validation_step=590, validation_windows=128,
            loop_tokens_per_second=1650.)})
    text = format_report("torch", report, 0, True, now=0)
    assert "2.53900" in text and "1800 ток/с" in text and "1650 ток/с" in text


def test_runtime_has_no_jax_dependency_and_requires_disk_headroom(tmp_path, monkeypatch):
    import extent.weekend_storage_176 as storage
    from types import SimpleNamespace
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=29 * 2**30))
    with pytest.raises(OSError):
        storage.disk_guard(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", "import sys, engine_torch176, extent.torch_continue_176; assert 'jax' not in sys.modules"],
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
