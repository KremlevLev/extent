import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from extent.training_diagnostic_175 import (
    WEIGHTS, make_objective, norm_and_cosine, plan_matched_steps, select_winner,
)


def test_planner_leaves_overhead_and_reserve_and_never_exceeds_matched_cap():
    assert plan_matched_steps(6600, 5.2) == 224
    assert plan_matched_steps(7200, 5.2) == 256
    assert plan_matched_steps(1500, 5.2) == 0
    with pytest.raises(ValueError):
        plan_matched_steps(7200, float("nan"))


def test_selection_requires_identical_completed_token_budgets_but_no_nll_gate():
    def row(nll, step=256, status="pilot_completed", returncode=0):
        return dict(returncode=returncode, report=dict(
            status=status, step=step, final_validation_nll=nll,
        ))
    rows = [row(2.7), row(2.8), row(2.1, step=128), row(0.1, returncode=1), row(float("nan"))]
    assert select_winner(rows, 256) is rows[0]


def test_norm_cosine_stays_finite_for_huge_or_zero_gradients():
    a = dict(kernel=jnp.asarray([1e25, -1e25]))
    aligned = norm_and_cosine(a, a)
    opposed = norm_and_cosine(a, jax.tree.map(lambda x: -x, a))
    assert np.isfinite(aligned["left_log10_norm"])
    np.testing.assert_allclose(aligned["cosine"], 1., atol=1e-6)
    np.testing.assert_allclose(opposed["cosine"], -1., atol=1e-6)
    assert not norm_and_cosine(a, dict(kernel=jnp.zeros(2)))["cosine_defined"]


def virtual_weighted_update():
    import runpy
    import optax
    from extent.clean_tpu_171 import layout, optimizer
    from extent.end_to_end_174 import make_gradient
    from extent.training_diagnostic_175 import gradient_diagnostics, update_diagnostics

    fixture_file = str(Path(__file__).with_name("test_tpu174.py"))
    c, tc, p, donor, tokens, param_layout, mesh, rep = runpy.run_path(fixture_file)["virtual_full_fixture"]()
    objective = make_objective(c, tc)
    fn = jax.jit(jax.value_and_grad(objective, has_aux=True), out_shardings=((rep, rep), param_layout))

    def weights(mode):
        return jax.device_put(np.asarray(WEIGHTS[mode], np.float32), rep)

    (mixed_loss, metrics), mixed = fn(p, donor, tokens, weights("mixed"))
    (ce_loss, _), ce = fn(p, donor, tokens, weights("ce"))
    (kl_loss, _), kl = fn(p, donor, tokens, weights("kl"))
    np.testing.assert_allclose(mixed_loss, .5 * (ce_loss + kl_loss), atol=1e-6)
    # Same computation and teacher stop as the physically exercised EXP174 path.
    previous = jax.jit(make_gradient(c, tc, True), out_shardings=(rep, param_layout, rep))
    reference_loss, reference_grad, reference_metrics = previous(p, donor, tokens)
    np.testing.assert_allclose(mixed_loss, reference_loss, atol=1e-6)
    np.testing.assert_allclose(metrics, reference_metrics, atol=1e-6)
    for actual, expected in zip(jax.tree.leaves(mixed), jax.tree.leaves(reference_grad)):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-3)
    summaries = jax.jit(gradient_diagnostics, out_shardings=rep)(ce, kl)
    assert bool(summaries["all"]["cosine_defined"])
    assert np.isfinite(summaries["all"]["cosine"])

    tx = optimizer()
    state = tx.init(p)
    state_layout = layout(state, mesh, stacked=False)
    state = jax.device_put(state, state_layout)

    def update(p, state, grad):
        change, new_state = tx.update(grad, state, p)
        return optax.apply_updates(p, jax.tree.map(lambda x: -1e-6 * x, change)), new_state

    new, new_state = jax.jit(update, out_shardings=(param_layout, state_layout))(p, state, mixed)
    changes = jax.jit(update_diagnostics, out_shardings=rep)(p, new, mixed)
    assert float(changes["all"]["cosine"]) < 0
    assert float(changes["embedding_head"]["fp32_changed_fraction"]) > 0
    assert 0 <= float(changes["embedding_head"]["bf16_changed_fraction"]) <= 1
    donor_grad = jax.jit(jax.grad(lambda d: objective(p, d, tokens, weights("mixed"))[0]))(donor)
    assert all(not np.any(x) for x in jax.tree.leaves(donor_grad))
    import tempfile
    from extent.recovery_durability import RecoveryCheckpointStore
    with tempfile.TemporaryDirectory() as directory:
        store = RecoveryCheckpointStore(Path(directory), "test175")
        payload = dict(parameters=new, optimizer=new_state, cursor=np.asarray(83886080 + 16384, np.int64))
        contract = dict(weights=list(WEIGHTS["mixed"]))
        store.save("mixed", contract, payload, 1)
        restored, meta = store.restore("mixed", contract, payload)
        assert meta["step"] == 1
        assert int(restored["cursor"]) == 83886080 + 16384
        for actual, expected in zip(jax.tree.leaves(restored["optimizer"]), jax.tree.leaves(new_state)):
            np.testing.assert_array_equal(actual, expected)


def test_virtual_eight_device_weighted_objective_and_actual_adam_diagnostics():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=8")
    script = f'import runpy; runpy.run_path({str(Path(__file__).resolve())!r})["virtual_weighted_update"]()'
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr


def test_notebook_budget_and_final_cell_are_executable():
    file = Path(__file__).resolve().parents[1] / "notebooks/extent-tpu-175.ipynb"
    notebook = json.loads(file.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    final = "".join(notebook["cells"][-1]["source"])
    assert "TOTAL_HOURS = 2" in final and "'engine_tpu175'" in final
    assert "SESSION_STARTED" in final


def test_supervisor_import_does_not_initialize_jax():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, engine_tpu175; assert 'jax' not in sys.modules"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_failed_arm_is_retained_and_selected_completed_state_continues(tmp_path, monkeypatch):
    import engine_tpu169
    import engine_tpu175
    import extent.notifications

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(engine_tpu175.signal, "signal", lambda *args: None)
    monkeypatch.setattr(engine_tpu169, "ram_root", lambda root: {})
    monkeypatch.setattr(engine_tpu175, "environment", lambda *args: {})
    monkeypatch.setattr(extent.notifications, "send_telegram_message", lambda *args: None)
    continuations = []

    def fake_worker(command, root, log_path, timeout, **options):
        stage = command[command.index("--stage") + 1]
        Path(log_path).write_text("worker log", encoding="utf-8")
        if stage == "prepare":
            return dict(returncode=0)
        target = Path(command[command.index("--report") + 1])
        arm = command[command.index("--arm") + 1]
        steps = int(command[command.index("--steps") + 1])
        code = 0
        if stage == "diagnose":
            report = dict(status="diagnostic_completed", seconds_per_update=5.2)
        elif stage == "pilot":
            if arm == "kl":
                code, report = 1, dict(status="failed", error={"type": "FloatingPointError"})
            else:
                state = root / "states" / arm
                state.mkdir(parents=True)
                (state / "checkpoint.json").write_text(arm)
                report = dict(status="pilot_completed", step=steps,
                              final_validation_nll=2.9 if arm == "ce" else 2.8)
        else:
            assert (root / "states" / arm / "checkpoint.json").read_text() == arm
            continuations.append(arm)
            report = dict(status="deadline_partial", hf_verified=True)
        target.write_text(json.dumps(report), encoding="utf-8")
        return dict(returncode=code)

    monkeypatch.setattr(engine_tpu175, "wait_child", fake_worker)
    result = engine_tpu175.main([
        "--work-dir", str(tmp_path / "run"), "--output-dir", str(tmp_path / "output"),
        "--max-wall-hours", "2",
    ])
    assert continuations == ["mixed"]
    assert len(result["pilots"]) == 3
    assert result["pilots"][1]["report"]["error"]["type"] == "FloatingPointError"
    assert result["matched_comparison_complete"] is False
    assert result["status"] == "partial"
