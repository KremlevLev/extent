from copy import deepcopy
from dataclasses import replace
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from extent.campaign_checkpoint import CampaignCheckpointStore
from extent.config import tiny_config, MLAConfig
from extent.layers.mamba3 import Mamba3MIMO
from extent.qwen3_teacher import Qwen3TeacherConfig
from extent.sharding import create_v5e_mesh, batch_sharding
from scripts import m3q_allocation_campaign as campaign


def test_registered_comparison_changes_only_placement_and_has_equal_counts():
    assert campaign.PLACEMENTS == {"UNIFORM": (6, 13, 20, 27), "ATLAS": (0, 1, 26, 27)}
    assert len(campaign.required_layers()) == 27
    assert 27 not in campaign.required_layers()
    assert all(28 - len(p) == 24 for p in campaign.PLACEMENTS.values())


def test_gate_requires_both_seeds_and_auc_not_just_final_win():
    result = {"arms": {}}
    for seed in campaign.SEEDS:
        result["arms"][str(seed)] = {}
        for arm, nll in (("UNIFORM", 5.0), ("ATLAS", 4.5)):
            result["arms"][str(seed)][arm] = {
                "complete": True, "finite": True,
                "evaluations": {str(s): {"student_nll": nll, "window_nll": [nll, nll]}
                                for s in campaign.CHECKPOINTS},
            }
    assert campaign.aggregate(result)["scientific_gate_passed"]
    bad = deepcopy(result)
    bad["arms"][str(campaign.SEEDS[-1])]["ATLAS"]["complete"] = False
    assert not campaign.aggregate(bad)["scientific_gate_passed"]
    bad = deepcopy(result)
    for s in campaign.CHECKPOINTS[:-1]:
        bad["arms"][str(campaign.SEEDS[-1])]["ATLAS"]["evaluations"][str(s)]["student_nll"] = 10.0
    assert not campaign.aggregate(bad)["scientific_gate_passed"]


def test_interrupted_training_resumes_identical_optimizer_and_data_cursor(tmp_path):
    import optax
    tx = optax.adam(0.01)
    initial = {"x": jnp.asarray(0.2)}
    tokens = np.arange(12, dtype=np.int32).reshape(4, 3)
    layout = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    @jax.jit
    def step(p, opt, batch):
        loss, g = jax.value_and_grad(lambda q: (q["x"] - batch.mean()) ** 2)(p)
        updates, opt = tx.update(g, opt, p)
        return optax.apply_updates(p, updates), opt, {"loss": loss, "grads_finite": jnp.asarray(True)}
    def evaluate(p):
        return {"student_nll": float(p["x"]), "finite": True}
    def row():
        return {"completed_steps": 0, "evaluations": {}, "training_metrics": {}, "complete": False}
    store = CampaignCheckpointStore(tmp_path, "exp")
    def save(p, opt, current):
        store.save("run", {"params": p, "opt_state": opt}, contract={}, step=current["completed_steps"], metrics=current)
    args = dict(step_fn=step, evaluate=evaluate, train_tokens=tokens, batch_layout=layout,
                deadline=float("inf"), save=save, checkpoints=(0, 1, 2, 4))
    campaign.run_training_segment(initial, tx.init(initial), row(), total_steps=4, **args)
    expected, _ = store.restore("run", {}, {"params": initial, "opt_state": tx.init(initial)})
    campaign.run_training_segment(initial, tx.init(initial), row(), total_steps=2, **args)
    restored, meta = store.restore("run", {}, {"params": initial, "opt_state": tx.init(initial)})
    campaign.run_training_segment(restored["params"], restored["opt_state"], meta["metrics"], total_steps=4, **args)
    actual, _ = store.restore("run", {}, {"params": initial, "opt_state": tx.init(initial)})
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(a, b)


def test_nonfinite_update_keeps_last_good_checkpoint(tmp_path):
    store = CampaignCheckpointStore(tmp_path, "exp")
    def save(p, o, row):
        store.save("good", {"params": p}, contract={}, step=row["completed_steps"], metrics=row)
    row = {"completed_steps": 0, "evaluations": {}, "training_metrics": {}, "complete": False}
    with pytest.raises(FloatingPointError):
        campaign.run_training_segment({"x": jnp.asarray(1.0)}, (), row,
            step_fn=lambda p, o, b: ({"x": jnp.asarray(float("nan"))}, o,
                                     {"loss": jnp.asarray(float("nan")), "grads_finite": False}),
            evaluate=lambda p: {"finite": True, "student_nll": 1.0},
            train_tokens=np.zeros((1, 2), np.int32),
            batch_layout=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
            deadline=float("inf"), save=save, total_steps=2, checkpoints=(0, 2))
    state, meta = store.restore("good", {})
    assert meta["step"] == 0
    assert float(state["params"]["x"]) == 1.0
    assert row["finite"] is False
    assert row["failed_update"]["attempted_step"] == 1
    assert row["failed_update"]["last_durable_step"] == 0
    assert row["failed_update"]["nonfinite_metrics"] == ["loss"]
    assert row["failed_update"]["metrics"]["loss"] == "nan"
    import json
    json.dumps(row, allow_nan=False)


def test_norm_overflow_is_reported_separately_from_nonfinite_gradients():
    row = {"completed_steps": 0, "evaluations": {}, "training_metrics": {}, "complete": False}
    with pytest.raises(FloatingPointError, match="grads_finite=True"):
        campaign.run_training_segment({}, (), row,
            step_fn=lambda p, o, b: (p, o, {"loss": 1.0, "grad_norm": float("inf"), "grads_finite": True}),
            evaluate=lambda p: {"finite": True}, train_tokens=np.zeros((1, 2), np.int32),
            batch_layout=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
            deadline=float("inf"), save=lambda *a: None, total_steps=1)
    assert row["failed_update"]["nonfinite_metrics"] == ["grad_norm"]
    assert row["failed_update"]["metrics"]["grad_norm"] == "inf"


def test_tiny_full_model_materializes_trains_saves_and_skips_completed_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "PLACEMENTS", {"UNIFORM": (1,), "ATLAS": (0,)})
    monkeypatch.setattr(campaign, "TOTAL_STEPS", 2)
    monkeypatch.setattr(campaign, "CHECKPOINTS", (0, 1, 2))
    monkeypatch.setattr(campaign, "SEQUENCE_LENGTH", 4)
    monkeypatch.setattr(campaign, "EVAL_WINDOWS", 2)
    # Use a real small JAX model, optimizer, sharding, and checkpoint round-trip.
    cfg = replace(tiny_config(), param_dtype="bfloat16", compute_dtype="bfloat16",
                  mla=MLAConfig(implementation="qwen3_gqa", num_heads=4, num_kv_heads=2,
                                qk_rope_head_dim=16, rope_original_head_dim=16, v_head_dim=16))
    source = Qwen3TeacherConfig(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
                               intermediate_size=cfg.intermediate_size, num_layers=cfg.num_layers,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                               param_dtype="bfloat16", compute_dtype="bfloat16", remat_policy="none")
    monkeypatch.setattr(campaign, "load_wikitext2_tokens", lambda count, *a, **k: np.arange(count, dtype=np.int32) % cfg.vocab_size)
    monkeypatch.setattr(campaign, "stream_teacher_qwen_weights", lambda p, *a: (p, None))
    monkeypatch.setattr(campaign, "stream_direct_qwen_weights", lambda p, *a: (p, None))
    monkeypatch.setattr(campaign, "load_mixer_arrays", lambda *a: None)
    monkeypatch.setattr(campaign, "materialize_qwen_gqa_layer", lambda p, *a: (p, None))
    # Small run needs a shorter warmup but exercises the actual Lion implementation.
    create_lion = campaign.create_lion
    monkeypatch.setattr(campaign, "create_lion", lambda **kw: create_lion(**dict(kw, warmup_steps=1)))
    mesh = create_v5e_mesh()
    store = CampaignCheckpointStore(tmp_path, "exp")
    contract = campaign.contract_for(cfg)
    for seed in campaign.SEEDS:
        for layer in range(cfg.num_layers):
            mamba = Mamba3MIMO(cfg.hidden_size, cfg.mamba, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
            p = mamba.init(jax.random.key(seed + layer), jnp.zeros((1, 4, cfg.hidden_size), jnp.bfloat16))["params"]
            store.save(f"prep/seed-{seed}/layer-{layer}", {"params": p},
                       contract=dict(contract, seed=seed, layer=layer, kind="prepared_mamba"), step=2, metrics={})
    result = {"arms": {}}
    campaign.run_full_models(cfg, source, None, mesh, batch_sharding(mesh), store, contract, result,
                             str(tmp_path), time.monotonic() + 1800, lambda: None)
    assert all(r["complete"] for arms in result["arms"].values() for r in arms.values())
    for seed in campaign.SEEDS:
        rows = result["arms"][str(seed)]
        assert rows["UNIFORM"]["prepared_checkpoint_hashes"]["2"] == rows["ATLAS"]["prepared_checkpoint_hashes"]["2"]
    # New invocation must recover the compact metrics and skip finished training.
    monkeypatch.setattr(campaign, "run_training_segment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    resumed = {"arms": {}}
    campaign.run_full_models(cfg, source, None, mesh, batch_sharding(mesh), store, contract, resumed,
                             str(tmp_path), time.monotonic() + 1800, lambda: None)
    assert resumed["arms"] == result["arms"]


def test_tiny_dual_preparation_readout_and_recovery_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "PREP_STEPS", 2)
    monkeypatch.setattr(campaign, "DUAL_STEPS", 2)
    monkeypatch.setattr(campaign, "validate_external_evaluation_cache", lambda *a, **k: slice(0, 2))
    cfg = tiny_config()
    source = Qwen3TeacherConfig(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
                               intermediate_size=cfg.intermediate_size, num_layers=cfg.num_layers,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 4, cfg.hidden_size)).astype(np.float32)
    arrays = {"normalized_input": x, "residual_input": x, "attention_target": x * 0.2}
    tail = campaign.Qwen3DecoderTail(cfg.hidden_size, cfg.intermediate_size, source.rms_norm_eps,
                                    jnp.bfloat16, jnp.float32)
    tail_params = tail.init(jax.random.key(3), jnp.asarray(x[:1]), jnp.asarray(x[:1]) * 0.2)["params"]
    monkeypatch.setattr(campaign, "qwen3_decoder_tail_params", lambda *a: tail_params)
    manifest = {"window_layout": {"calibration": [0, 2], "training": [2, 4]}}
    params, metrics = campaign.prepare_layer(cfg, source, None, arrays, arrays, manifest, {},
                                             0, 123, time.monotonic() + 600)
    assert metrics["bridge"]["finite"]
    assert metrics["recovery"]["finite"]
    store = CampaignCheckpointStore(tmp_path, "exp")
    store.save("prepared", {"params": params}, contract={}, step=2, metrics=metrics)
    restored, meta = store.restore("prepared", {})
    assert meta["metrics"]["recovery"]["complete"]
    for value in jax.tree.leaves(restored["params"]):
        assert np.all(np.isfinite(value))
