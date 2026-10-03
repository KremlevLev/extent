"""Numerical and registered-contract checks for EXP-098--100."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from extent import HybridForCausalLM, tiny_config
from extent.campaign_checkpoint import CampaignCheckpointStore
from extent.recovery_subspace import apply_corrections, initialize_corrections
from extent.stable_gradient_clip import stable_clip_by_global_norm
from scripts import m3q_subspace_engine as engine
from scripts.m3q_safe_recovery_campaign import make_step, aggregate
from scripts import m3q_dynamics_protection_campaign as exp098
from scripts import m3q_protected_capacity_campaign as exp099
from scripts import m3q_weak_anchor_campaign as exp100

CAMPAIGNS = (exp098, exp099, exp100)
ARMS = tuple(a for c in CAMPAIGNS for a in c.SPEC.arms)


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.name)
def test_compiled_safe_step_and_binary_resume(arm, tmp_path):
    config = tiny_config()
    model = HybridForCausalLM(config)
    tokens = jnp.arange(4, dtype=jnp.int32)[None, :]
    if jax.device_count() == 8:
        from extent.initialization import initialize_sharded_parameters
        from extent.sharding import create_v5e_mesh, batch_sharding
        mesh = create_v5e_mesh()
        tokens = jax.device_put(tokens, batch_sharding(mesh))
        base = initialize_sharded_parameters(model, jax.random.key(98), tokens, mesh).params
    else:
        base = model.init(jax.random.key(98), tokens)["params"]
    inner = int(config.hidden_size * config.mamba.expand)
    active = 2 * inner + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
    coords = initialize_corrections(base, (0, 2), "INOUT-LORA", seed=123,
        rank=arm.rank, head_dim=config.mamba.head_dim)
    tx = optax.chain(stable_clip_by_global_norm(), optax.adam(3e-4))
    state = tx.init(coords)
    teacher_base = jax.tree.map(lambda x: x * 0.9, base)  # Non-identical target for teacher-KL check.
    step = make_step(model, model, tx, arm, (0, 2), active, config.mamba.head_dim)
    new, state, loss, norm, finite, health = step(base, coords, state, teacher_base, tokens, jnp.array(1))
    assert bool(finite) and np.isfinite(float(loss)) and np.isfinite(float(norm))
    assert float(loss) == pytest.approx(float(health["ce"] + 0.1 * health["anchor_kl"]), rel=1e-6)
    assert any(not np.array_equal(x, y) for x, y in zip(jax.tree.leaves(coords), jax.tree.leaves(new)))
    if arm.protected:
        changed = apply_corrections(base, new, protected_input_columns=active, head_dim=config.mamba.head_dim)
        for layer in (0, 2):
            np.testing.assert_array_equal(changed[f"layers_{layer}"]["mamba"]["in_proj"]["kernel"][:, active:],
                base[f"layers_{layer}"]["mamba"]["in_proj"]["kernel"][:, active:])
    store = CampaignCheckpointStore(tmp_path, "unit-safe")
    template = {"coordinates": new, "optimizer": state}
    store.save("arm", template, contract={"arm": arm.name}, step=1, metrics={})
    restored, _ = store.restore("arm", {"arm": arm.name}, template)
    expected = step(base, new, state, teacher_base, tokens, jnp.array(2))
    actual = step(base, restored["coordinates"], restored["optimizer"], teacher_base, tokens, jnp.array(2))
    for a, b in zip(jax.tree.leaves(expected[:2]), jax.tree.leaves(actual[:2])):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_registered_comparisons_separate_terminal_failure_and_domain_gate(campaign):
    spec = campaign.SPEC
    result = {"branches": {str(s): {a.name: {"complete": True,
        "locked_test_nll": 7 if a.name == spec.primary else 8, "start_test_nll": 12,
        "pg19_test_nll": 9 if a.name == spec.primary else 10, "start_pg19_nll": 14}
        for a in spec.arms} for s in (123, 456)}}
    assert aggregate(spec, result)["scientific_gate_passed"]
    assert aggregate(spec, result)["cross_domain_gate_passed"]
    result["branches"]["456"][spec.primary]["pg19_test_nll"] = 15
    assert aggregate(spec, result)["scientific_gate_passed"]
    assert not aggregate(spec, result)["cross_domain_gate_passed"]
    result["branches"]["456"][spec.primary] = {"failed": True}
    assert aggregate(spec, result)["terminal"]
    assert not aggregate(spec, result)["scientific_gate_passed"]
    del result["branches"]["456"][spec.primary]
    assert not aggregate(spec, result)["terminal"]


def test_plan_only_is_offline_and_does_not_mutate_other_campaigns(monkeypatch):
    original = (engine.HORIZONS, engine.DATA, engine.TRAIN_WINDOWS)
    monkeypatch.setattr(engine, "artifact_config_from_env", lambda: pytest.fail("plan-only contacted HF"))
    monkeypatch.setattr(engine.jax, "devices", lambda: pytest.fail("plan-only initialized backend"))
    plans = [c.main(["--plan-only"]) for c in CAMPAIGNS]
    assert (engine.HORIZONS, engine.DATA, engine.TRAIN_WINDOWS) == original
    assert [p["contract"]["horizons"][-1] for p in plans] == [16384, 12288, 8192]
    assert [p["contract"]["student_training_tokens"] for p in plans] == [16777216, 18874368, 12582912]
    for plan in plans:
        contract = plan["contract"]
        assert contract["schedule"]["max_wall_hours"] == 8
        assert contract["schedule"]["reserve_minutes"] == 45
        assert contract["optimizer"]["lr"] == 3e-4
        assert set(contract["additional_implementation_sha256"])
    with pytest.raises(ValueError, match="eight-hour"):
        exp098.main(["--plan-only", "--max-wall-hours", "8.25"])


def test_legacy_097_engine_contract_is_preserved():
    from scripts.m3q_numerical_stability_campaign import SPEC
    config = tiny_config()
    assert engine.contract_for(SPEC, config)["implementation_sha256"]["scripts/m3q_subspace_engine.py"] == engine.EXP097_ENGINE_SHA256


def test_final_checkpoint_requires_both_evaluations_before_completion():
    assert engine.needs_work({"step": 16384}, 16384, 16384)
    assert not engine.needs_work({"step": 16384, "complete": True}, 16384, 16384)
    with pytest.raises(ValueError, match="unique train windows"):
        engine.Schedule((8192, 16384), 8192, exp098.SCHEDULE.data, 0)
