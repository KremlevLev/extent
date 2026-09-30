import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from extent import HybridForCausalLM, tiny_config
from extent.recovery_subspace import initialize_corrections, apply_corrections
from extent.campaign_checkpoint import CampaignCheckpointStore
from scripts.m3q_subspace_engine import make_train_step, aggregate, needs_work, HORIZONS, coordinate_plan
from scripts.m3q_subspace_campaign import SPEC as S94
from scripts.m3q_subspace_objective_campaign import SPEC as S95
from scripts.m3q_subspace_capacity_campaign import SPEC as S96

@pytest.mark.parametrize("arm", S94.arms + S95.arms[1:] + S96.arms[1:])
def test_zero_change_full_model_step_and_finite_coordinates(arm):
    config = tiny_config()
    model = HybridForCausalLM(config)
    tokens = jnp.arange(4, dtype=jnp.int32)[None, :]
    base = model.init(jax.random.key(94), tokens)["params"]
    order = (0, 2)
    coords = initialize_corrections(base, order, arm.subspace, seed=123,
                                   rank=arm.rank, head_dim=config.mamba.head_dim)
    assert sum(int(x.size) for x in jax.tree.leaves(coords)) == coordinate_plan(config, arm, 2)["allocated_coordinates"]
    inner = int(config.hidden_size * config.mamba.expand)
    active = 2 * inner + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
    changed = apply_corrections(base, coords, head_dim=config.mamba.head_dim,
                               protected_input_columns=active if arm.protected else None)
    for old, new in zip(jax.tree.leaves(base), jax.tree.leaves(changed)):
        np.testing.assert_array_equal(old, new)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    step = make_train_step(model, model, tx, arm, order, active, config.mamba.head_dim)
    updated, _, loss, norm, finite = step(base, coords, tx.init(coords), base, tokens, jnp.array(1))
    assert bool(finite) and np.isfinite(float(loss)) and np.isfinite(float(norm))
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(coords), jax.tree.leaves(updated)))
    if arm.objective == "delta_then_teacher":
        _, _, loss, _, finite = step(base, coords, tx.init(coords), base, tokens, jnp.array(2049))
        assert bool(finite) and np.isfinite(float(loss))

def test_protected_projection_columns_are_exactly_unchanged_and_zero_gradient():
    base = {"layers_0": {"mamba": {"in_proj": {"kernel": jnp.zeros((4, 10))},
                                   "out_proj": {"kernel": jnp.zeros((4, 4))}}}}
    coords = initialize_corrections(base, (0,), "INOUT-LORA", seed=1, rank=2)
    coords = jax.tree.map(jnp.ones_like, coords)
    result = apply_corrections(base, coords, protected_input_columns=6)
    kernel = result["layers_0"]["mamba"]["in_proj"]["kernel"]
    np.testing.assert_array_equal(kernel[:, 6:], 0)
    np.testing.assert_array_equal(kernel[:, :6], 2)
    grad = jax.grad(lambda c: jnp.sum(apply_corrections(base, c, protected_input_columns=6)
                    ["layers_0"]["mamba"]["in_proj"]["kernel"]))(coords)
    np.testing.assert_array_equal(grad["layers_0/mamba/in_proj/kernel"]["b"][:, 6:], 0)

def test_binary_optimizer_resume_has_identical_next_update(tmp_path):
    coords = {"a": jnp.ones((2, 2))}
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adam(3e-4))
    state = tx.init(coords)
    update, state = tx.update(coords, state, coords)
    coords = optax.apply_updates(coords, update)
    template = {"coordinates": coords, "optimizer": state}
    store = CampaignCheckpointStore(tmp_path, "test")
    store.save("seed/arm", template, contract={"v": 1}, step=1, metrics={})
    restored, metadata = store.restore("seed/arm", {"v": 1}, template)
    assert metadata["step"] == 1
    a, _ = tx.update(coords, state, coords)
    b, _ = tx.update(restored["coordinates"], restored["optimizer"], restored["coordinates"])
    for old, new in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(old, new)
    with pytest.raises(ValueError, match="contract mismatch"):
        store.restore("seed/arm", {"v": 2}, template)

def test_registered_gate_requires_all_arms_and_both_seeds():
    result = {"branches": {str(seed): {arm.name: {"complete": True,
        "locked_test_nll": 8 if arm.name == S94.primary else 9,
        "start_test_nll": 10} for arm in S94.arms} for seed in (123, 456)}}
    assert aggregate(S94, result)["scientific_gate_passed"]
    result["branches"]["456"][S94.primary]["locked_test_nll"] = 10
    assert not aggregate(S94, result)["scientific_gate_passed"]
    del result["branches"]["456"]["HEAD-GAIN"]
    assert not aggregate(S94, result)["complete"]

def test_final_evaluation_can_resume_after_last_training_checkpoint():
    assert needs_work({"step": HORIZONS[-1]}, HORIZONS[-1])
    assert not needs_work({"step": HORIZONS[-1], "complete": True}, HORIZONS[-1])
    assert not needs_work({"step": 2048}, 2048)
    assert len({spec.prefix for spec in (S94, S95, S96)}) == 3

@pytest.mark.skipif(jax.device_count() != 8, reason="requires eight CPU or TPU devices")
@pytest.mark.parametrize("arm", [S96.arms[-1], S95.arms[1], S95.arms[-1]])
def test_sharded_full_model_correction_step(arm):
    from extent.initialization import initialize_sharded_parameters
    from extent.sharding import create_v5e_mesh, batch_sharding
    config = tiny_config()
    model = HybridForCausalLM(config)
    mesh = create_v5e_mesh()
    tokens = jax.device_put(np.arange(4, dtype=np.int32)[None, :], batch_sharding(mesh))
    base = initialize_sharded_parameters(model, jax.random.key(94), tokens, mesh).params
    coords = initialize_corrections(base, (0, 2), arm.subspace, seed=123,
                                   rank=arm.rank, head_dim=config.mamba.head_dim)
    inner = int(config.hidden_size * config.mamba.expand)
    active = 2 * inner + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adam(3e-4))
    step = make_train_step(model, model, tx, arm, (0, 2), active, config.mamba.head_dim)
    _, _, loss, norm, finite = step(base, coords, tx.init(coords), base, tokens, jnp.array(1))
    assert bool(finite) and np.isfinite(float(loss)) and np.isfinite(float(norm))
