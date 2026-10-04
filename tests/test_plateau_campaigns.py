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
from scripts.m3q_plateau_campaign import make_step, source_initializer
from scripts import m3q_late_distillation_campaign as exp101
from scripts import m3q_plateau_learning_rate_campaign as exp102
from scripts import m3q_late_dynamics_release_campaign as exp103

CAMPAIGNS = (exp101, exp102, exp103)
ARMS = tuple((a, c.SETTINGS[a.name]) for c in CAMPAIGNS for a in c.SPEC.arms)


@pytest.mark.parametrize("entry", ARMS, ids=lambda e: e[0].name)
def test_compiled_stage2_step_and_binary_resume(entry, tmp_path):
    arm, (lr, weight) = entry
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
    # Nonzero recovered corrections, including trained A factors.
    for name, factors in coords.items():
        factors["a"] = factors["a"] * 1.1
        factors["b"] = jnp.full_like(factors["b"], 0.01)
        if "/in_proj/" in name:
            factors["b"] = factors["b"].at[:, active:].set(0)
    protected_start = apply_corrections(base, coords, protected_input_columns=active, head_dim=config.mamba.head_dim)
    released_start = apply_corrections(base, coords, head_dim=config.mamba.head_dim)
    for x,y in zip(jax.tree.leaves(protected_start),jax.tree.leaves(released_start)):
        np.testing.assert_array_equal(x,y)
    tx = optax.chain(stable_clip_by_global_norm(), optax.adam(lr))
    state = tx.init(coords)
    teacher_base = jax.tree.map(lambda x: x * 0.9, base)  # Non-identical target for teacher-KL check.
    step = make_step(model, model, tx, arm, (0, 2), active, config.mamba.head_dim, learning_rate=lr, anchor_weight=weight)
    new, state, loss, norm, finite, health = step(base, coords, state, teacher_base, tokens, jnp.array(1))
    assert bool(finite) and np.isfinite(float(loss)) and np.isfinite(float(norm))
    assert float(loss) == pytest.approx(float(health["ce"] + weight * health["anchor_kl"]), rel=1e-6)
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


def test_source_initializer_pins_hash_base_and_resets_optimizer(tmp_path):
    from dataclasses import asdict
    config = tiny_config()
    model = HybridForCausalLM(config)
    base = model.init(jax.random.key(1), jnp.zeros((1,4),jnp.int32))["params"]
    coords = initialize_corrections(base,(0,2),"INOUT-LORA",seed=123,rank=32,head_dim=config.mamba.head_dim)
    tx = optax.adam(3e-4)
    template = {"coordinates":coords,"optimizer":tx.init(coords)}
    store = CampaignCheckpointStore(tmp_path,"source")
    import json
    contract = json.loads(json.dumps({"source_sha256":{"0":"pinned"},"model":asdict(config)}))
    meta = store.save("seed",template,contract=contract,step=16384,metrics={})
    manifest = {"branches":{"123":{"slot":"seed","metadata":meta}}}
    initialize = source_initializer(manifest,store)
    restored = initialize(123,exp103.SPEC.arms[1],template,{"0":"pinned"})
    for x,y in zip(jax.tree.leaves(restored),jax.tree.leaves(coords)):
        np.testing.assert_array_equal(x,y)
    with pytest.raises(ValueError,match="assembled base"):
        initialize(123,exp103.SPEC.arms[1],template,{"0":"wrong"})
    original_sha = meta["checkpoint_sha256"]
    meta["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError,match="SHA/step"):
        initialize(123,exp103.SPEC.arms[1],template,{"0":"pinned"})
    meta["checkpoint_sha256"] = original_sha
    for name, factors in coords.items():
        if "/in_proj/" in name:
            factors["b"] = factors["b"].at[:,-1].set(0.1)
    new_meta = store.save("seed",template,contract=contract,step=16384,metrics={})
    manifest["branches"]["123"]["metadata"] = new_meta
    with pytest.raises(ValueError,match="nonzero protected columns"):
        initialize(123,exp103.SPEC.arms[1],template,{"0":"pinned"})


def test_registered_plans_and_legacy_resume_contracts(monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.setattr(engine,"artifact_config_from_env",lambda:pytest.fail("offline plan contacted HF"))
    for c in CAMPAIGNS:
        p=c.main(["--plan-only"])["contract"]
        assert p["horizons"][-1]==8192 and p["schedule"]["max_wall_hours"]==8
        assert p["stage2_source"]["step"]==16384
        assert len(p["expected_data_sha256"])==4
    from scripts import m3q_dynamics_protection_campaign as e98, m3q_protected_capacity_campaign as e99, m3q_weak_anchor_campaign as e100
    root=Path(__file__).resolve().parents[1]
    for c in (e98,e99,e100):
        old=json.loads((root/f"results/EXP-{c.SPEC.number:03d}-completed.json").read_text(encoding="utf-8"))["contract"]
        current=c.main(["--plan-only"])["contract"]
        # Kaggle checkout uses LF; this Windows checkout may use CRLF. Compare
        # exact deployed digests after only Git's line-ending conversion.
        import hashlib
        for group in ("implementation_sha256", "additional_implementation_sha256"):
            for file in current[group]:
                if file == "scripts/m3q_subspace_engine.py":
                    continue  # Frozen digest for completed 098--100.
                current[group][file] = hashlib.sha256((root/file).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
        assert old==current
