import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from extent.stable_gradient_clip import stable_clip_by_global_norm, scaled_norm_parts
from extent import HybridForCausalLM, tiny_config
from extent.recovery_subspace import initialize_corrections
from scripts.m3q_numerical_stability_campaign import SPEC, make_step, aggregate

@pytest.mark.parametrize("magnitude", [0.0, 0.01, 3.0, 1e20, 1e38])
def test_stable_clip_matches_float64_reference(magnitude):
    grad = {"a": jnp.array([magnitude, -magnitude], jnp.float32),
            "b": jnp.array([magnitude / 2], jnp.float32)}
    tx = stable_clip_by_global_norm(1.0)
    clipped, _ = jax.jit(tx.update)(grad, tx.init(grad))
    host = np.array([magnitude, -magnitude, magnitude / 2], np.float64)
    expected = host / max(1.0, np.linalg.norm(host))
    actual = np.concatenate([np.asarray(clipped["a"]), np.asarray(clipped["b"])])
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert np.all(np.isfinite(actual))
    if magnitude >= 1e20:
        assert not np.isfinite(float(optax.global_norm(grad)))
        assert np.isfinite(float(scaled_norm_parts(grad)[2]))

def test_nonfinite_input_is_not_silently_repaired():
    tx = stable_clip_by_global_norm()
    grad = {"x": jnp.array([jnp.inf, 1.0])}
    clipped, _ = tx.update(grad, tx.init(grad))
    assert not bool(jnp.all(jnp.isfinite(clipped["x"])))

@pytest.mark.parametrize("arm", SPEC.arms)
def test_diagnostic_step_compiles_for_every_arm(arm):
    config = tiny_config()
    model = HybridForCausalLM(config)
    tokens = jnp.arange(4, dtype=jnp.int32)[None, :]
    if jax.device_count() == 8:
        from extent.initialization import initialize_sharded_parameters
        from extent.sharding import create_v5e_mesh, batch_sharding
        mesh = create_v5e_mesh()
        tokens = jax.device_put(tokens, batch_sharding(mesh))
        base = initialize_sharded_parameters(model, jax.random.key(97), tokens, mesh).params
    else:
        base = model.init(jax.random.key(97), tokens)["params"]
    coords = initialize_corrections(base, (0, 2), "INOUT-LORA", seed=123,
        rank=arm.rank, head_dim=config.mamba.head_dim)
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adam(3e-4))
    inner = int(config.hidden_size * config.mamba.expand)
    active = 2 * inner + 2 * config.mamba.mimo_rank * config.mamba.groups * config.mamba.d_state
    step = make_step(model, model, tx, arm, (0, 2), active, config.mamba.head_dim)
    _, _, loss, _, finite, health = step(base, coords, tx.init(coords), base, tokens, jnp.array(1))
    assert bool(finite) and np.isfinite(float(loss))
    assert bool(health["gradients_finite"]) and bool(health["safe_proposal_finite"])

def test_gate_does_not_call_a_single_seed_or_partial_campaign_success():
    assert not aggregate({})["scientific_gate_passed"]
    result = {"branches": {str(s): {a.name: {"failed": True} for a in SPEC.arms} for s in (123, 456)}}
    result["branches"]["123"]["NAIVE-R8"]["first_norm_only_overflow"] = {"step": 1}
    for s in (123, 456):
        result["branches"][str(s)]["SAFE-R8"] = {"complete": True, "locked_test_nll": 7, "start_test_nll": 12}
    assert aggregate(result)["scientific_gate_passed"]
    result["branches"]["456"]["SAFE-R8"] = {"failed": True}
    assert not aggregate(result)["scientific_gate_passed"]

def test_same_gradient_shadow_proves_norm_only_failure():
    @jax.custom_vjp
    def synthetic(x):
        return jnp.array(0.0)
    def forward(x):
        return jnp.array(0.0), x
    def backward(x, cotangent):
        return (jnp.ones_like(x) * 1e20,)
    synthetic.defvjp(forward, backward)
    class SyntheticModel:
        def apply(self, variables, tokens, return_hidden_states=False):
            value = synthetic(jnp.sum(variables["params"]["layers_0"]["mamba"]["out_proj"]["kernel"]))
            logits = jnp.zeros(tokens.shape + (2,)).at[..., 0].set(value)
            return (logits, (jnp.zeros(tokens.shape + (4,)),)) if return_hidden_states else logits
    base = {"layers_0": {"mamba": {"in_proj": {"kernel": jnp.zeros((4, 10))},
        "out_proj": {"kernel": jnp.zeros((4, 4))}}}}
    coords = initialize_corrections(base, (0,), "INOUT-LORA", seed=123, rank=8)
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adam(3e-4))
    for arm, expected in ((SPEC.arms[0], False), (SPEC.arms[1], True)):
        step = make_step(SyntheticModel(), None, tx, arm, (0,), 6, 1)
        _, _, _, _, finite, health = step(base, coords, tx.init(coords), {},
            jnp.array([[0, 1, 0, 1]], jnp.int32), jnp.array(1))
        assert bool(finite) == expected
        assert bool(health["norm_only_overflow"])
        assert bool(health["gradients_finite"]) and bool(health["safe_proposal_finite"])
        assert not bool(health["naive_norm_finite"])
