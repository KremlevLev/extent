from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import optax

from extent import HybridForCausalLM, tiny_config
from extent.downstream_recovery import (
    HybridDecoderSuffix, suffix_parameters, make_downstream_recovery_step,
)


def test_suffix_matches_full_model_and_candidate_only_backward_is_finite():
    config = replace(tiny_config(), param_dtype="float32", compute_dtype="float32")
    model = HybridForCausalLM(config)
    tokens = jnp.arange(4)[None]
    params = model.init(jax.random.key(1), tokens)["params"]
    logits, states = model.apply({"params": params}, tokens, return_hidden_states=True)
    suffix = HybridDecoderSuffix(config, 1)
    frozen = suffix_parameters(params, 1)
    actual = jax.jit(suffix.apply)({"params": frozen}, states[0])
    np.testing.assert_allclose(actual, logits, atol=1e-5, rtol=1e-5)
    # Early pair: suffix must backpropagate through all following frozen layers.
    suffix = HybridDecoderSuffix(config, 0)
    pair = config.mamba_layer_indices[:2]
    candidate = {f"layers_{i}": params[f"layers_{i}"]["mamba"] for i in pair}
    tx = optax.sgd(0.001)
    hidden = params["embed_tokens"]["embedding"][tokens]
    before = jax.tree.map(np.asarray, params)
    updated, state, metrics = jax.jit(make_downstream_recovery_step(
        suffix, tx, bf16_gradients=False,
    ))(candidate, tx.init(candidate), params, hidden, jnp.zeros_like(logits))
    assert bool(metrics["grads_finite"])
    assert float(metrics["loss"]) > 0
    assert set(updated) == set(candidate)
    for key in candidate:
        assert any(np.any(np.asarray(a) != np.asarray(b)) for a, b in zip(
            jax.tree.leaves(candidate[key]), jax.tree.leaves(updated[key]), strict=True,
        ))
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(a, b)
