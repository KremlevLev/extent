import jax
import jax.numpy as jnp
import numpy as np
import pytest

from extent import HybridForCausalLM, tiny_config
from extent.full_model_distillation import causal_cross_entropy
from extent.mixer_gain import effective_gains, scaled_mamba_parameters


def test_gain_is_folded_only_into_selected_mamba_output_kernels():
    params = {
        "layers_0": {"mamba": {"out_proj": {"kernel": jnp.ones((2, 3))},
                                  "in_proj": {"kernel": jnp.ones((2, 3))}}},
        "layers_1": {"mamba": {"out_proj": {"kernel": jnp.ones((2, 3))}}},
    }
    result = scaled_mamba_parameters(params, (0, 1), jnp.array([0.5, 1.5]))
    np.testing.assert_allclose(result["layers_0"]["mamba"]["out_proj"]["kernel"], 0.5)
    np.testing.assert_allclose(result["layers_1"]["mamba"]["out_proj"]["kernel"], 1.5)
    np.testing.assert_allclose(result["layers_0"]["mamba"]["in_proj"]["kernel"], 1.0)
    np.testing.assert_allclose(params["layers_0"]["mamba"]["out_proj"]["kernel"], 1.0)


def test_effective_gains_are_bounded_and_differentiable():
    global_gains = effective_gains(jnp.array([0.4]), "GLOBAL", 3)
    assert global_gains.shape == (3,)
    assert bool(jnp.all(global_gains > 0.5) & jnp.all(global_gains < 1.5))
    gradient = jax.grad(lambda raw: jnp.sum(effective_gains(raw, "PER-LAYER", 3)))(
        jnp.zeros((3,))
    )
    np.testing.assert_allclose(gradient, 0.5)


def test_mismatched_gain_shape_is_rejected():
    with pytest.raises(ValueError):
        effective_gains(jnp.zeros((2,)), "GLOBAL", 3)
    with pytest.raises(ValueError):
        scaled_mamba_parameters({}, (0, 1), jnp.ones((1,)))


def test_full_model_gain_step_compiles_and_has_finite_gradient():
    model = HybridForCausalLM(tiny_config())
    tokens = (jnp.arange(8, dtype=jnp.int32) % 128)[None, :]
    params = model.init(jax.random.key(93), tokens)["params"]

    @jax.jit
    def loss_and_gradient(raw):
        def objective(coordinates):
            gains = effective_gains(coordinates, "PER-LAYER", 2)
            candidate = scaled_mamba_parameters(params, (0, 2), gains)
            return causal_cross_entropy(model.apply({"params": candidate}, tokens), tokens)
        return jax.value_and_grad(objective)(raw)

    loss, grad = loss_and_gradient(jnp.zeros((2,), dtype=jnp.float32))
    assert np.isfinite(float(loss))
    assert bool(jnp.all(jnp.isfinite(grad)))
