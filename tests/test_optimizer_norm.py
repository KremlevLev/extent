import jax
import jax.numpy as jnp
import numpy as np
import pytest

from extent.optimizer import global_norm_fp32, clip_by_global_norm_fp32, gradient_health


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_finite_large_gradients_have_finite_norm_and_nonzero_clipping(dtype):
    # EXP-069-like magnitudes: individual squares fit; their sum overflows FP32.
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("shard",))
    layout = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("shard"))
    values = jax.device_put(jnp.full((1024,), 1.636029321665577e18, dtype), layout)
    grads = {"a": values, "b": -values}
    expected = np.sqrt(sum(np.sum(np.asarray(x, dtype=np.float64) ** 2) for x in grads.values()))
    naive = jax.jit(lambda x: jnp.sum(x.astype(jnp.float32) ** 2))(values)
    assert np.isinf(float(naive))
    health = jax.jit(gradient_health)(grads)
    assert bool(health["grads_finite"])
    assert int(health["nonfinite_grad_leaves"]) == 0
    np.testing.assert_allclose(float(health["grad_norm"]), expected, rtol=2e-6)
    tx = clip_by_global_norm_fp32(1.0)
    clipped, _ = jax.jit(tx.update)(grads, tx.init(grads))
    norm = np.sqrt(sum(np.sum(np.asarray(x, dtype=np.float64) ** 2) for x in clipped.values()))
    assert norm == pytest.approx(1.0, rel=0.01)
    assert clipped["a"].dtype == dtype
    assert np.all(np.asarray(clipped["a"], dtype=np.float32) > 0)
    assert np.all(np.asarray(clipped["b"], dtype=np.float32) < 0)


def test_normal_reduction_is_unchanged_and_zero_is_safe():
    values = {"a": jnp.asarray([3., 4.]), "b": jnp.asarray([0.25])}
    previous = jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(v ** 2) for v in values.values()])))
    np.testing.assert_array_equal(global_norm_fp32(values), previous)
    assert float(global_norm_fp32({})) == 0.0
    assert float(jax.jit(global_norm_fp32)({"a": jnp.zeros(3)})) == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_actual_nonfinite_gradients_are_not_sanitized(bad):
    health = jax.jit(gradient_health)({"a": jnp.asarray([1., bad])})
    assert not bool(health["grads_finite"])
    assert int(health["nonfinite_grad_leaves"]) == 1
    assert not np.isfinite(float(health["grad_norm"]))
