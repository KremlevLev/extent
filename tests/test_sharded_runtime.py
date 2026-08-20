import jax
import numpy as np
import pytest
from dataclasses import replace
from flax import traverse_util
from types import SimpleNamespace

from extent import HybridForCausalLM, tiny_config
from extent.optimizer import create_lion
from extent.hardware import recommended_compute_dtype
from extent.sharding import create_v5e_mesh
from extent.train_step import initialize_sharded_runtime, shard_host_batch


def test_t4_uses_safe_fp32_compute_policy():
    decision = recommended_compute_dtype(
        [SimpleNamespace(platform="gpu", device_kind="Tesla T4")]
    )
    assert decision.dtype == "float32"
    assert "pre-Ampere" in decision.reason


def test_sharded_initialization_lion_layout_and_train_step():
    mesh = create_v5e_mesh()
    tokens = np.arange(mesh.shape["data"] * 3, dtype=np.int32).reshape(mesh.shape["data"], 3)
    batch = shard_host_batch(
        {
            "input_ids": tokens,
            "labels": tokens.copy(),
            "attention_mask": np.ones_like(tokens, dtype=np.bool_),
            "loss_mask": np.ones_like(tokens, dtype=np.bool_),
        },
        mesh,
    )
    dtype_decision = recommended_compute_dtype()
    model = HybridForCausalLM(replace(tiny_config(), compute_dtype=dtype_decision.dtype))
    tx = create_lion(total_steps=4, warmup_steps=1)
    runtime = initialize_sharded_runtime(model, tx, jax.random.key(7), batch, mesh, donate_state=False)

    param = runtime.state.params["layers_0"]["mamba"]["in_proj"]["kernel"]
    momentum = runtime.state.opt_state[1].mu["layers_0"]["mamba"]["in_proj"]["kernel"]
    assert param.sharding.spec == momentum.sharding.spec
    assert param.sharding.spec[-1] == "tensor"

    _, metrics = runtime.train_step(runtime.state, batch)
    jax.block_until_ready(metrics)
    assert np.isfinite(float(metrics["loss"]))
    if not bool(metrics["grads_finite"]):
        _, diagnostics = runtime.diagnose_gradients(runtime.state.params, batch)
        jax.block_until_ready(diagnostics)
        finite = traverse_util.flatten_dict(diagnostics["finite"])
        bad_paths = ["/".join(path) for path, value in finite.items() if not bool(value)]
        pytest.fail(f"non-finite gradient parameters: {bad_paths}")
    assert np.isfinite(float(metrics["grad_norm"]))
    assert int(metrics["nonfinite_grad_leaves"]) == 0
