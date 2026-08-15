import jax
import numpy as np

from singularity import HybridForCausalLM, tiny_config
from singularity.optimizer import create_lion
from singularity.sharding import create_v5e_mesh
from singularity.train_step import initialize_sharded_runtime, shard_host_batch


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
    model = HybridForCausalLM(tiny_config())
    tx = create_lion(total_steps=4, warmup_steps=1)
    runtime = initialize_sharded_runtime(model, tx, jax.random.key(7), batch, mesh, donate_state=False)

    param = runtime.state.params["layers_0"]["mamba"]["in_proj"]["kernel"]
    momentum = runtime.state.opt_state[1].mu["layers_0"]["mamba"]["in_proj"]["kernel"]
    assert param.sharding.spec == momentum.sharding.spec
    assert param.sharding.spec[-1] == "tensor"

    _, metrics = runtime.train_step(runtime.state, batch)
    jax.block_until_ready(metrics)
    assert np.isfinite(float(metrics["loss"]))
    assert np.isfinite(float(metrics["grad_norm"]))
