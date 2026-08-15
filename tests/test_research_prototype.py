import jax
import jax.numpy as jnp
import numpy as np

from singularity import HybridForCausalLM, tiny_config
from singularity.estimate import parameter_count
from singularity.layers.mamba3 import Mamba3MIMO, heavy_tail_activation
from singularity.layers.mla import MultiHeadLatentAttention
from singularity.optimizer import cast_grads_bf16, create_lion
from singularity.sharding import create_v5e_mesh, parameter_partition_spec
from singularity.weight_mapping import fit_matrix, truncated_svd


def test_mamba3_shape_jit_and_grad():
    config = tiny_config()
    block = Mamba3MIMO(config.hidden_size, config.mamba)
    x = jnp.ones((2, 7, config.hidden_size), jnp.bfloat16)
    variables = block.init(jax.random.key(0), x)
    output = jax.jit(block.apply)(variables, x)
    grads = jax.grad(lambda p: block.apply({"params": p}, x).astype(jnp.float32).mean())(variables["params"])
    assert output.shape == x.shape
    assert output.dtype == jnp.bfloat16
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(grads))


def test_mla_and_hybrid_shapes_and_bf16_weights():
    config = tiny_config()
    x = jnp.ones((1, 6, config.hidden_size), jnp.bfloat16)
    mla = MultiHeadLatentAttention(config.hidden_size, config.mla)
    assert mla.apply(mla.init(jax.random.key(1), x), x).shape == x.shape
    model = HybridForCausalLM(config)
    tokens = jnp.ones((1, 6), jnp.int32)
    variables = model.init(jax.random.key(2), tokens)
    logits = jax.jit(model.apply)(variables, tokens)
    assert logits.shape == (1, 6, config.vocab_size)
    assert logits.dtype == jnp.float32
    assert {leaf.dtype for leaf in jax.tree.leaves(variables["params"])} == {jnp.dtype(jnp.bfloat16)}


def test_optimizer_mesh_and_partition_contracts():
    params = {"kernel": jnp.ones((4, 4), jnp.bfloat16)}
    grads = cast_grads_bf16(jax.grad(lambda p: jnp.sum(p["kernel"].astype(jnp.float32) ** 2))(params))
    tx = create_lion(total_steps=4, warmup_steps=1, accumulation_steps=1)
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    assert updates["kernel"].dtype == jnp.bfloat16
    assert create_v5e_mesh().size == jax.device_count()
    assert parameter_partition_spec(("mlp", "gate_proj", "kernel"), (4, 8)) == jax.sharding.PartitionSpec("fsdp", "tensor")


def test_mapping_math_and_parameter_estimate():
    source = np.arange(48, dtype=np.float32).reshape(6, 8)
    down, up = truncated_svd(source, 3)
    assert down.shape == (6, 3) and up.shape == (3, 8)
    assert fit_matrix(source, (9, 11)).shape == (9, 11)
    counts = parameter_count(tiny_config())
    assert counts["total"] > 0
    assert counts["total"] == counts["embeddings_and_head"] + counts["mamba_layers"] + counts["mla_layers"]


def test_heavy_tail_is_positive_and_continuous():
    values = heavy_tail_activation(jnp.array([-1e-5, 0.0, 1e-5]))
    assert jnp.all(values > 0)
    assert jnp.max(jnp.abs(jnp.diff(values))) < 2e-5
