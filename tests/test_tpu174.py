import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from engine_tpu174 import PILOT_STEPS, select_winner
from extent.end_to_end_174 import make_gradient, separate_head_loss


def test_selection_uses_completed_pilots_even_above_start_nll():
    def row(candidate, nll, step=PILOT_STEPS, code=0):
        return dict(candidate=candidate, returncode=code, report=dict(
            status="pilot_completed", step=step, selection_nll=nll,
            evaluations=[dict(step=0, nll=2.58)],
        ))

    winner = select_winner([
        row("full-1e-06", 5.0), row("mamba-1e-06", 4.0),
        row("full-3e-06", 1.0, step=16), row("full-1e-05", 0.5, code=1),
    ])
    assert winner["candidate"] == "mamba-1e-06"


def test_separate_heads_match_dense_ce_kl_and_teacher_is_detached():
    rng = np.random.default_rng(174)
    student, teacher = jnp.asarray(rng.normal(size=(2, 2, 5, 8)), dtype=jnp.float32)
    student_head, teacher_head = jnp.asarray(rng.normal(size=(2, 19, 8)), dtype=jnp.float32)
    labels = jnp.asarray(rng.integers(0, 19, size=(2, 5)))

    def dense(head):
        logp = jax.nn.log_softmax(student @ head.T)
        logq = jax.nn.log_softmax(teacher @ teacher_head.T)
        ce = -jnp.take_along_axis(logp, labels[..., None], -1).mean()
        kl = jnp.sum(jnp.exp(logq) * (logq - logp), -1).mean()
        return 0.5 * (ce + kl)

    def tiled(head, donor):
        return separate_head_loss(student, teacher, head, donor, labels, tile=4)[0]

    np.testing.assert_allclose(tiled(student_head, teacher_head), dense(student_head), rtol=1e-6)
    actual, donor_grad = jax.grad(tiled, argnums=(0, 1))(student_head, teacher_head)
    np.testing.assert_allclose(actual, jax.grad(dense)(student_head), atol=1e-7)
    assert not np.any(donor_grad)


def virtual_full_fixture():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from extent.clean_transplant_170 import initialize
    from extent.clean_tpu_171 import layout
    from extent.config import Mamba3Config
    from extent.qwen3_teacher import Qwen3TeacherConfig

    rng = np.random.default_rng(174)
    tc = Qwen3TeacherConfig(vocab_size=24, hidden_size=8, intermediate_size=16, num_layers=28,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        param_dtype="bfloat16", compute_dtype="bfloat16")
    c = SimpleNamespace(hidden_size=8, rms_norm_eps=1e-6,
        mamba=Mamba3Config(head_dim=4, d_state=4, groups=1, expand=1., mimo_rank=4))

    def matrix(a, b):
        return jnp.asarray(rng.normal(0, .08, (a, b)), dtype=jnp.bfloat16)

    attn = {name: dict(kernel=matrix(a, b)) for name, a, b in (
        ("q_proj", 8, 8), ("k_proj", 8, 4), ("v_proj", 8, 4), ("o_proj", 8, 8))}
    attn.update(q_norm=dict(scale=jnp.ones(4, jnp.bfloat16)), k_norm=dict(scale=jnp.ones(4, jnp.bfloat16)))
    norm = dict(scale=jnp.ones(8, jnp.bfloat16))
    block = dict(self_attn=attn, input_layernorm=norm, post_attention_layernorm=norm,
        mlp=dict(gate_proj=dict(kernel=matrix(8, 16)), up_proj=dict(kernel=matrix(8, 16)),
                 down_proj=dict(kernel=matrix(16, 8))))
    arrays = {f"model.layers.0.self_attn.{name}.weight": np.asarray(
        value["kernel"].T if "kernel" in value else value["scale"]
    ) for name, value in attn.items()}
    bank = jax.tree.map(lambda x: jnp.stack([x] * 27), initialize(arrays, tc, c.mamba, 0))
    layers = jax.tree.map(lambda x: jnp.stack([x] * 27), block)
    donor = (layers, block, matrix(24, 8), norm)
    mesh = Mesh(np.array(jax.devices()), ("model",))
    assert mesh.size == 8
    rep = NamedSharding(mesh, PartitionSpec())
    donor_layout = (layout(layers, mesh), *(layout(x, mesh, stacked=False) for x in donor[1:]))
    param_layout = {"mamba": layout(bank, mesh), "qwen": donor_layout}
    p = jax.device_put(dict(mamba=bank, qwen=jax.tree.map(lambda x: x.astype(jnp.float32), donor)), param_layout)
    donor = jax.device_put(donor, donor_layout)
    tokens = jax.device_put(np.arange(64, dtype=np.int32)[None] % 24, rep)
    return c, tc, p, donor, tokens, param_layout, mesh, rep


def virtual_full_update():
    import optax
    from extent.clean_tpu_171 import layout, optimizer

    c, tc, p, donor, tokens, param_layout, mesh, rep = virtual_full_fixture()
    full = jax.jit(make_gradient(c, tc, True), out_shardings=(rep, param_layout, rep))
    loss, grad, metrics = full(p, donor, tokens)
    frozen_layout = {"mamba": param_layout["mamba"]}
    frozen = jax.jit(make_gradient(c, tc, False), out_shardings=(rep, frozen_layout, rep))
    base_loss, _, base_metrics = frozen(dict(mamba=p["mamba"]), donor, tokens)
    np.testing.assert_allclose(loss, base_loss, atol=1e-5)
    np.testing.assert_allclose(metrics, base_metrics, atol=1e-5)
    student_layers, last, embedding, final_norm = grad["qwen"]
    assert np.any(embedding)
    assert np.any(final_norm["scale"])
    assert np.any(last["mlp"]["down_proj"]["kernel"])
    q_grad = np.asarray(student_layers["self_attn"]["q_proj"]["kernel"])
    assert np.any(q_grad[3]) and not np.any(q_grad[0])
    tx = optimizer()
    state = tx.init(p)
    state_layout = layout(state, mesh, stacked=False)
    state = jax.device_put(state, state_layout)

    def update(params, state, grad, rate):
        updates, state = tx.update(grad, state, params)
        return optax.apply_updates(params, jax.tree.map(lambda x: -rate * x, updates)), state

    update = jax.jit(update, out_shardings=(param_layout, state_layout))
    new, _ = update(p, state, grad, jax.device_put(np.asarray(1e-5, np.float32), rep))
    assert np.any(np.asarray(new["qwen"][2]) != np.asarray(p["qwen"][2]))


def test_virtual_eight_device_full_update():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=8")
    script = f'import runpy; runpy.run_path({str(Path(__file__).resolve())!r})["virtual_full_update"]()'
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
