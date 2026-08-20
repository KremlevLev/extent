from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from extent.offline_distillation import (
    deterministic_batch_indices,
    restore_offline_checkpoint,
    save_offline_checkpoint,
)


def test_deterministic_batches_cover_epochs_and_resume_exactly():
    batches = [deterministic_batch_indices(step, 2, 5, 17) for step in range(5)]
    stream = np.concatenate(batches)
    assert sorted(stream[:5].tolist()) == list(range(5))
    assert sorted(stream[5:10].tolist()) == list(range(5))
    np.testing.assert_array_equal(
        deterministic_batch_indices(3, 2, 5, 17), batches[3]
    )
    with pytest.raises(ValueError, match="positive"):
        deterministic_batch_indices(0, 0, 5, 17)


def test_offline_checkpoint_round_trip_and_contract_gate(tmp_path):
    params = {"kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)}
    tx = optax.adam(1e-3)
    opt_state = tx.init(params)
    compatibility = {"source": "teacher@revision", "total_steps": 100}
    metadata = save_offline_checkpoint(
        tmp_path, params, opt_state, step=7, compatibility=compatibility
    )
    restored_params, restored_state, step, restored_metadata = (
        restore_offline_checkpoint(
            tmp_path,
            {"kernel": jnp.zeros((2, 3), jnp.float32)},
            tx.init({"kernel": jnp.zeros((2, 3), jnp.float32)}),
            expected_compatibility=compatibility,
        )
    )
    np.testing.assert_array_equal(restored_params["kernel"], params["kernel"])
    assert jax_tree_shapes(restored_state) == jax_tree_shapes(opt_state)
    assert step == 7
    assert restored_metadata["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert json.loads((tmp_path / "checkpoint.json").read_text())["step"] == 7
    with pytest.raises(ValueError, match="compatibility"):
        restore_offline_checkpoint(
            tmp_path,
            params,
            opt_state,
            expected_compatibility={"source": "wrong", "total_steps": 100},
        )


def test_split_run_matches_uninterrupted_optimizer_trajectory(tmp_path):
    tx = optax.lion(learning_rate=1e-2)
    initial = {"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}

    def update(params, state, step):
        indices = deterministic_batch_indices(step, 2, 5, 41)
        target = jnp.asarray(indices, dtype=jnp.float32)
        grads = {"weight": params["weight"] - target}
        updates, state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates), state

    uninterrupted_params = initial
    uninterrupted_state = tx.init(uninterrupted_params)
    for step in range(4):
        uninterrupted_params, uninterrupted_state = update(
            uninterrupted_params, uninterrupted_state, step
        )

    split_params = initial
    split_state = tx.init(split_params)
    for step in range(2):
        split_params, split_state = update(split_params, split_state, step)
    compatibility = {"data_seed": 41, "total_steps": 4}
    save_offline_checkpoint(
        tmp_path,
        split_params,
        split_state,
        step=2,
        compatibility=compatibility,
    )
    split_params, split_state, restored_step, _ = restore_offline_checkpoint(
        tmp_path,
        initial,
        tx.init(initial),
        expected_compatibility=compatibility,
    )
    for step in range(restored_step, 4):
        split_params, split_state = update(split_params, split_state, step)

    np.testing.assert_array_equal(
        split_params["weight"], uninterrupted_params["weight"]
    )
    split_leaves = jax_tree_leaves(split_state)
    uninterrupted_leaves = jax_tree_leaves(uninterrupted_state)
    for split, uninterrupted in zip(split_leaves, uninterrupted_leaves, strict=True):
        np.testing.assert_array_equal(split, uninterrupted)


def jax_tree_shapes(tree):
    import jax

    return jax.tree.map(lambda value: getattr(value, "shape", None), tree)


def jax_tree_leaves(tree):
    import jax

    return jax.tree.leaves(tree)
