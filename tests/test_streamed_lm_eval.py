from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from extent.streamed_lm_eval import (
    create_lm_metrics_runner,
    end_to_end_loss_comparison,
    full_model_shard_last_use,
    hidden_relative_l2,
    next_token_statistics,
)


class _IdentityNorm:
    def apply(self, variables, hidden):
        del variables
        return hidden


def test_full_model_shard_last_use_keeps_shared_final_shard():
    weight_map = {
        "model.embed_tokens.weight": "a.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "a.safetensors",
        "model.layers.1.mlp.up_proj.weight": "b.safetensors",
        "model.norm.weight": "b.safetensors",
        "lm_head.weight": "c.safetensors",
    }
    assert full_model_shard_last_use(weight_map, 2) == {
        "a.safetensors": 0,
        "b.safetensors": 2,
        "c.safetensors": 2,
    }


def test_next_token_statistics_matches_manual_cross_entropy_and_padding():
    logits = jnp.asarray(
        [
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            [[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [4.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    tokens = jnp.asarray([[2, 0, 1], [0, 1, 2]], dtype=jnp.int32)
    loss, count, correct = next_token_statistics(
        logits, tokens, jnp.asarray([True, False])
    )
    expected = -np.log(np.exp(4.0) / (np.exp(4.0) + 2.0)) * 2
    np.testing.assert_allclose(loss, expected, rtol=5e-6)
    assert int(count) == 2
    assert int(correct) == 2


def test_hidden_relative_l2_and_shape_validation():
    reference = np.asarray([1.0, 0.0], dtype=np.float32)
    assert hidden_relative_l2(reference, reference) == 0.0
    np.testing.assert_allclose(hidden_relative_l2(reference, np.zeros(2)), 1.0)
    with pytest.raises(ValueError, match="sequence"):
        next_token_statistics(jnp.zeros((1, 1, 2)), jnp.zeros((1, 1), jnp.int32))


def test_end_to_end_gate_uses_excess_nll_recovery():
    passed = end_to_end_loss_comparison(
        original_nll=2.0,
        calibrated_nll=2.5,
        mixer_only_nll=2.4,
        joint_nll=2.3,
        all_finite=True,
    )
    assert passed["scientific_gate_passed"] is True
    np.testing.assert_allclose(
        passed["joint_recovered_mixer_excess_fraction"], 0.25
    )
    failed = end_to_end_loss_comparison(
        original_nll=2.0,
        calibrated_nll=2.5,
        mixer_only_nll=2.4,
        joint_nll=2.39,
        all_finite=True,
    )
    assert failed["scientific_gate_passed"] is False


def test_lm_metrics_runner_reduces_logits_on_device():
    runner = create_lm_metrics_runner(_IdentityNorm())
    hidden = jnp.asarray([[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]])
    kernel = jnp.eye(2, dtype=jnp.float32)
    loss, count, correct = runner(
        {}, kernel, hidden, jnp.asarray([[1, 0, 1]]), jnp.asarray([True])
    )
    expected = -np.log(np.exp(4.0) / (np.exp(4.0) + 1.0)) * 2
    np.testing.assert_allclose(loss, expected, rtol=5e-6)
    assert int(count) == 2
    assert int(correct) == 2
