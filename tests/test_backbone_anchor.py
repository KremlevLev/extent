from flax.core import freeze
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from extent.backbone_anchor import scale_copied_backbone_updates
from extent.config import tiny_config
from extent.full_model_distillation import make_prediction_distill_step
from scripts import m3q_long_recoverability_campaign as campaign
from scripts import m3q_onpolicy_anchor_campaign as exp091


def test_lion_updates_are_scaled_only_after_optimizer_for_copied_weights():
    updates = freeze({
        "embed_tokens": {"embedding": jnp.array([2.0])},
        "layers_0": {
            "mamba": {"in_proj": {"kernel": jnp.array([3.0])}},
            "mlp": {"up_proj": {"kernel": jnp.array([4.0])}},
        },
    })
    scaled = scale_copied_backbone_updates(updates, scale=0.03)
    np.testing.assert_allclose(scaled["embed_tokens"]["embedding"], [0.06])
    np.testing.assert_allclose(scaled["layers_0"]["mamba"]["in_proj"]["kernel"], [3.0])
    np.testing.assert_allclose(scaled["layers_0"]["mlp"]["up_proj"]["kernel"], [0.12])
    assert type(scaled) is type(updates)


def test_invalid_update_scale_fails_before_tpu_work():
    with pytest.raises(ValueError):
        scale_copied_backbone_updates({"x": jnp.array(1.0)}, scale=1.1)


def test_exp091_contract_has_equal_budget_and_pinned_warm_start(monkeypatch):
    monkeypatch.setattr(campaign, "PROTOCOL", exp091.campaign.PROTOCOL)
    monkeypatch.setattr(campaign, "WARM_START_ARMS", ("ONPOLICY-PLAIN", "ONPOLICY-ANCHOR"))
    monkeypatch.setattr(campaign, "BACKBONE_UPDATE_SCALE", {"ONPOLICY-ANCHOR": 0.03})
    monkeypatch.setattr(campaign, "ARMS", ("ONPOLICY-PLAIN", "ONPOLICY-ANCHOR"))
    monkeypatch.setattr(campaign, "TOTAL_STEPS", 24_576)
    monkeypatch.setattr(campaign, "TOKENS_PER_TRAJECTORY", 24_576 * 256)
    monkeypatch.setattr(campaign, "CHECKPOINTS", (0, 3_072, 8_192, 16_384, 24_576))
    contract = campaign.experiment_contract(tiny_config())
    assert contract["warm_start"]["source"].startswith("EXP-072-v2")
    assert contract["warm_start"]["backbone_update_scale"]["ONPOLICY-ANCHOR"] == 0.03
    assert contract["tokens_per_trajectory"] == 24_576 * 256
    assert contract["arms"] == ["ONPOLICY-PLAIN", "ONPOLICY-ANCHOR"]


def test_update_hook_changes_parameter_step_not_optimizer_state():
    params = {"lm_head": {"kernel": jnp.asarray(0.2, jnp.float32)}}
    tokens = jnp.asarray([[0, 1, 0]], jnp.int32)

    def student_apply(p, ids, _):
        value = p["lm_head"]["kernel"]
        return jnp.broadcast_to(jnp.stack([value, -value]), (*ids.shape, 2)), ()

    def teacher_apply(_, ids, __):
        return jnp.zeros((*ids.shape, 2), jnp.float32), ()

    tx = optax.sgd(0.1)
    normal = make_prediction_distill_step(
        student_apply, teacher_apply, tx, temperature=2.0,
        cross_entropy_weight=0.1, bf16_gradients=False,
    )
    anchored = make_prediction_distill_step(
        student_apply, teacher_apply, tx, temperature=2.0,
        cross_entropy_weight=0.1, bf16_gradients=False,
        update_transform=lambda updates: scale_copied_backbone_updates(
            updates, scale=0.03
        ),
    )
    base, _, _ = normal(params, tx.init(params), {}, tokens)
    slow, _, _ = anchored(params, tx.init(params), {}, tokens)
    baseline_move = float(base["lm_head"]["kernel"] - params["lm_head"]["kernel"])
    anchored_move = float(slow["lm_head"]["kernel"] - params["lm_head"]["kernel"])
    assert np.isclose(anchored_move / baseline_move, 0.03, rtol=1e-4)


def test_source_manifest_preflight_reports_missing_before_model_loading(monkeypatch):
    class Hub:
        repo_id = "test/repo"
        repo_type = "dataset"
        revision = "main"
        token = "test-token"

    class Api:
        def list_repo_files(self, **kwargs):
            assert kwargs["repo_id"] == "test/repo"
            return []

    monkeypatch.setattr(exp091, "artifact_config_from_env", lambda: Hub())
    with pytest.raises(FileNotFoundError, match="source manifests"):
        exp091.preflight_source_endpoints(api=Api())
