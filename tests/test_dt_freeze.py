from flax.core import freeze
import jax.numpy as jnp
import numpy as np

from extent.config import tiny_config
from extent.dt_freeze import freeze_mamba_dt_gradients
from extent.mamba3_transplant import mamba3_projection_slices
from scripts import m3q_dt_freeze_campaign
import scripts.m3q_long_recoverability_campaign as base_campaign


def test_freeze_dt_zeros_only_dt_projection_and_bias():
    config = tiny_config()
    width = mamba3_projection_slices(config.hidden_size, config.mamba)["angle"].stop
    kernel = jnp.ones((config.hidden_size, width), jnp.float32)
    grads = {}
    for layer in range(config.num_layers):
        name = "mamba" if layer in config.mamba_layer_indices else "attention"
        grads[f"layers_{layer}"] = {
            name: {"in_proj": {"kernel": kernel}, "dt_bias": jnp.ones((1,))}
            if name == "mamba" else {"kernel": jnp.ones((2, 2))}
        }
    masked = freeze_mamba_dt_gradients(freeze(grads), config)
    dt = mamba3_projection_slices(config.hidden_size, config.mamba)["dt"]
    for layer in config.mamba_layer_indices:
        mixer = masked[f"layers_{layer}"]["mamba"]
        assert np.count_nonzero(np.asarray(mixer["in_proj"]["kernel"][:, dt])) == 0
        assert np.count_nonzero(np.asarray(mixer["dt_bias"])) == 0
        assert np.all(np.asarray(mixer["in_proj"]["kernel"][:, : dt.start]) == 1)


def test_exp090_entrypoint_selects_separate_resumable_contract(monkeypatch):
    captured = {}
    original = {
        name: getattr(base_campaign, name)
        for name in (
            "PROTOCOL", "HF_PREFIX", "ARMS", "CONTROL_ARM", "PRIMARY_ARM",
            "EXPERIMENT_ID", "ARTIFACT_STEM", "EXPERIMENT_LABEL",
            "FROZEN_DT_STEPS", "PRIMARY_QUESTION",
        )
    }

    def fake_main(argv):
        captured.update(
            protocol=base_campaign.PROTOCOL,
            prefix=base_campaign.HF_PREFIX,
            arms=base_campaign.ARMS,
            primary=base_campaign.PRIMARY_ARM,
            frozen_steps=base_campaign.FROZEN_DT_STEPS,
        )
        return captured

    monkeypatch.setattr(base_campaign, "main", fake_main)
    try:
        result = m3q_dt_freeze_campaign.main([])
    finally:
        for name, value in original.items():
            setattr(base_campaign, name, value)
    assert result["protocol"] == "exp090-random-dt-freeze-v1"
    assert result["prefix"] == "experiments/exp090-random-dt-freeze"
    assert result["arms"] == ("RANDOM", "RANDOM-DT-FROZEN")
    assert result["primary"] == "RANDOM-DT-FROZEN"
    assert result["frozen_steps"] == 32_768


def test_exp090_contract_records_dt_intervention():
    original = base_campaign.FROZEN_DT_STEPS
    try:
        base_campaign.FROZEN_DT_STEPS = 32_768
        contract = base_campaign.experiment_contract(tiny_config())
    finally:
        base_campaign.FROZEN_DT_STEPS = original
    assert contract["intervention"] == {"frozen_dt_steps": 32_768}
