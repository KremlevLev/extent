import jax.numpy as jnp
import numpy as np

from extent.readout_calibration import fit_dual_ridge_readout
from scripts.qwen_mamba3_readout_probe import _fit_residualized_context_readout


def test_dual_ridge_readout_recovers_heldout_linear_signal():
    rng = np.random.default_rng(61)
    kernel = rng.normal(size=(12, 7)).astype(np.float32)
    features = rng.normal(size=(48, 12)).astype(np.float32)
    targets = features @ kernel
    fitted, report = fit_dual_ridge_readout(
        jnp.asarray(features[:32]), jnp.asarray(targets[:32]), relative_ridge=1e-6
    )
    predicted = features[32:] @ np.asarray(fitted)
    relative_l2 = np.linalg.norm(predicted - targets[32:]) / np.linalg.norm(targets[32:])
    assert relative_l2 < 1e-4
    assert report.calibration_relative_l2 < 1e-4
    assert report.samples == 32


def test_residualized_context_readout_adds_heldout_signal():
    rng = np.random.default_rng(62)
    raw = rng.normal(size=(64, 6)).astype(np.float32)
    context = rng.normal(size=(64, 5)).astype(np.float32)
    target = raw @ rng.normal(size=(6, 4)).astype(np.float32)
    target += context @ rng.normal(size=(5, 4)).astype(np.float32)
    raw_kernel, context_kernel, report = _fit_residualized_context_readout(
        jnp.asarray(raw),
        jnp.asarray(context),
        jnp.asarray(target),
        calibration_tokens=48,
        selection_tokens=12,
        ridge_values=(1e-3, 1e-5),
        raw_ridge=1e-5,
    )
    prediction = raw[48:] @ np.asarray(raw_kernel)
    prediction += context[48:] @ np.asarray(context_kernel)
    relative_l2 = np.linalg.norm(prediction - target[48:]) / np.linalg.norm(target[48:])
    raw_only, _ = fit_dual_ridge_readout(
        jnp.asarray(raw[:48]), jnp.asarray(target[:48]), relative_ridge=1e-5
    )
    raw_relative_l2 = np.linalg.norm(raw[48:] @ np.asarray(raw_only) - target[48:])
    raw_relative_l2 /= np.linalg.norm(target[48:])
    assert relative_l2 < raw_relative_l2
    assert report["selected_context_relative_ridge"] in (1e-3, 1e-5)
