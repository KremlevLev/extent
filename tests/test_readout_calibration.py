import jax.numpy as jnp
import numpy as np

from singularity.readout_calibration import fit_dual_ridge_readout


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
