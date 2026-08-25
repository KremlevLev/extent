from __future__ import annotations

from scripts.qwen_depth_scaling_atlas_campaign import (
    TARGET_LAYERS,
    aggregate_depth_trend,
)
from scripts.qwen_long_horizon_campaign import aggregate_long_horizon


def _budget_result(step: int, deltas: dict[int, float] | None = None) -> dict:
    layers = {}
    for layer in TARGET_LAYERS:
        metrics = {}
        delta = 0.0 if deltas is None else deltas[layer]
        for seed in (123, 456, 789):
            for branch in ("MIXER-ONLY", "JOINT", "CONTRIBUTION"):
                mean = 4.0 + layer / 100 + delta
                metrics[f"SEED-{seed}-{branch}-STEP{step}"] = {
                    "mean_nll": mean,
                    "window_mean_nll": [
                        mean - 0.03,
                        mean - 0.01,
                        mean + 0.01,
                        mean + 0.03,
                    ],
                    "finite": True,
                }
        layers[str(layer)] = {"lm_metrics": metrics, "passed": True}
    return {"layer_results": layers, "passed": True}


def test_depth_atlas_detects_positive_depth_slope():
    short = _budget_result(2048)
    long = _budget_result(8192, {6: -0.30, 12: -0.15, 29: 0.05})
    aggregate = aggregate_long_horizon(
        short,
        long,
        bootstrap_samples=100,
        bootstrap_seed=3,
        short_step=2048,
        long_step=8192,
        target_layers=TARGET_LAYERS,
    )
    result = aggregate_depth_trend(
        aggregate,
        {"short": short, "long": long},
        bootstrap_samples=100,
        bootstrap_seed=5,
        short_step=2048,
        long_step=8192,
        target_layers=TARGET_LAYERS,
    )
    trend = result["depth_trend"]["arms"]["joint"]
    assert trend["nll_delta_per_layer_index_slope"] > 0
    assert trend["slope_95ci"][0] > 0
    assert result["scientific_gate_passed"]
