from __future__ import annotations

from scripts.qwen_long_horizon_campaign import aggregate_long_horizon


def _budget_result(step: int, offset: float) -> dict:
    layers = {}
    for layer in (0, 18):
        metrics = {}
        for seed_index, seed in enumerate((123, 456, 789)):
            for arm_index, branch in enumerate(
                ("MIXER-ONLY", "JOINT", "CONTRIBUTION")
            ):
                mean = 4.0 + layer / 100 + seed_index / 1000 + arm_index / 100
                mean += offset
                metrics[f"SEED-{seed}-{branch}-STEP{step}"] = {
                    "mean_nll": mean,
                    "window_mean_nll": [mean - 0.03, mean - 0.01, mean + 0.01, mean + 0.03],
                    "finite": True,
                }
        layers[str(layer)] = {"lm_metrics": metrics, "passed": True}
    return {"layer_results": layers, "passed": True}


def test_long_horizon_aggregate_uses_paired_seeds_and_windows():
    result = aggregate_long_horizon(
        _budget_result(1024, 0.0),
        _budget_result(4096, -0.1),
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    assert result["scientific_gate_passed"]
    for layer in result["layers"]:
        joint = layer["arms"]["joint"]
        assert joint["long_wins"] == 3
        assert joint["mean_long_minus_short_nll"] < 0
        assert joint["long_minus_short_95ci"][1] < 0


def test_extended_horizon_aggregate_accepts_exp049_step_names():
    result = aggregate_long_horizon(
        _budget_result(2048, 0.0),
        _budget_result(8192, -0.05),
        bootstrap_samples=100,
        bootstrap_seed=11,
        short_step=2048,
        long_step=8192,
    )
    assert result["scientific_gate_passed"]
