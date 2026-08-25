from __future__ import annotations

import numpy as np

from scripts.qwen_extended_horizon_campaign import main as run_extended_campaign
from scripts.qwen_long_horizon_campaign import ARM_BRANCHES, SEEDS


TARGET_LAYERS = (6, 12, 29)
BUDGETS = {
    "short": {
        "experiment": "exp050-short",
        "steps": 2048,
        "checkpoints": "0,1024,2048",
        "minimum_free_gib": 6,
    },
    "long": {
        "experiment": "exp050-long",
        "steps": 8192,
        "checkpoints": "0,2048,4096,8192",
        "minimum_free_gib": 11,
    },
}


def aggregate_depth_trend(
    aggregate: dict,
    budget_results: dict,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    short_step: int,
    long_step: int,
    target_layers: tuple[int, ...],
) -> dict:
    depths = np.asarray(target_layers, dtype=np.float64)
    trends = {}
    for arm_index, (arm, branch) in enumerate(ARM_BRANCHES.items()):
        layer_deltas = []
        for layer in target_layers:
            short_layer = budget_results["short"]["layer_results"][str(layer)]
            long_layer = budget_results["long"]["layer_results"][str(layer)]
            seed_deltas = []
            for seed in SEEDS:
                short_windows = np.asarray(
                    short_layer["lm_metrics"][
                        f"SEED-{seed}-{branch}-STEP{short_step}"
                    ]["window_mean_nll"],
                    dtype=np.float64,
                )
                long_windows = np.asarray(
                    long_layer["lm_metrics"][
                        f"SEED-{seed}-{branch}-STEP{long_step}"
                    ]["window_mean_nll"],
                    dtype=np.float64,
                )
                if short_windows.shape != long_windows.shape:
                    raise ValueError("depth atlas requires paired window shapes")
                seed_deltas.append(long_windows - short_windows)
            layer_deltas.append(np.stack(seed_deltas))
        paired = np.stack(layer_deltas)
        if not np.all(np.isfinite(paired)):
            raise ValueError("depth atlas contains non-finite paired deltas")
        mean_by_depth = np.mean(paired, axis=(1, 2))
        slope = float(np.polyfit(depths, mean_by_depth, 1)[0])
        rng = np.random.default_rng(bootstrap_seed + arm_index)
        draws = np.empty(bootstrap_samples, dtype=np.float64)
        for sample in range(bootstrap_samples):
            indices = rng.integers(0, paired.shape[2], paired.shape[2])
            sampled_means = np.mean(paired[:, :, indices], axis=(1, 2))
            draws[sample] = float(np.polyfit(depths, sampled_means, 1)[0])
        interval = [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ]
        trends[arm] = {
            "mean_long_minus_short_nll_by_layer": {
                str(layer): float(value)
                for layer, value in zip(target_layers, mean_by_depth)
            },
            "nll_delta_per_layer_index_slope": slope,
            "slope_95ci": interval,
            "positive_slope": slope > 0,
            "positive_slope_95ci": interval[0] > 0,
        }

    layer_lookup = {
        int(layer["target_layer"]): layer for layer in aggregate["layers"]
    }
    early_joint = layer_lookup[min(target_layers)]["arms"]["joint"]
    early_benefit = bool(
        early_joint["mean_long_minus_short_nll"] < 0
        and early_joint["long_wins"] >= 2
        and early_joint["long_minus_short_95ci"][1] < 0
    )
    joint_trend = trends["joint"]
    depth_gate = bool(
        aggregate["all_finite"]
        and early_benefit
        and joint_trend["positive_slope_95ci"]
    )
    aggregate["depth_trend"] = {
        "arms": trends,
        "primary_arm": "joint",
        "earliest_tested_layer": min(target_layers),
        "early_long_budget_benefit_required": True,
        "early_long_budget_benefit_passed": early_benefit,
        "positive_slope_95ci_required": True,
        "scientific_gate_passed": depth_gate,
    }
    aggregate["scientific_gate_passed"] = depth_gate
    return aggregate


EXP050_CONFIG = {
    "experiment_id": "exp050-depth-scaling-atlas",
    "display_name": "EXP-050 depth-scaling atlas",
    "method": "multidepth_paired_recovery_budget_scaling_atlas",
    "protocol": "exp050-short2048-vs-long8192-layers6-12-29",
    "target_layers": TARGET_LAYERS,
    "budgets": BUDGETS,
    "stage_manifest": "exp050-campaign-stage-manifest.json",
    "failure_json": "extent-depth-scaling-atlas-failure.json",
    "default_result_json": (
        "/kaggle/working/output/extent-depth-scaling-atlas.json"
    ),
    "default_qwen_cache_dir": "/kaggle/working/qwen3-exp050-weights",
    "default_bootstrap_seed": 20260829,
    "aggregate_transform": aggregate_depth_trend,
}


def main(argv: list[str] | None = None) -> dict:
    return run_extended_campaign(argv, campaign_config=EXP050_CONFIG)


if __name__ == "__main__":
    main()
