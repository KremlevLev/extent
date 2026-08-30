from extent.config import HybridConfig
from scripts.qwen_production_aligned_scaling_campaign import (
    LAYER_BUDGETS,
    LAYER_SETS,
    TARGET_LAYERS,
)
from scripts.qwen_progressive_composition_eval import analyze_exact_lift_scaling


def test_exp061_uses_nested_production_mamba_layers():
    config = HybridConfig()
    assert set(LAYER_SETS[4]) < set(LAYER_SETS[8]) < set(LAYER_SETS[12])
    assert TARGET_LAYERS == tuple(sorted(LAYER_SETS[12]))
    assert set(TARGET_LAYERS) <= set(config.mamba_layer_indices)
    assert not set(TARGET_LAYERS) & set(config.attention_layer_indices)


def test_exp061_has_matched_long_recovery_budget():
    assert LAYER_BUDGETS[0] == 8192
    assert all(
        LAYER_BUDGETS[layer] == 4096
        for layer in TARGET_LAYERS
        if layer != 0
    )


def test_scaling_gate_description_uses_actual_stage_counts():
    def metric(value):
        return {"mean_nll": value, "window_mean_nll": [value] * 16}

    metrics = {}
    for count in LAYER_SETS:
        for seed in (123, 456, 789):
            metrics[f"SEED-{seed}-COMPOSED-{count}"] = metric(5.0)
            metrics[f"SEED-{seed + 1000}-COMPOSED-{count}"] = metric(4.8)
    result = analyze_exact_lift_scaling(
        layer_sets=LAYER_SETS,
        original_metric=metric(4.0),
        composed_metrics=metrics,
        bootstrap_samples=100,
        bootstrap_seed=1,
    )
    assert "4/8/12-layer stage" in result["gate_definition"]
