from scripts.qwen_exact_lift_scaling_campaign import LAYER_BUDGETS, LAYER_SETS, TARGET_LAYERS
from scripts.qwen_progressive_composition_eval import analyze_exact_lift_scaling


def _metric(value):
    return {"mean_nll": value, "window_mean_nll": [value] * 32}


def test_scaling_protocol_uses_nested_2_4_8_sets_and_long_layer0():
    assert LAYER_SETS[2] == (0, 18)
    assert set(LAYER_SETS[2]) < set(LAYER_SETS[4]) < set(LAYER_SETS[8])
    assert TARGET_LAYERS == tuple(sorted(LAYER_SETS[8]))
    assert LAYER_BUDGETS[0] == 8192
    assert all(LAYER_BUDGETS[layer] == 4096 for layer in TARGET_LAYERS if layer != 0)


def test_scaling_gate_requires_every_stage_with_paired_ci():
    metrics = {}
    for count in (2, 4, 8):
        for seed in (123, 456, 789):
            metrics[f"SEED-{seed}-COMPOSED-{count}"] = _metric(5.0 + count / 10)
            metrics[f"SEED-{seed + 1000}-COMPOSED-{count}"] = _metric(4.8 + count / 10)
    result = analyze_exact_lift_scaling(
        layer_sets=LAYER_SETS, original_metric=_metric(4.0),
        composed_metrics=metrics, bootstrap_samples=200, bootstrap_seed=9,
    )
    assert result["scientific_gate_passed"]
    assert len(result["stages"]) == 3
    assert all(stage["wins"] == 3 for stage in result["stages"])
