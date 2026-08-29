from extent.config import HybridConfig
from scripts.qwen_production_aligned_scaling_campaign import (
    LAYER_BUDGETS,
    LAYER_SETS,
    TARGET_LAYERS,
)


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
