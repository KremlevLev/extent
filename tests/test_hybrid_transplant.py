import pytest

from extent.config import HybridConfig
from extent.hybrid_transplant import (
    MAMBA_TRANSPLANT_METHOD,
    MLA_TRANSPLANT_METHOD,
    build_hybrid_transplant_plan,
)


def test_production_plan_covers_every_source_and_layer_once():
    plan = build_hybrid_transplant_plan(HybridConfig())

    assert plan.attention_layer_indices == (5, 12, 19, 25, 32, 39)
    assert len(plan.mamba_layer_indices) == 34
    assert plan.attention_fraction == 0.15
    assert [action.layer_index for action in plan.actions] == list(range(40))
    assert plan.direct_tensor_count == 203
    assert plan.mixer_tensor_count == 240
    assert plan.source_tensor_count == 443
    assert plan.source_coverage_fraction == 1.0

    for action in plan.actions:
        if action.layer_index in plan.attention_layer_indices:
            assert action.method == MLA_TRANSPLANT_METHOD
            assert action.requires_calibration
        else:
            assert action.method == MAMBA_TRANSPLANT_METHOD
            assert not action.requires_calibration
        assert len(action.source_tensors) == 6


def test_cache_accounting_matches_locked_exp019_recipe():
    plan = build_hybrid_transplant_plan(HybridConfig())

    assert plan.source_gqa_cache_elements_per_token_per_layer == 2048
    assert plan.target_mla_cache_elements_per_token_per_layer == 576
    assert plan.retained_attention_cache_reduction_fraction == 0.71875
    assert plan.source_retained_attention_cache_elements_per_token == 12_288
    assert plan.target_retained_attention_cache_elements_per_token == 3_456


def test_production_mla_contract_matches_exp019():
    plan = build_hybrid_transplant_plan(HybridConfig())

    assert plan.architecture_ready
    assert plan.blockers == ()


def test_plan_refuses_non_15_percent_layout():
    with pytest.raises(ValueError, match="exactly 6 attention"):
        build_hybrid_transplant_plan(
            HybridConfig(attention_layer_indices=(4, 9, 14, 19, 24, 29, 34, 39))
        )
