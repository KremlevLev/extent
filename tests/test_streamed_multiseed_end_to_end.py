from __future__ import annotations

from scripts.qwen_streamed_multiseed_end_to_end import (
    EXTERNAL_EVALUATION_PROTOCOLS,
    OBJECTIVE_COMPARISON_PROTOCOLS,
    PROTOCOL_LABELS,
    PROTOCOL_METHODS,
    PROTOCOL_NOTES,
    branch_names,
    collect_per_seed_lm_metrics,
    protocol_collects_window_nll,
    reference_reproduction,
)


def test_multiseed_branch_order_is_frozen():
    assert branch_names((123, 456, 789), 1024) == (
        "ORIGINAL-CACHED-QWEN",
        "SEED-123-CALIBRATED-STEP0",
        "SEED-123-MIXER-ONLY-STEP1024",
        "SEED-123-JOINT-STEP1024",
        "SEED-456-CALIBRATED-STEP0",
        "SEED-456-MIXER-ONLY-STEP1024",
        "SEED-456-JOINT-STEP1024",
        "SEED-789-CALIBRATED-STEP0",
        "SEED-789-MIXER-ONLY-STEP1024",
        "SEED-789-JOINT-STEP1024",
    )


def test_objective_metric_collection_does_not_require_calibrated_branch():
    metrics = {
        "SEED-123-MIXER-ONLY-STEP1024": {"mean_nll": 2.0},
        "SEED-123-JOINT-STEP1024": {"mean_nll": 1.8},
        "SEED-123-CONTRIBUTION-STEP1024": {"mean_nll": 1.6},
    }
    collected = collect_per_seed_lm_metrics(
        metrics,
        (123,),
        1024,
        objective_comparison=True,
    )
    assert set(collected["123"]) == {
        "mixer_only",
        "joint",
        "contribution",
    }


def test_legacy_metric_collection_keeps_calibrated_branch():
    metrics = {
        "SEED-123-CALIBRATED-STEP0": {"mean_nll": 2.2},
        "SEED-123-MIXER-ONLY-STEP1024": {"mean_nll": 2.0},
        "SEED-123-JOINT-STEP1024": {"mean_nll": 1.8},
    }
    collected = collect_per_seed_lm_metrics(
        metrics,
        (123,),
        1024,
        objective_comparison=False,
    )
    assert set(collected["123"]) == {"calibrated", "mixer_only", "joint"}


def test_every_bootstrapped_campaign_protocol_collects_window_nll():
    for protocol in (
        "exp043-validation",
        "exp045-depth-objective",
        "exp046-depth-objective",
        "exp047-context-transfer",
        "exp048-long-horizon",
        "exp049-extended-horizon",
        "exp050-depth-scaling-atlas",
        "exp051-progressive-composition",
        "exp052-boundary-scaling",
    ):
        assert protocol_collects_window_nll(protocol)
    assert not protocol_collects_window_nll("exp041")


def test_protocol_registries_are_complete_for_composition_campaigns():
    assert set(PROTOCOL_METHODS) == set(PROTOCOL_NOTES) == set(PROTOCOL_LABELS)
    assert OBJECTIVE_COMPARISON_PROTOCOLS <= EXTERNAL_EVALUATION_PROTOCOLS
    assert "exp051-progressive-composition" in OBJECTIVE_COMPARISON_PROTOCOLS
    assert "exp051-progressive-composition" in EXTERNAL_EVALUATION_PROTOCOLS
    assert "exp052-boundary-scaling" in OBJECTIVE_COMPARISON_PROTOCOLS
    assert "exp052-boundary-scaling" in EXTERNAL_EVALUATION_PROTOCOLS


def test_reference_reproduction_applies_absolute_nll_tolerance():
    reference = {
        "lm_metrics": {
            "ORIGINAL-CACHED-QWEN": {"mean_nll": 1.0},
            "CALIBRATED-STEP0": {"mean_nll": 2.0},
            "MIXER-ONLY-STEP1024": {"mean_nll": 1.8},
            "JOINT-STEP1024": {"mean_nll": 1.5},
        }
    }
    metrics = {
        "ORIGINAL-CACHED-QWEN": {"mean_nll": 1.001},
        "SEED-123-CALIBRATED-STEP0": {"mean_nll": 2.001},
        "SEED-123-MIXER-ONLY-STEP1024": {"mean_nll": 1.799},
        "SEED-123-JOINT-STEP1024": {"mean_nll": 1.501},
    }
    report = reference_reproduction(
        reference,
        metrics,
        seed=123,
        total_steps=1024,
        tolerance=0.01,
    )
    assert report["passed"] is True
    metrics["SEED-123-JOINT-STEP1024"]["mean_nll"] = 1.6
    report = reference_reproduction(
        reference,
        metrics,
        seed=123,
        total_steps=1024,
        tolerance=0.01,
    )
    assert report["passed"] is False
